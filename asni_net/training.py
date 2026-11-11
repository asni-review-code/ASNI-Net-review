"""ASNI-only training with fixed sampling, validation selection and full resume."""
import copy
import os
from pathlib import Path
import time
import torch
import torch.nn.functional as F
import yaml
from .data import read_rows,verify_data,PROJECT
from .transforms import LesionDataset,small_threshold
from .model import ASNINet
from .runtime import UpdatePlan,loader,optimizer_update,make_scheduler,autocast
from .io import seed_everything,random_state,restore_random_state,environment,source_hashes,sha256,write_json,write_csv
from .evaluation import evaluate_model

def segmentation_loss(output,target,valid,training,total_pixels=None,total_images=None):
    z=F.interpolate(output['logits'].float(),target.shape[-2:],mode='bilinear',align_corners=False)
    bce=(F.binary_cross_entropy_with_logits(z,target.float(),reduction='none')*valid).sum()
    bce=bce/(total_pixels if total_pixels is not None else valid.sum().clamp_min(1))
    p,g=z.sigmoid()*valid,target*valid
    dims=(1,2,3)
    dice=(1-(2*(p*g).sum(dims)+1e-6)/(p.sum(dims)+g.sum(dims)+1e-6)).sum()
    dice=dice/(total_images if total_images is not None else target.shape[0])
    return training['bce_weight']*bce+training['dice_weight']*dice

def early_stop(history,updates_per_epoch,policy):
    anchor=None;anchor_step=0;last=0
    for entry in history:
        if entry['val_dice'] is None:continue
        last=entry['step'];value=entry['val_dice']
        if anchor is None or value>anchor+policy['min_delta']:
            anchor=value;anchor_step=last
    return bool(last>=policy['min_epochs']*updates_per_epoch and last-anchor_step>=policy['patience_epochs']*updates_per_epoch)

def validate_config(config):
    if config.get('model')!='ASNI-Net' or config['dataset'] not in ('bus_bra','tn3k','mmotu'):
        raise ValueError('Only ASNI-Net and the three internal datasets can be trained')
    t=config['training']
    if config['input_size']%32 or config['input_size']<32:
        raise ValueError('Input size must be a positive multiple of 32')
    if t['batch_size']<2 or min(t['epochs'],t['accumulation_steps'],t['validate_every'],t['checkpoint_every'])<1:
        raise ValueError('Invalid batch, budget or checkpoint interval')
    if t['optimizer']!='adamw' or t['loss']!='bce_dice' or t['scheduler']!='cosine':
        raise ValueError('This release implements the formal AdamW/BCE+Dice/cosine recipe')
    if config['initialization']!='scratch_exact' or config['determinism']!='strict':
        raise ValueError('Unsupported initialization or determinism mode')
    if config['runtime']['precision'] not in ('fp32','bf16'):
        raise ValueError('Use paper BF16 or explicitly selected CPU FP32 smoke validation')

def train(config,data_root,output,device='cuda',resume=None,stop_after=None,smoke=False):
    config=copy.deepcopy(config);validate_config(config)
    if not smoke and config['input_size']!=512:
        raise ValueError('Formal training requires input_size 512; use --smoke for pipeline tests')
    if stop_after is not None and stop_after<1:raise ValueError('stop_after must be positive')
    device=torch.device(device)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(config['cpu_threads']);seed_everything(config['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    runtime=config['runtime'];t=config['training'];autocast(device,runtime['precision'])
    split_path=Path(config.get('split_file',f"splits/{config['dataset']}.csv"))
    if not split_path.is_absolute():split_path=PROJECT/split_path
    rows=read_rows(config['dataset'],split_path,verify=not smoke)
    verify_data([r for r in rows if r['role'] in ('train','val')],data_root)
    data=LesionDataset(rows,'train',config['input_size'],data_root,augmentation=config['augmentation'])
    plan=UpdatePlan(len(data),t['batch_size'],t['accumulation_steps'],config['seed'],0)
    total=t['epochs']*plan.updates_per_epoch
    if t['warmup_steps']>=total:raise ValueError('Warmup exceeds training budget')
    output=Path(output).resolve();checkpoint_dir=output/'checkpoints'
    signature={'config':config,'split_sha256':sha256(split_path),'source':source_hashes(),'smoke':smoke}
    if not resume:output.mkdir(parents=True,exist_ok=False)
    elif not output.is_dir():raise ValueError('Resume requires the existing output directory')
    checkpoint_dir.mkdir(exist_ok=True)
    model=ASNINet().to(device)
    optimizer=torch.optim.AdamW([{'params':list(model.parameters()),'group_name':'all'}],lr=t['lr'],
                                weight_decay=t['weight_decay'],betas=tuple(t['betas']),eps=t['eps'])
    scheduler=make_scheduler(optimizer,t,total)
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    history=[];step=0;best=-1.;best_step=None;seen=0
    cutoff=small_threshold(rows)
    if resume:
        # Resume only files produced by this trainer and trusted by the operator.
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        if saved['signature']!=signature:raise ValueError('Resume source/configuration/split mismatch')
        if saved.get('finished'):raise ValueError('Training already finished')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']);scaler.load_state_dict(saved['scaler'])
        history=saved['history'];step=saved['step'];best=saved['best'];best_step=saved['best_step'];seen=saved['samples_seen']
        restore_random_state(saved['rng'])
    else:
        (output/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False),'utf-8')
        write_json(output/'environment.json',environment());write_json(output/'signature.json',signature)
        write_json(output/'budget.json',{'updates':total,'updates_per_epoch':plan.updates_per_epoch,
                   'dropped_tail_images_per_epoch':len(data)%t['batch_size'],
                   'physical_batch_size':t['batch_size'],'accumulation_steps':t['accumulation_steps'],
                   'validation_precision':'fp32','smoke_test':smoke})
    def save(finished=False):
        state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
               'scaler':scaler.state_dict(),'signature':signature,'history':history,'step':step,
               'best':best,'best_step':best_step,'samples_seen':seen,'rng':random_state(),'finished':finished}
        tmp=checkpoint_dir/'last.pt.tmp';torch.save(state,tmp);tmp.replace(checkpoint_dir/'last.pt')
        write_csv(output/'history.csv',history)
    target=min(total,step+stop_after) if stop_after else total
    plan.total_updates=target;plan.start_update=step
    batches=loader(data,config,sampler=plan,training=True);iterator=iter(batches)
    stopped=False;start=time.perf_counter()
    try:
        while step<target:
            group=[next(iterator) for _ in range(plan.microbatches(step))]
            model.train();lr=optimizer.param_groups[0]['lr']
            stats=optimizer_update(model,optimizer,group,t,runtime,scaler)
            scheduler.step();step+=1;seen+=stats['effective_batch']
            entry={'step':step,'epoch':step/plan.updates_per_epoch,'lr':lr,**stats,'val_dice':None,'val_iou':None}
            if step%t['validate_every']==0 or step==total:
                _,summary=evaluate_model(model,rows,'val',config,data_root,cutoff,lightweight=True)
                entry['val_dice']=summary['positive']['dice']['mean'];entry['val_iou']=summary['positive']['iou']['mean']
                if entry['val_dice']>best:
                    best=entry['val_dice'];best_step=step
                    tmp=checkpoint_dir/'best.pt.tmp';torch.save(model.state_dict(),tmp);tmp.replace(checkpoint_dir/'best.pt')
                    write_json(checkpoint_dir/'best.json',{'step':step,'epoch':step/plan.updates_per_epoch,'validation_dice':best,'seed':config['seed']})
            history.append(entry)
            if entry['val_dice'] is not None:stopped=early_stop(history,plan.updates_per_epoch,config['early_stopping'])
            if step%t['checkpoint_every']==0 or entry['val_dice'] is not None or step==target or stopped:
                save(stopped or step==total)
                print(f'update {step}/{total}; loss={stats["loss"]:.6f}; val_dice={entry["val_dice"]}',flush=True)
            if stopped:break
    finally:
        if hasattr(iterator,'_shutdown_workers'):iterator._shutdown_workers()
    status={'finished':stopped or step==total,'stop_reason':'validation_plateau' if stopped else ('max_epochs' if step==total else 'paused'),
            'actual_end_epoch':step/plan.updates_per_epoch,'best_epoch':best_step/plan.updates_per_epoch if best_step else None,
            'best_val_dice':best,'step':step,'seconds_this_session':time.perf_counter()-start,'smoke_test':smoke}
    write_json(output/'status.json',status)
    return status
