"""Standalone ASNI descriptive statistics, mechanism inspection and timing."""
import csv
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .model import ASNINet
from .layers import AxisSampler,NodeInteraction,SamplingInteractionPath
from .transforms import prepare_image,restore_logits
from .metrics import grouped_values
from .io import load_weights,write_json,write_csv,environment,sha256

def summarize_seeds(files):
    grouped={}
    metrics=('dice','iou','precision','recall','hd95_penalized_px','asd_penalized_px')
    seen=set()
    for file in files:
        with Path(file).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        if not rows:raise ValueError('Empty results')
        ids={(r['dataset'],r['seed']) for r in rows}
        if len(ids)!=1 or ids&seen:raise ValueError('Provide one table per unique dataset/seed')
        seen|=ids;dataset,seed=next(iter(ids))
        units='group_id' if dataset in ('bus_bra','bus_uclm') else 'sample_id'
        positive=[r for r in rows if int(r['gt_positive'])]
        record={'seed':int(seed)}
        for metric in metrics:
            values=grouped_values(positive,metric,units)
            record[metric]=float(np.mean(list(values.values()))) if values else None
        grouped.setdefault(dataset,[]).append(record)
    result={}
    for dataset,runs in grouped.items():
        result[dataset]={'n_seeds':len(runs),'per_seed':sorted(runs,key=lambda r:r['seed']),'metrics':{}}
        for metric in metrics:
            values=[r[metric] for r in runs if r[metric] is not None]
            if values:
                result[dataset]['metrics'][metric]={'mean':float(np.mean(values)),
                  'sample_sd':float(np.std(values,ddof=1)) if len(values)>1 else None,
                  'range':float(np.ptp(values))}
    return result

@torch.inference_mode()
def predict(weights,image_path,output,device='cpu',size=512):
    torch.set_num_threads(2)
    model=load_weights(weights,ASNINet()).to(device).eval()
    with Image.open(image_path) as im:x,geo=prepare_image(im.convert('RGB'),size)
    z=model(x[None].to(device))['logits']
    probability=restore_logits(z,geo).sigmoid()[0,0].cpu().numpy()
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    Image.fromarray((probability>=.5).astype(np.uint8)*255).save(output/'mask.png')
    np.save(output/'probability.npy',probability)
    write_json(output/'prediction.json',{'weights_sha256':sha256(weights),'threshold':.5,'geometry':geo})

@torch.inference_mode()
def inspect(weights,image_path,output,device='cpu',size=512):
    torch.set_num_threads(2)
    model=load_weights(weights,ASNINet()).to(device).eval()
    arrays={};stats={};handles=[];shapes={}
    def sampler_hook(name):
        def hook(module,args,result):
            tokens,h,w=args;shapes[name.split('.sampling.')[0]]=(h,w)
            delta=module.offset_projection(module.offset_norm(tokens.float()))[0]
            field=delta.transpose(0,1).reshape(-1,h,w)
            length=h if module.axis==2 else w
            grid=torch.arange(length,device=field.device).reshape((1,h,1) if module.axis==2 else (1,1,w))
            coord=grid+field
            values=field.abs().cpu().numpy()
            key=name.replace('.','_')
            arrays[key+'_channel_mean_offset']=field.mean(0).cpu().numpy()
            arrays[key+'_channel_mean_abs_offset']=field.abs().mean(0).cpu().numpy()
            stats[name]={'axis':'height' if module.axis==2 else 'width','unit':'feature-grid spacing',
                         'absolute_p50_p95_p99':np.quantile(values,[.5,.95,.99]).tolist(),
                         'outside_coordinate_fraction':float(((coord<0)|(coord>length-1)).float().mean()),
                         'population':'all feature grid sites, including letterbox padding'}
        return hook
    def assignment_hook(name):
        def hook(module,args,result):
            key=name.replace('.','_')
            arrays[key+'_A_NxK']=result[0].float().softmax(0).cpu().numpy()
            arrays[key+'_B_NxK']=result[0].float().softmax(-1).cpu().numpy()
        return hook
    def feature_hook(name):
        def hook(module,args,result):
            key=name.replace('.','_')
            arrays[key+'_channel_rms']=result[0].float().square().mean(-1).sqrt().cpu().numpy()
        return hook
    for name,module in model.named_modules():
        if isinstance(module,AxisSampler):handles.append(module.register_forward_hook(sampler_hook(name)))
        if isinstance(module,NodeInteraction):
            handles.append(module.assignment.register_forward_hook(assignment_hook(name)))
            handles.append(module.register_forward_hook(feature_hook(name)))
        if isinstance(module,SamplingInteractionPath):handles.append(module.register_forward_hook(feature_hook(name)))
    try:
        with Image.open(image_path) as im:x,geo=prepare_image(im.convert('RGB'),size)
        logits=model(x[None].to(device))['logits']
        arrays['probability']=restore_logits(logits,geo).sigmoid()[0,0].cpu().numpy()
    finally:
        for handle in handles:handle.remove()
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'features.npz',**arrays)
    write_json(output/'inspection.json',{'offset_statistics':stats,'stage_shapes':shapes,'geometry':geo,
               'weights_sha256':sha256(weights),'node_count':16,
               'interpretation':'Descriptive single-image observations; nodes have no assigned anatomical labels.'})

@torch.inference_mode()
def benchmark(weights,output,device='cpu',warmup=50,repeats=5,iterations=200):
    if min(repeats,iterations)<1 or warmup<0:raise ValueError('Invalid timing counts')
    device=torch.device(device);torch.set_num_threads(4)
    torch.use_deterministic_algorithms(False);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=load_weights(weights,ASNINet()).to(device).eval()
    x=torch.zeros(1,3,512,512,device=device)
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    for _ in range(warmup):model(x)
    sync()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    rows=[]
    for repeat in range(repeats):
        for iteration in range(iterations):
            sync();start=time.perf_counter();result=model(x);sync()
            elapsed=1000*(time.perf_counter()-start)
            rows.append({'repeat':repeat+1,'iteration':iteration+1,'milliseconds':elapsed})
            del result
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    values=[r['milliseconds'] for r in rows]
    summary={'parameters':sum(p.numel() for p in model.parameters()),'forward_p50_ms':float(np.median(values)),
             'forward_p95_ms':float(np.quantile(values,.95)),'environment':environment(),
             'peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
             'protocol':{'size':512,'batch_size':1,'precision':'fp32','warmup':warmup,'repeats':repeats,
                         'iterations':iterations,'threads':4,'tf32':False,'compile':False},
             'scope':'Standalone forward-only benchmark. Excludes input loading, postprocessing and other models. Memory lifetime differs from the historical multi-model profiler.'}
    write_csv(output/'timings.csv',rows);write_json(output/'summary.json',summary)
    return summary
