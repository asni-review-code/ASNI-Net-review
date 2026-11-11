import math
import torch
from torch.utils.data import DataLoader, Sampler
from .transforms import collate
from .io import random_state,restore_random_state

class UpdatePlan(Sampler):
    """Reconstruct draws from committed updates, never from a prefetched cursor."""
    def __init__(self, size, batch, accumulation, seed, total_updates, start_update=0):
        self.size,self.batch,self.accumulation,self.seed=size,batch,accumulation,seed
        self.batches_per_epoch=size//batch
        if self.batches_per_epoch<1:raise ValueError('Training subset smaller than physical batch')
        self.updates_per_epoch=math.ceil(self.batches_per_epoch/accumulation)
        self.total_updates,self.start_update=total_updates,start_update

    def microbatches(self, update):
        offset=(update%self.updates_per_epoch)*self.accumulation
        return min(self.accumulation,self.batches_per_epoch-offset)

    def __iter__(self):
        cached_epoch,permutation=None,None
        for update in range(self.start_update,self.total_updates):
            epoch=update//self.updates_per_epoch
            if cached_epoch!=epoch:
                generator=torch.Generator().manual_seed(self.seed+epoch)
                permutation=torch.randperm(self.size,generator=generator).tolist()
                cached_epoch=epoch
            start_batch=(update%self.updates_per_epoch)*self.accumulation
            for micro in range(self.microbatches(update)):
                offset=(start_batch+micro)*self.batch
                yield [(idx,(self.seed*1000003+epoch*self.size+offset+i)%(2**63-1))
                       for i,idx in enumerate(permutation[offset:offset+self.batch])]

    def __len__(self):
        return sum(self.microbatches(i) for i in range(self.start_update,self.total_updates))


def loader(dataset, config, *, sampler=None, training=False):
    runtime=config.get('runtime',{})
    workers=runtime.get('workers',0)
    options={'num_workers':workers,'pin_memory':runtime.get('pin_memory',True) and torch.cuda.is_available(),
             'collate_fn':collate,'generator':torch.Generator().manual_seed(config['seed']+17)}
    if workers:
        options.update(prefetch_factor=runtime.get('prefetch_factor',2),persistent_workers=True,
                       multiprocessing_context='spawn')
    if sampler is not None:options['batch_sampler']=sampler
    else:options.update(batch_size=runtime.get('validation_batch_size',1),shuffle=False)
    return DataLoader(dataset,**options)


def autocast(device, precision):
    device=torch.device(device)
    if precision not in ('fp32','bf16','fp16'):raise ValueError('Unsupported precision')
    if precision!='fp32' and device.type!='cuda':raise ValueError('Mixed precision requires CUDA in this runner')
    if precision=='bf16' and not torch.cuda.is_bf16_supported():raise ValueError('GPU does not support BF16')
    return torch.autocast(device.type,dtype=torch.bfloat16 if precision=='bf16' else torch.float16,enabled=precision!='fp32')


def optimizer_update(model, optimizer, batches, training, runtime, scaler):
    from .training import segmentation_loss
    from .model import forward_model
    device=next(model.parameters()).device
    precision=runtime['precision']
    total_pixels=sum(float(b['valid'].sum()) for b in batches)
    total_images=sum(len(b['image']) for b in batches)
    # Float16 overflow retries replay the same data, dropout and mutable buffers.
    rng=random_state() if scaler.is_enabled() else None
    buffers={k:v.detach().clone() for k,v in model.named_buffers()} if scaler.is_enabled() else {}
    for attempt in range(runtime.get('overflow_retries',5)+1):
        optimizer.zero_grad(set_to_none=True)
        losses=[]
        for batch in batches:
            x,y,valid=[batch[k].to(device,non_blocking=True) for k in ('image','mask','valid')]
            with autocast(device,precision):
                output=forward_model(model,x,valid)
                loss=segmentation_loss(output,y,valid,training,total_pixels,total_images)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss; precision/configuration requires investigation')
            scaler.scale(loss).backward()
            losses.append(loss.detach())
            del output
        scaler.unscale_(optimizer)
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),float(training.get('clip_grad_norm',100.)))
        if torch.isfinite(grad):
            scaler.step(optimizer);scaler.update()
            result={'loss':float(torch.stack(losses).sum()),'grad_norm':float(grad),
                    'overflow_retries':attempt,'effective_batch':total_images,'amp_scale':scaler.get_scale()}
            return result
        if not scaler.is_enabled():raise FloatingPointError('Non-finite gradient')
        scaler.update(new_scale=scaler.get_scale()/2)  # Never call optimizer.step on a failed attempt.
        with torch.no_grad():
            for name,value in buffers.items():model.get_buffer(name).copy_(value)
        restore_random_state(rng)
    raise FloatingPointError('FP16 overflow retries exhausted; no successful optimizer update')


def make_scheduler(optimizer, training, updates):
    warmup=training.get('warmup_steps',0)
    if warmup>=updates:raise ValueError('warmup_steps must be smaller than total updates')
    if training['scheduler']=='none' and not warmup:return None
    functions=[]
    for group in optimizer.param_groups:
        floor=training.get('min_lr',0.)/group['lr']
        if not 0<=floor<=1:raise ValueError('Minimum learning rate exceeds a parameter-group learning rate')
        def factor(step,floor=floor):
            if step<warmup:return (step+1)/max(1,warmup)
            if training['scheduler']=='none':return 1.
            progress=min(1.,max(0.,(step-warmup)/max(1,updates-warmup)))
            return floor+(1-floor)*.5*(1+math.cos(math.pi*progress))
        functions.append(factor)
    return torch.optim.lr_scheduler.LambdaLR(optimizer,functions)
