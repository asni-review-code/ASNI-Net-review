"""Evaluation at original image resolution, with the frozen aggregation units."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .data import read_rows,verify_data
from .transforms import LesionDataset,load_pair,restore_logits,small_threshold
from .runtime import loader
from .metrics import binary_metrics,summarize,summarize_light,per_group
from .model import ASNINet
from .io import load_weights,write_csv,write_json,sha256

@torch.inference_mode()
def evaluate_model(model,rows,role,config,data_root,small_cutoff,output=None,lightweight=False,threshold=.5):
    dataset=LesionDataset(rows,role,config['input_size'],data_root)
    device=next(model.parameters()).device
    was_training=model.training;model.eval();records=[]
    out=Path(output) if output else None
    if out:
        out.mkdir(parents=True,exist_ok=False)
        (out/'predictions').mkdir();(out/'probabilities').mkdir()
    try:
        for batch in loader(dataset,config):
            logits=model(batch['image'].to(device))['logits'].float()
            for i,row in enumerate(batch['row']):
                probability=restore_logits(logits[i:i+1],batch['geometry'][i]).sigmoid()[0,0].cpu().numpy()
                _,target=load_pair(row,data_root)
                pred=probability>=threshold
                record={k:row[k] for k in ('dataset','sample_id','group_id','fold','role')}
                record.update(model='ASNI-Net',seed=config['seed'],small_threshold=small_cutoff,
                              small_target=int(0<float(target.mean())<=small_cutoff))
                record.update(binary_metrics(pred,target,boundaries=not lightweight));records.append(record)
                if out:
                    name=row['sample_id'].replace('/','__').replace('\\','__')
                    Image.fromarray(pred.astype(np.uint8)*255).save(out/'predictions'/f'{name}.png')
                    np.savez_compressed(out/'probabilities'/f'{name}.npz',probability=probability)
        summary=(summarize_light if lightweight else summarize)(records,rows[0]['dataset'])
        if out:
            write_csv(out/'per_image.csv',records);write_csv(out/'per_group.csv',per_group(records,rows[0]['dataset']))
            write_json(out/'summary.json',summary)
            write_json(out/'protocol.json',{'input_size':config['input_size'],'threshold':threshold,
                        'tta':False,'ensemble':False,'original_resolution':True,
                        'small_threshold_from_source_train':small_cutoff,
                        'interpolation':'bilinear_logits_align_corners_false'})
        return records,summary
    finally:
        model.train(was_training)

def evaluate(weights,config,data_root,output,device='cpu',dataset=None,role='test'):
    dataset=dataset or config['dataset']
    if dataset!=config['dataset'] and not (dataset=='bus_uclm' and config['dataset']=='bus_bra'):
        raise ValueError('Only BUS-BRA weights can evaluate the external BUS-UCLM dataset')
    if dataset=='bus_uclm' and role!='external_test':raise ValueError('Use external_test role')
    rows=read_rows(dataset);verify_data([r for r in rows if r['role']==role],data_root)
    cutoff=small_threshold(read_rows(config['dataset']))
    torch.set_num_threads(config['cpu_threads'])
    model=load_weights(weights,ASNINet()).to(device)
    records,summary=evaluate_model(model,rows,role,config,data_root,cutoff,output)
    write_json(Path(output)/'weights_reference.json',{'sha256':sha256(weights),'source_dataset':config['dataset'],'seed':config['seed']})
    return summary
