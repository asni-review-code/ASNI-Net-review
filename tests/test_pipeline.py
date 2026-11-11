"""Portable CPU checks. Synthetic fixtures are not scientific experiments."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
import torch
import yaml
from asni_net.data import pixel_sha,read_rows,verify_data
from asni_net.io import sha256,write_csv,load_weights
from asni_net.model import ASNINet,fractional_sample
from asni_net.training import train,early_stop
from asni_net.transforms import letterbox,restore_logits
from asni_net.metrics import binary_metrics,grouped_values
from asni_net.evaluation import evaluate_model
from asni_net.analysis import summarize_seeds

PROJECT=Path(__file__).resolve().parents[1]

def fixture(root):
    rows=[]
    for i in range(6):
        h,w=25+i,37+i
        yy,xx=np.mgrid[:h,:w];mask=((xx-w/2)**2+(yy-h/2)**2<36).astype(np.uint8)
        image=np.stack([mask*180+20]*3,-1).astype(np.uint8)
        ip=root/f'image{i}.png';mp=root/f'mask{i}.png'
        Image.fromarray(image).save(ip);Image.fromarray(mask).save(mp)
        rows.append({'dataset':'bus_bra','sample_id':str(i),'group_id':str(i),'fold':1,
                     'role':'train' if i<4 else 'val','image_path':ip.name,'mask_path':mp.name,
                     'image_sha256':sha256(ip),'mask_sha256':sha256(mp),'mask_pixel_sha256':pixel_sha(mask),
                     'width':w,'height':h,'foreground_fraction':float(mask.mean())})
    split=root/'split.csv';write_csv(split,rows)
    cfg=yaml.safe_load((PROJECT/'configs/bus_bra_20260928.yaml').read_text('utf-8'))
    cfg.update(input_size=32,cpu_threads=2,split_file=str(split))
    cfg['runtime'].update(precision='fp32',workers=0,pin_memory=False)
    cfg['training'].update(epochs=3,batch_size=2,accumulation_steps=1,warmup_steps=1,validate_every=2,checkpoint_every=1)
    return cfg,rows

class PipelineTests(unittest.TestCase):
    def test_sampling_zero_extension(self):
        x=torch.tensor([[[[2.,4.,8.]]]])
        offsets=torch.tensor([[[[-.5,0.,.5]]]],requires_grad=True)
        y=fractional_sample(x,offsets,3)
        torch.testing.assert_close(y,torch.tensor([[[[1.,4.,4.]]]]))
        y.sum().backward();self.assertTrue(torch.isfinite(offsets.grad).all())

    def test_metrics_and_grouping(self):
        p=np.zeros((5,5),bool);g=p.copy();g[2,2]=True
        m=binary_metrics(p,g);self.assertEqual(m['dice'],0.)
        self.assertEqual(m['hd95_penalized_px'],np.hypot(5,5))
        self.assertIsNone(m['hd95_px'])
        self.assertEqual(binary_metrics(g,g)['asd_px'],0.)
        self.assertEqual(binary_metrics(g,p)['normal_image_fp'],1)
        self.assertIsNone(binary_metrics(g,p)['dice'])
        rows=[{'group_id':'a','dice':0.},{'group_id':'a','dice':1.},{'group_id':'b','dice':1.}]
        self.assertEqual(np.mean(list(grouped_values(rows,'dice').values())),.75)

    def test_geometry(self):
        im=Image.fromarray(np.zeros((25,51,3),np.uint8));mask=np.zeros((25,51),np.uint8)
        x,y,valid,geo=letterbox(im,mask,64)
        self.assertEqual(tuple(x.shape),(3,64,64))
        self.assertEqual(tuple(restore_logits(torch.ones(1,1,64,64),geo).shape),(1,1,25,51))
        self.assertEqual(int(valid.sum()),geo['resized_height']*geo['resized_width'])

    def test_plateau_is_separate_from_best(self):
        history=[{'step':5,'val_dice':.8},{'step':60,'val_dice':.8005}]
        self.assertTrue(early_stop(history,1,{'min_epochs':60,'patience_epochs':20,'min_delta':.001}))

    def test_frozen_members(self):
        counts={'bus_bra':(1204,295,376),'tn3k':(2591,288,614),'mmotu':(703,82,382)}
        for dataset,expected in counts.items():
            rows=read_rows(dataset)
            self.assertEqual(tuple(sum(r['role']==role for r in rows) for role in ('train','val','test')),expected)
        self.assertEqual(len(read_rows('bus_uclm')),665)

    def test_resume_and_evaluation(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cfg,rows=fixture(root)
            verify_data(rows,root)
            uninterrupted=root/'full';resumed=root/'resumed'
            train(cfg,root,uninterrupted,'cpu',smoke=True)
            train(cfg,root,resumed,'cpu',stop_after=3,smoke=True)
            train(cfg,root,resumed,'cpu',resume=resumed/'checkpoints/last.pt',smoke=True)
            a=torch.load(uninterrupted/'checkpoints/last.pt',weights_only=False,map_location='cpu')
            b=torch.load(resumed/'checkpoints/last.pt',weights_only=False,map_location='cpu')
            for k,v in a['model'].items():self.assertTrue(torch.equal(v,b['model'][k]),k)
            self.assertEqual(a['history'],b['history'])
            self.assertEqual(a['scheduler'],b['scheduler'])
            model=load_weights(resumed/'checkpoints/best.pt',ASNINet())
            self.assertEqual(sum(p.numel() for p in model.parameters()),1448945)
            records,summary=evaluate_model(model,rows,'val',cfg,root,.1,root/'evaluation')
            self.assertEqual(len(records),2);self.assertTrue(np.isfinite(summary['positive']['dice']['mean']))
            # Pixel checks must reject an altered mask.
            Image.fromarray(np.zeros((rows[0]['height'],rows[0]['width']),np.uint8)).save(root/rows[0]['mask_path'])
            with self.assertRaises(ValueError):verify_data(rows,root)

    def test_reference_statistics(self):
        files=sorted((PROJECT/'reference_results').glob('*_test.csv'))
        summary=summarize_seeds(files)
        expected={'bus_bra':.9041551655748044,'tn3k':.8384450437455432,'mmotu':.8510044743072526}
        for dataset,value in expected.items():
            self.assertEqual(summary[dataset]['n_seeds'],3)
            self.assertAlmostEqual(summary[dataset]['metrics']['dice']['mean'],value,places=12)

if __name__=='__main__':
    unittest.main()
