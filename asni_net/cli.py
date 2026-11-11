"""Command line entry point; every command works from a standalone checkout."""
import argparse
import json
from pathlib import Path
import yaml

def main():
    parser=argparse.ArgumentParser(prog='asni',description='ASNI-Net reproducibility tools')
    commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare');p.add_argument('--dataset',required=True)
    p.add_argument('--source-root',required=True);p.add_argument('--data-root',required=True)
    p=commands.add_parser('verify-data');p.add_argument('--dataset',required=True);p.add_argument('--data-root',required=True)
    p=commands.add_parser('train');p.add_argument('--config',required=True);p.add_argument('--data-root',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');p.add_argument('--resume')
    p.add_argument('--stop-after',type=int);p.add_argument('--smoke',action='store_true')
    p=commands.add_parser('evaluate');p.add_argument('--config',required=True);p.add_argument('--weights',required=True)
    p.add_argument('--data-root',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--dataset');p.add_argument('--role',choices=['val','test','external_test'],default='test')
    for command in ('predict','inspect'):
        p=commands.add_parser(command);p.add_argument('--weights',required=True);p.add_argument('--image',required=True)
        p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    p=commands.add_parser('summarize');p.add_argument('files',nargs='+');p.add_argument('--output')
    p=commands.add_parser('benchmark');p.add_argument('--weights',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--warmup',type=int,default=50)
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--iterations',type=int,default=200)
    args=parser.parse_args();kw=vars(args).copy();cmd=kw.pop('command');result=None
    if 'config' in kw:kw['config']=yaml.safe_load(Path(kw['config']).read_text('utf-8-sig'))
    if cmd=='prepare':
        from .data import prepare
        result=prepare(**kw)
    elif cmd=='verify-data':
        from .data import read_rows,verify_data
        result=verify_data(read_rows(args.dataset),args.data_root)
    elif cmd=='train':
        from .training import train
        result=train(**kw)
    elif cmd=='evaluate':
        from .evaluation import evaluate
        result=evaluate(**kw)
    elif cmd in ('predict','inspect'):
        from .analysis import predict,inspect
        kw['image_path']=kw.pop('image');(predict if cmd=='predict' else inspect)(**kw)
    elif cmd=='summarize':
        from .analysis import summarize_seeds
        from .io import write_json
        result=summarize_seeds(args.files)
        if args.output:write_json(args.output,result)
    elif cmd=='benchmark':
        from .analysis import benchmark
        result=benchmark(**kw)
    if result is not None:print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))

if __name__=='__main__':
    main()
