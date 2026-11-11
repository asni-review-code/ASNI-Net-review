"""Portable files and complete random states for deterministic resumption."""
import csv
import hashlib
import json
import platform
import random
from pathlib import Path
import numpy as np
import torch

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n','utf-8')
    tmp.replace(path)

def write_csv(path,rows):
    if not rows:
        raise ValueError('Cannot write empty table')
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def random_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}

def restore_random_state(state):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])

def environment():
    import PIL,scipy,yaml
    return {'python':platform.python_version(),'platform':platform.platform(),
            'torch':str(torch.__version__),'numpy':np.__version__,'scipy':scipy.__version__,
            'pillow':PIL.__version__,'pyyaml':yaml.__version__,'cuda_runtime':torch.version.cuda,
            'gpu':torch.cuda.get_device_name() if torch.cuda.is_initialized() else None}

def source_hashes():
    return {p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}

def load_weights(path,model):
    """Public weight files contain tensors only; reject missing/unexpected keys."""
    state=torch.load(path,map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    return model
