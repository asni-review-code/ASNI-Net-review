"""Fixed membership, hash-checked ingestion, no split regeneration."""
import csv
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from .io import sha256,write_json

PROJECT=Path(__file__).resolve().parents[1]
DATASETS=('bus_bra','tn3k','mmotu','bus_uclm')

def contained(root,relative):
    root=Path(root).resolve();path=(root/relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Manifest path escapes data root')
    return path

def pixel_sha(mask):
    return hashlib.sha256(np.asarray(mask,dtype=np.uint8).tobytes()).hexdigest()

def read_rows(dataset,split_file=None,verify=True):
    if dataset not in DATASETS:
        raise ValueError('Unknown dataset')
    path=Path(split_file) if split_file else PROJECT/'splits'/f'{dataset}.csv'
    if verify:
        index=json.loads((PROJECT/'splits/manifest.json').read_text('utf-8'))
        if sha256(path)!=index[dataset]['sha256']:
            raise ValueError('Fixed split checksum mismatch')
    with path.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    ids=set();groups={}
    for r in rows:
        if r['dataset']!=dataset or r['sample_id'] in ids:
            raise ValueError('Incorrect dataset or duplicate sample')
        ids.add(r['sample_id'])
        r.update(width=int(r['width']),height=int(r['height']),fold=1,
                 foreground_fraction=float(r['foreground_fraction']))
        for key in ('image_path','mask_path'):
            contained('.',r[key])
        groups.setdefault(r['group_id'],set()).add(r['role'])
        if r['role'] not in ('train','val','test','external_test'):
            raise ValueError('Unknown partition')
    if dataset=='bus_bra' and any(len(roles)>1 for roles in groups.values()):
        raise ValueError('Patient groups cross partitions')
    if dataset=='bus_uclm' and any(r['role']!='external_test' for r in rows):
        raise ValueError('BUS-UCLM is external-test only')
    return rows

def verify_data(rows,root):
    for r in rows:
        image=contained(root,r['image_path']);mask=contained(root,r['mask_path'])
        if sha256(image)!=r['image_sha256']:
            raise ValueError(f'Image checksum mismatch: {r["sample_id"]}')
        with Image.open(image) as im:
            if im.size!=(r['width'],r['height']):raise ValueError('Image dimensions differ')
        with Image.open(mask) as im:
            a=np.asarray(im.convert('L'))
        if a.shape!=(r['height'],r['width']) or not set(np.unique(a)).issubset({0,1,255}):
            raise ValueError('Mask dimensions/labels differ')
        if pixel_sha(a>0)!=r['mask_pixel_sha256']:
            raise ValueError(f'Mask pixel checksum mismatch: {r["sample_id"]}')
    return {'images_verified':len(rows),'masks_verified':len(rows)}

def prepare(dataset,source_root,data_root):
    """Locate official downloaded files by content hash and prepare fixed members.

    Input files are read-only. No downloads or patient files are redistributed.
    """
    rows=read_rows(dataset)
    source_root=Path(source_root).resolve();data_root=Path(data_root).resolve()
    if data_root==source_root or data_root.is_relative_to(source_root):
        raise ValueError('Output must be outside the read-only source directory')
    wanted={r['image_sha256'] for r in rows}|{r['source_mask_sha256'] for r in rows}
    by_hash={}
    for p in source_root.rglob('*'):
        if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg','.bmp','.tif','.tiff'):
            h=sha256(p)
            if h in wanted:by_hash.setdefault(h,p)
    missing=wanted-set(by_hash)
    if missing:
        raise FileNotFoundError(f'{len(missing)} required source hashes not found; use the documented dataset version')
    for r in rows:
        image_out=contained(data_root,r['image_path']);mask_out=contained(data_root,r['mask_path'])
        if image_out.exists() and sha256(image_out)!=r['image_sha256']:
            raise FileExistsError(f'Refusing to overwrite different image: {image_out}')
        with Image.open(by_hash[r['source_mask_sha256']]) as im:
            if dataset=='bus_uclm':
                a=np.asarray(im.convert('RGB'))
                green=np.all(a==(0,255,0),axis=-1);red=np.all(a==(255,0,0),axis=-1)
                if not np.all(np.all(a==0,axis=-1)|green|red):raise ValueError('Unknown UCLM label color')
                binary=green|red
            else:
                a=np.asarray(im.convert('L'))
                binary=a>=128 if dataset=='tn3k' else a>0
        if pixel_sha(binary)!=r['mask_pixel_sha256']:
            raise ValueError(f'Processed mask mismatch: {r["sample_id"]}')
        if mask_out.exists():
            with Image.open(mask_out) as im:
                if pixel_sha(np.asarray(im.convert('L'))>0)!=r['mask_pixel_sha256']:
                    raise FileExistsError('Refusing to overwrite different mask')
        image_out.parent.mkdir(parents=True,exist_ok=True);mask_out.parent.mkdir(parents=True,exist_ok=True)
        if not image_out.exists():shutil.copyfile(by_hash[r['image_sha256']],image_out)
        if not mask_out.exists():Image.fromarray(binary.astype(np.uint8)).save(mask_out)
    receipt=verify_data(rows,data_root)
    write_json(data_root/f'{dataset}_verified.json',receipt)
    return receipt
