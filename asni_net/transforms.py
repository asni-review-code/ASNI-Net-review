import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

def small_threshold(rows):
    values = [r['foreground_fraction'] for r in rows if r['role'] == 'train' and r['foreground_fraction'] > 0]
    if not values:
        raise ValueError('Small-target threshold requires source training data')
    return float(np.quantile(values, .25))


def load_pair(row, root="data"):
    with Image.open(Path(root) / row['image_path']) as im:
        image = im.convert('RGB').copy()
    with Image.open(Path(root) / row['mask_path']) as m:
        mask = np.asarray(m.convert('L'))
    if not set(np.unique(mask)).issubset({0, 1, 255}):
        raise ValueError(f'Non-binary processed mask: {row["sample_id"]}')
    mask = (mask > 0).astype(np.uint8)
    if image.size != (row['width'], row['height']) or mask.shape != (row['height'], row['width']):
        raise ValueError('Image/mask dimensions differ from manifest')
    return image, mask


def prepare_image(image, size):
    w, h = image.size
    scale = min(size / w, size / h)
    nw, nh = min(size, max(1, round(w * scale))), min(size, max(1, round(h * scale)))
    left, top = (size - nw) // 2, (size - nh) // 2
    x = np.zeros((size, size, 3), dtype=np.float32)
    x[top:top+nh, left:left+nw] = np.asarray(image.resize((nw, nh), Image.Resampling.BILINEAR), dtype=np.float32) / 255.
    geo = {'height': h, 'width': w, 'top': top, 'left': left, 'resized_height': nh, 'resized_width': nw, 'size': size}
    return torch.from_numpy(x.transpose(2, 0, 1).copy()), geo


def letterbox(image, mask, size):
    x, geo = prepare_image(image, size)
    top, left, nh, nw = [geo[k] for k in ('top', 'left', 'resized_height', 'resized_width')]
    y = np.zeros((size, size), dtype=np.float32)
    valid = np.zeros_like(y)
    y[top:top+nh, left:left+nw] = np.asarray(Image.fromarray(mask).resize((nw, nh), Image.Resampling.NEAREST))
    valid[top:top+nh, left:left+nw] = 1
    return x, torch.from_numpy(y[None]), torch.from_numpy(valid[None]), geo


def restore_logits(logits, geo):
    z = F.interpolate(logits, (geo['size'], geo['size']), mode='bilinear', align_corners=False)
    z = z[..., geo['top']:geo['top']+geo['resized_height'], geo['left']:geo['left']+geo['resized_width']]
    return F.interpolate(z, (geo['height'], geo['width']), mode='bilinear', align_corners=False)


class LesionDataset(Dataset):
    def __init__(self, rows, role, size=512, root="data", development=False, limit=None, augmentation=None):
        if development and role not in ('train', 'val'):
            raise ValueError('Development runs cannot access test/external_test')
        self.rows = sorted([r for r in rows if r['role'] == role], key=lambda r: r['sample_id'])
        if limit is not None:
            if not development:
                raise ValueError('Subset limits are development-only')
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f'No samples for role {role}')
        self.size, self.root = size, Path(root)
        self.augmentation=augmentation or {}
        if role!='train' and any(self.augmentation.values()):raise ValueError('Augmentation is training-only')
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        draw_seed=None
        if isinstance(index,tuple):index,draw_seed=index
        row = self.rows[index]
        im, mask = load_pair(row, self.root)
        x, y, valid, geo = letterbox(im, mask, self.size)
        if self.augmentation:
            if draw_seed is None:raise ValueError('Training augmentation requires an explicit draw seed')
            rng=random.Random(draw_seed)
            if rng.random()<self.augmentation.get('horizontal_flip',0):
                x,y,valid=[v.flip(-1) for v in (x,y,valid)]
            brightness=self.augmentation.get('brightness',0)
            contrast=self.augmentation.get('contrast',0)
            if brightness or contrast:
                x=((x-.5)*(1+rng.uniform(-contrast,contrast))+.5+rng.uniform(-brightness,brightness)).clamp(0,1)*valid
        return {'image': x, 'mask': y, 'valid': valid, 'geometry': geo, 'row': row}


def collate(samples):
    return {'image': torch.stack([s['image'] for s in samples]), 'mask': torch.stack([s['mask'] for s in samples]),
            'valid': torch.stack([s['valid'] for s in samples]), 'geometry': [s['geometry'] for s in samples],
            'row': [s['row'] for s in samples]}
