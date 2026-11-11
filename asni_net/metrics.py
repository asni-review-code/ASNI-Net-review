"""Explicit empty-target rules; Euclidean distances between 4-connected pixel borders."""
import math
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

def binary_metrics(prediction, target, boundaries=True):
    p, g = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if p.shape != g.shape or p.ndim != 2:
        raise ValueError('Metrics require equally sized 2D masks')
    tp, fp, fn, tn = [int(v) for v in (np.sum(p & g), np.sum(p & ~g), np.sum(~p & g), np.sum(~p & ~g))]
    positive = bool(g.any())
    result = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'gt_positive': int(positive),
              'prediction_empty': int(not p.any()), 'empty_prediction_failure': int(positive and not p.any()),
              'dice': 2 * tp / (2 * tp + fp + fn) if positive else None,
              'iou': tp / (tp + fp + fn) if positive else None,
              'precision': tp / (tp + fp) if positive and tp + fp else (0. if positive else None),
              'recall': tp / (tp + fn) if positive else None,
              'normal_image_fp': int(p.any()) if not positive else None,
              'predicted_foreground_fraction': float(p.mean()),
              'gt_foreground_fraction': float(g.mean()), 'hd95_px': None, 'hd95_penalized_px': None,
              'asd_px': None, 'asd_penalized_px': None, 'nhd95': None, 'nhd95_penalized': None}
    if positive and boundaries:
        diagonal = math.hypot(*g.shape)
        if p.any():
            structure = generate_binary_structure(2, 1)
            pb = p ^ binary_erosion(p, structure=structure, border_value=0)
            gb = g ^ binary_erosion(g, structure=structure, border_value=0)
            distances = np.concatenate((distance_transform_edt(~gb)[pb], distance_transform_edt(~pb)[gb]))
            hd = float(np.percentile(distances, 95))
            result['hd95_px'] = hd
            result['asd_px'] = float(distances.mean())
            result['nhd95'] = hd / diagonal
        else:
            hd = diagonal
        result['hd95_penalized_px'] = hd
        result['asd_penalized_px'] = result['asd_px'] if p.any() else diagonal
        result['nhd95_penalized'] = hd / diagonal
    return result

def summarize_light(rows,dataset):
    """Same image/patient Dice and IoU means; no distance transforms or bootstrap."""
    group='group_id' if dataset in ('bus_bra','bus_uclm') else 'sample_id'
    positive=[r for r in rows if int(r['gt_positive'])]
    result={'dataset':dataset,'n_images':len(rows),'n_positive':len(positive),'lightweight':True,'positive':{}}
    for key in ('dice','iou'):
        values=grouped_values(positive,key,group)
        result['positive'][key]={'mean':float(np.mean(list(values.values()))) if values else None}
    return result

def _number(x):
    return x is not None and x != '' and math.isfinite(float(x))

def grouped_values(rows, key, group_key='group_id'):
    grouped = {}
    for r in rows:
        if _number(r.get(key)):
            grouped.setdefault(str(r[group_key]), []).append(float(r[key]))
    return {g: float(np.mean(v)) for g, v in grouped.items()}

def per_group(rows,dataset):
    group_key='group_id' if dataset in ('bus_bra','bus_uclm') else 'sample_id'
    output=[]
    for group in sorted({r[group_key] for r in rows}):
        selected=[r for r in rows if r[group_key]==group]
        positive=[r for r in selected if int(r['gt_positive'])]
        negative=[r for r in selected if not int(r['gt_positive'])]
        record={k:selected[0][k] for k in ('model','dataset','fold','seed') if k in selected[0]}
        record.update({'group_id':group,'n_images':len(selected),'n_positive':len(positive),'n_normal':len(negative)})
        for key in ('dice','iou','precision','recall','asd_px','asd_penalized_px','hd95_px','nhd95','hd95_penalized_px','nhd95_penalized','empty_prediction_failure','normal_image_fp','predicted_foreground_fraction'):
            source=negative if key in ('normal_image_fp','predicted_foreground_fraction') else positive
            values=[float(r[key]) for r in source if _number(r.get(key))]
            name='normal_predicted_area_fraction' if key=='predicted_foreground_fraction' else key
            record[name]=float(np.mean(values)) if values else None
        small=[float(r['dice']) for r in positive if int(r.get('small_target',0)) and _number(r.get('dice'))]
        record['small_target_dice']=float(np.mean(small)) if small else None
        output.append(record)
    return output

def bootstrap_mean(rows, key, group_key='group_id', patient_weighted=True, repeats=2000, seed=20260926):
    groups = {}
    for r in rows:
        if _number(r.get(key)):
            groups.setdefault(str(r[group_key]), []).append(float(r[key]))
    if not groups:
        return {'mean': None, 'ci95': None, 'n_groups': 0, 'n_images': 0}
    arr = list(groups.values())
    sums, counts = np.array([sum(v) for v in arr]), np.array([len(v) for v in arr])
    if patient_weighted:
        sums = sums / counts
        counts = np.ones_like(counts)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(arr), size=(repeats, len(arr)))
    estimates = sums[indices].sum(1) / counts[indices].sum(1)
    return {'mean': float(sums.sum() / counts.sum()), 'ci95': np.quantile(estimates, [.025, .975]).tolist(),
            'n_groups': len(arr), 'n_images': sum(len(v) for v in arr)}

def summarize(rows, dataset, repeats=2000):
    if not rows:
        raise ValueError('Cannot summarize empty evaluation')
    keys = ('dice', 'iou', 'precision', 'recall', 'asd_px', 'asd_penalized_px', 'hd95_px', 'nhd95', 'hd95_penalized_px', 'nhd95_penalized', 'empty_prediction_failure')
    positives = [r for r in rows if int(r['gt_positive'])]
    normals = [r for r in rows if not int(r['gt_positive'])]
    group = 'group_id' if dataset in ('bus_bra', 'bus_uclm') else 'sample_id'
    patient = dataset in ('bus_bra', 'bus_uclm')
    result = {'dataset': dataset, 'n_images': len(rows), 'n_positive': len(positives), 'n_normal': len(normals),
              'statistical_unit': group, 'positive': {k: bootstrap_mean(positives, k, group, patient, repeats) for k in keys}}
    small = [r for r in positives if int(r.get('small_target', 0))]
    result['small_target_dice'] = bootstrap_mean(small, 'dice', group, patient, repeats)
    if dataset == 'bus_uclm':
        result['positive_image_weighted'] = {k: bootstrap_mean(positives, k, group, False, repeats) for k in keys}
        result['normal_image_fpr'] = bootstrap_mean(normals, 'normal_image_fp', group, False, repeats)
        result['normal_predicted_area_fraction'] = bootstrap_mean(normals, 'predicted_foreground_fraction', group, False, repeats)
    return result

def paired_bootstrap(left, right, key='dice', patient_weighted=True, repeats=2000):
    def indexed(rows):
        mapping = {}
        for r in rows:
            identity = (r['dataset'], r['sample_id'])
            if identity in mapping:
                raise ValueError('Repeated sample: aggregate folds/seeds before paired inference')
            mapping[identity] = r
        return mapping
    a, b = indexed(left), indexed(right)
    if set(a) != set(b):
        raise ValueError('Paired comparisons require identical samples')
    differences = []
    for identity in sorted(a):
        x, y = a[identity], b[identity]
        if x['group_id'] != y['group_id']:
            raise ValueError('Paired group mismatch')
        if _number(x.get(key)) and _number(y.get(key)):
            differences.append({'group_id': x['group_id'], 'difference': float(x[key]) - float(y[key])})
    return bootstrap_mean(differences, 'difference', patient_weighted=patient_weighted, repeats=repeats)
