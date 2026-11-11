"""Render the shipped three-seed summaries; optional dependency: matplotlib."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
values = json.loads((ROOT/'reference_results/summary_three_seeds.json').read_text('utf-8'))
datasets = [('bus_bra', 'BUS-BRA'), ('tn3k', 'TN3K'), ('mmotu', 'MMOTU')]
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'svg.fonttype':'none'})
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
colors = ['#246a89', '#bb7941', '#54835c']
for ax, (metric, title, factor, limits) in zip(axes, [
    ('dice', 'Dice (%)', 100, (78, 94)),
    ('iou', 'IoU (%)', 100, (68, 88)),
    ('hd95_penalized_px', 'HD95 (pixels)', 1, (0, 70)),
]):
    for x, ((key, label), color) in enumerate(zip(datasets, colors)):
        data = values[key]['metrics'][metric]
        mean, sd = data['mean']*factor, data['sample_sd']*factor
        seeds = [run[metric]*factor for run in values[key]['per_seed']]
        ax.scatter(x+np.array([-.1,0,.1]), seeds, s=24, facecolors='white', edgecolors=color, zorder=4)
        ax.errorbar(x, mean, yerr=sd, color=color, marker='D', markersize=5, capsize=5, linewidth=1.7, zorder=3)
        ax.text(x, mean+sd+(limits[1]-limits[0])*.045, f'{mean:.2f} ± {sd:.2f}', ha='center', fontsize=9, color='#273746')
    ax.set_title(title, weight='bold', pad=12)
    ax.set_xticks(range(3), [label for _,label in datasets])
    ax.set_ylim(limits); ax.set_xlim(-.5, 2.5)
    ax.grid(axis='y', color='#e5e9ed', linewidth=.7)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#b3bdc7')
    ax.tick_params(length=3, color='#b3bdc7')
fig.suptitle('ASNI-Net · Internal test results across three training seeds', fontsize=13, weight='bold', y=1.02)
fig.text(.5, -.01, 'Diamonds: seed means   |   Error bars: sample SD   |   Circles: individual seeds', ha='center', fontsize=10, color='#4b5a65')
fig.tight_layout(w_pad=2)
(ROOT/'assets').mkdir(exist_ok=True)
for suffix in ('png','svg'):
    fig.savefig(ROOT/f'assets/results.{suffix}', dpi=180, bbox_inches='tight', facecolor='white')
plt.close(fig)
