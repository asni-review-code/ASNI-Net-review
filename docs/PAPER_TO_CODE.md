# Paper-to-code correspondence

Reference: the English revision manuscript checked on 2026-11-11, titled
**ASNI-Net: An Adaptive Sampling and Node Interaction Network for 2D Ultrasound
Lesion Segmentation**. Section/table numbers below follow that revision.

| Manuscript location | Released implementation or evidence | What to check |
|---|---|---|
| §3.1, Figure 1: architecture | [`ASNINet`, `ASNIBlock`](../asni_net/model.py), [Figure 1](../assets/architecture.pdf) | Encoder/decoder, additive skips, SIP → NI → CM |
| §3.2, Figure 2: adaptive sampling | [`AxisSampler`, `SamplingInteractionPath`](../asni_net/layers.py), [Figure 2](../assets/adaptive_sampling.pdf) | Per-position/channel offsets, H→W sampling, zero extension, fusion weights |
| §3.3, Figure 3: NI | [`NodeInteraction`](../asni_net/layers.py), [Figure 3](../assets/node_interaction.pdf) | Same S; softmax over N versus K; K=16 and C/4 channels |
| §4.1: datasets/splits/metrics | [Data guide](DATASETS.md), [splits](../splits/), [`transforms.py`](../asni_net/transforms.py), [`metrics.py`](../asni_net/metrics.py) | Membership, patient/group aggregation, original-size logits, empty predictions |
| §4.2, Table 2: implementation | [Nine configurations](../configs/), [`training.py`](../asni_net/training.py), [`initialization.py`](../asni_net/initialization.py), [protocol](PROTOCOL.md) | Seeds, optimizer, BF16, accumulation, early stopping and validation-selected weights |
| §4.3, Table 3: ASNI-Net row | [Per-image tables and summaries](../reference_results/), [nine weights](MODEL_ZOO.md) | Recalculate Dice/IoU/HD95 means and sample SD with `summarize` |
| §4.4.2: sampling and node behavior | `python -m asni_net inspect ...` | Offsets, A/B matrices, single-image feature summaries |
| §4.6: computational efficiency | `python -m asni_net benchmark ...`, [protocol](PROTOCOL.md) | Raw synchronized timings; same input size/precision defaults, but standalone memory lifecycle |
| §4.7: external evaluation | `evaluate --dataset bus_uclm --role external_test`, BUS-BRA/20260928 weights | Frozen checkpoint, threshold 0.5, lesion metrics and normal-image false positives |

## Suggested review route

1. Read the three structure diagrams and inspect the corresponding classes.
2. Check [data membership](../splits/manifest.json) and [training settings](../configs/).
3. Recalculate the shipped three-seed summaries without downloading images:

```bash
python -m asni_net summarize reference_results/bus_bra_20260928_test.csv reference_results/bus_bra_20260929_test.csv reference_results/bus_bra_20260930_test.csv
```

4. Run the CPU tests, then predict an image or evaluate a supplied checkpoint
   after reconstructing the appropriate dataset.
5. Inspect the [verification records](../validation/) and [weight hashes](../weights/index.json).

## Interpretation boundaries

- Table 3 contains 11 models in the manuscript. This repository reproduces its
  ASNI-Net row; comparator implementations, checkpoints and cross-model paired
  inference are outside this package.
- The A0–A5 and replacement/node-count studies remain in the manuscript and
  supplement. Alternative architectures are not supplied as runnable models here.
- `inspect` exports the mechanism quantities used for explanation; it is not a
  claim to recreate the manuscript's entire aggregated mechanism-analysis campaign.
- `benchmark` is a standalone measurement. Its memory lifecycle differs from
  the historical multi-model profiler, and measured speed depends on hardware.
- The external row is a single source-trained checkpoint, not a three-seed
  external study. The README retains its normal-image false-positive limitation.
- The release contains new descriptive state keys with unchanged tensor values.
  Resume existing v1 training states with v1 code; see [compatibility](ARCHITECTURE.md).

[Reproducibility checklist](REVIEWER_REQUIREMENTS.md) · [Complete reproduction guide](REPRODUCING.md)
