# Reproduction protocol

## Formal model and initialization

The released network is the frozen ASNI-Net model (`asni_local_b` in historical
experiment records). It has three ASNI blocks, K=16 nodes, C/4 node channels,
and no local DSC branch in those blocks. NI uses the same scores S with softmax
over N for aggregation and over K for redistribution. Floating-point softmax
and node contractions use FP32 under BF16 autocast. Axial sampling is H then W,
with width offsets predicted from the height stage's updated features. The
fusion is X + 0.5 F1 + 0.5 F2. Height projection uses GELU; width projection does
not add another GELU. The bottleneck uses dropout 0.1, other stages 0.

Version 2 uses descriptive parameter keys; the nine checkpoints were converted
without changing tensor values. Exact scratch initialization RNG consumption and
parameter iteration order are preserved. The separate initialization helper
temporarily creates/discards historical tensor shapes but contains no alternate
model forward implementation. See [ARCHITECTURE.md](ARCHITECTURE.md) for the
module layout, unused compatibility parameters and weight-version boundary,
and [the provenance notice](../LICENSE) for adapted portions.

## Training

- RGB conversion, [0,1] scaling; grayscale is repeated into three channels.
  Existing RGB images retain their color values. No ImageNet normalization.
- Isotropic resize to fit 512×512; centered zero padding; bilinear images and
  nearest-neighbor labels. Loss excludes the padded region.
- Horizontal flip probability 0.5, brightness and contrast ranges ±0.1;
  contrast about 0.5 followed by additive brightness, clamp, then reapply valid mask.
- AdamW: lr 3e-4, weight decay 1e-4, betas (0.9,0.999), eps 1e-8,
  gradient norm clipping at 100; all trainable parameters are in one group.
- Physical batch 4, accumulation 2, BF16 training. Incomplete physical batches
  are dropped. The last update in an epoch may contain one microbatch.
- BCE (sum over valid pixels divided by effective-batch valid pixels) + Dice
  (per-image loss averaged across the effective batch), weights both 1,
  Dice smoothing 1e-6. Validation is FP32, batch 1.
- Max 100 epochs. Five-epoch linear warmup followed by cosine decay to 1e-6.
  Validation every five epochs, raw maximum validation Dice selects best weights.
- Early stop: minimum 60 epochs; 20 epochs without improvement strictly larger
  than 0.001 relative to the stopping anchor. This anchor is separate from best
  checkpoint selection. Seeds: 20260928/29/30.
- Deterministic algorithms enabled; cuDNN benchmarking and TF32 disabled.

The original asynchronous controller sometimes stopped a few optimizer updates
after the eligible validation event. This standalone trainer stops at that event.
The original **actual end epochs** and selected best epochs are retained in
`weights/index.json`; they have not been rounded into fictional historical runs.
This control-layer change does not redefine the selected archived weights.

## Evaluation

Resize logits to the network input size, crop letterbox padding, then bilinearly
restore logits to the original resolution (`align_corners=False`), apply sigmoid
and threshold at 0.5. No TTA, ensembling or connected-component postprocessing.

Dice, IoU, precision and recall are calculated per image. Surfaces are
4-connected boundaries; HD95 and ASD use concatenated bidirectional Euclidean
surface distances in pixels. An empty prediction on a positive target scores
zero Dice/IoU, and the penalized boundary metrics use the image diagonal; the
unpenalized boundary values remain missing. Normal images are evaluated by
nonempty prediction and foreground fraction, not assigned artificial lesion Dice.

The small-lesion cutoff is the lower quartile of positive foreground fractions
in the **source training partition**. External evaluation reuses that cutoff.
BUS-BRA uses equal-weight patient means; TN3K/MMOTU use image means. BUS-UCLM
positive results use patient-group means; normal-image rates are image-weighted.

`summarize` reports per-seed means, mean across seeds, sample SD (ddof=1), and
range in metric units (multiply Dice/IoU by 100 for percentage points).
Per-checkpoint evaluation CIs describe test-sampling uncertainty conditional on
that trained model; they are not training-seed uncertainty or Holm significance.
The descriptive bootstrap in the evaluator is not the manuscript's multi-model,
seed-averaged paired inferential analysis. Comparators were deliberately excluded
from this repository, so it does not reproduce cross-model p-values/rankings.

## What has and has not been rerun for this release

Validated: exact model initialization/RNG equivalence, CPU forward/backward
equivalence, nine original checkpoint loads and real validation image forwards,
data conversion/checksum verification, CPU synthetic training/resume/evaluation,
and numerical reconstruction of the ASNI seed summaries from stored results.
Detailed receipts are in `validation/`.

Not rerun: nine full 100-epoch training budgets, full-dataset GPU evaluation,
GPU/low-power hardware timing, or all paper analyses. The shipped weights and
test tables are original archived results; release smoke tests are not new
scientific performance results. Reproducing exact results across devices,
driver versions and PyTorch builds is not guaranteed.
