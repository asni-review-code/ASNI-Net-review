# Source layout and checkpoint format (v2)

`asni_net/model.py` defines the complete ASNI-Net and its ASNI block. The block's
forward path is explicitly `sampling -> node_interaction -> channel_mapping`,
corresponding to SIP, NI and CM in the paper. `asni_net/layers.py` defines the
individual operations. `asni_net/initialization.py` preserves the frozen
experiment's scratch initialization; no pretrained tensors are used to initialize
a new model.

| Parameter prefix | Role |
| --- | --- |
| `encoder.0/1/2` | Convolution stages with 16/32/64 output channels |
| `token_projections.encoder/bottleneck/decoder` | Spatial projection and token normalization |
| `asni_blocks.encoder/bottleneck/decoder.sampling` | Height then width sampling and fusion |
| `asni_blocks.*.node_interaction` | Shared-score aggregation, mixing and redistribution |
| `asni_blocks.*.channel_mapping` | Block-end CM |
| `output_norms.*` | Normalization after each ASNI block |
| `decoder_projection`, `decoder_normalization` | 128-to-64 spatial projection |
| `decoder.0/1/2`, `segmentation_head` | Additive skip decoding and lesion logits |

The computation uses three ASNI blocks with widths 128/256/128, K=16 nodes and
C/4 node channels. Architecture, layer order, additive skips and operation order
are preserved. A refactor changes implementation organization, not scientific
model identity or the source history described in
[LICENSE](../LICENSE).

## Initialization and reserved tensors

The frozen constructor made draws for temporary tensors before discarding them.
The initialization helper replays those draws so the same seed produces exactly
the same final parameters and leaves the same random state. It has no forward
method and its temporary tensors do not remain in the final network.

The reported 1,448,945 registered parameters include 1,024 normalization
parameters that were already unused in the frozen model. They are now explicitly
named `asni_blocks.*.reserved_normalization`. They have no forward effect and
receive no gradients. They are retained for checkpoint and reported-count
identity; deleting them would change the parameter accounting. The remaining
1,447,921 parameters participate in the forward computation.

## Weights and resumption

The nine files in `weights/` use the v2 parameter schema. Conversion changes only
keys: every tensor, including buffers and reserved parameters, has an exact
one-to-one counterpart in the frozen checkpoint. `weights/index.json` records
both the new file hash and the original checkpoint/v1 weight hashes.

Use v2 code with v2 weights. A v1 `best.pt` does not load directly into v2, and
an in-progress v1 `last.pt` must be resumed with the v1 code. Within v2, full
optimizer, scheduler and random-state resumption remains supported and tested.
The previous release archive is retained separately by the authors.

`validation/model_equivalence.json` reports the comparison to the frozen
implementation: the three paper seeds plus seed 7, parameter order and RNG,
train/eval logits and gradients, an AdamW update, and nine trained checkpoints
on one real 512x512 validation image each. This is numerical software validation,
not a new full training campaign or a guarantee across hardware/software builds.
