# Reproducibility checklist

| Material | Delivered files / commands |
|---|---|
| Source code | `asni_net/`, MIT license and third-party notices |
| Fixed splits | `splits/{bus_bra,tn3k,mmotu,bus_uclm}.csv`, fixed membership and checksums |
| Training configurations | all nine `configs/*.yaml`, exact scratch model initialization, `train`, full `--resume` |
| Evaluation scripts | `evaluate`, `predict`, original-size geometry, empty-mask rules and grouping |
| Split reconstruction | full member lists, MMOTU curation, provider/version records and `prepare` / `verify-data` |
| Training variation | all nine tensor-only checkpoints, per-seed reference tables, `summarize` |
| Coordinate / node / offset explanations | documented model code, `inspect` with offsets and A/B matrices |
| Runtime evidence | `benchmark` with raw synchronized samples and explicit standalone measurement scope |
| Practical execution | installation instructions, CPU tests, provenance and validation receipts |

This is an ASNI-only repository by design. It does not include competitor
implementations, their weights, cross-model paired tests, alternate-network
ablations or a claim to reproduce every figure/table of the entire manuscript.
The comparison evidence remains in the manuscript and supplement.

See [the paper-to-code guide](PAPER_TO_CODE.md) for manuscript section and figure correspondence.
