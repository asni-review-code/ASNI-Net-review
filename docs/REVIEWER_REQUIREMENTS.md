# Reviewer request coverage

| Request | Delivered files / commands |
|---|---|
| R1 minor 6: source code | `asni_net/`, MIT license and third-party notices |
| R1 minor 6: fixed splits | `splits/{bus_bra,tn3k,mmotu,bus_uclm}.csv`, immutable membership and checksums |
| R1 minor 6: training configurations | all nine `configs/*.yaml`, exact scratch model initialization, `train`, full `--resume` |
| R1 minor 6: evaluation scripts | `evaluate`, `predict`, original-size geometry, empty-mask rules and grouping |
| R1 major 6: reproducible split delivery | full member lists, MMOTU curation, provider/version records and `prepare` / `verify-data` |
| Training variation | all nine tensor-only checkpoints, per-seed reference tables, `summarize` |
| Coordinate / node / offset explanations | documented model code, `inspect` with offsets and A/B matrices |
| Runtime evidence | `benchmark` with raw synchronized samples and explicit standalone measurement scope |
| Practical execution | installation instructions, CPU tests, provenance and validation receipts |

This is an ASNI-only repository by design. It does not include competitor
implementations, their weights, cross-model paired tests, alternate-network
ablations or a claim to reproduce every figure/table of the entire manuscript.
The comparison evidence remains in the manuscript and supplement.

See [the paper-to-code guide](PAPER_TO_CODE.md) for manuscript section and figure correspondence.
