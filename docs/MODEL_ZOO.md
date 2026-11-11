# Checkpoints and configurations

All nine checkpoint files are included in this repository and contain tensors only.
The v2 schema is required. Each best checkpoint was selected using validation Dice.

| Dataset | Seed | Selected epoch | Actual end epoch | Configuration | Checkpoint |
|---|---:|---:|---:|---|---|
| bus_bra | 20260928 | 40 | 60.0331 | [YAML](../configs/bus_bra_20260928.yaml) | [weights](../weights/bus_bra_20260928.pt) |
| bus_bra | 20260929 | 60 | 80.0199 | [YAML](../configs/bus_bra_20260929.yaml) | [weights](../weights/bus_bra_20260929.pt) |
| bus_bra | 20260930 | 60 | 70.0066 | [YAML](../configs/bus_bra_20260930.yaml) | [weights](../weights/bus_bra_20260930.pt) |
| tn3k | 20260928 | 60 | 80.0123 | [YAML](../configs/tn3k_20260928.yaml) | [weights](../weights/tn3k_20260928.pt) |
| tn3k | 20260929 | 40 | 60.0123 | [YAML](../configs/tn3k_20260929.yaml) | [weights](../weights/tn3k_20260929.pt) |
| tn3k | 20260930 | 65 | 70.0062 | [YAML](../configs/tn3k_20260930.yaml) | [weights](../weights/tn3k_20260930.pt) |
| mmotu | 20260928 | 40 | 60.0114 | [YAML](../configs/mmotu_20260928.yaml) | [weights](../weights/mmotu_20260928.pt) |
| mmotu | 20260929 | 60 | 80.0227 | [YAML](../configs/mmotu_20260929.yaml) | [weights](../weights/mmotu_20260929.pt) |
| mmotu | 20260930 | 55 | 75.0227 | [YAML](../configs/mmotu_20260930.yaml) | [weights](../weights/mmotu_20260930.pt) |

The fractional actual end epochs preserve the original controller's stopping
delay; they are not rounded into newly invented training histories. The public
trainer stops at the eligible validation event. Checksums, selected steps and
original checkpoint identities are in [weights/index.json](../weights/index.json).

For BUS-UCLM, use **bus_bra_20260928.pt** without retraining or selecting a new
threshold on the external set. See [the reproduction guide](REPRODUCING.md).
