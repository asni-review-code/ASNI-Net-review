# Reproducing ASNI-Net

Run every command below from the repository root.

Start from the `main` branch linked on the repository homepage. For an existing
Git checkout, `git status` shows local changes and `git rev-parse HEAD` identifies
the code being used. Keep that identifier with your run records; each saved
release remains a separate snapshot. Archive downloads include
[`release_manifest.json`](../release_manifest.json) with file checksums.

## Installation

Use Python 3.12. The original experiments used Python 3.12.3 and PyTorch 2.8.0
with CUDA 12.8 on an RTX 5090. From this repository root:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
```

For CPU use, replace `cu128` with `cpu` in the PyTorch installation command.
Exact original scientific-library versions are listed in `requirements-paper.txt`;
install them with `pip install -r requirements-paper.txt --extra-index-url https://download.pytorch.org/whl/cu128`
when reconstructing the paper environment. CPU smoke tests exercise the code in
FP32; the formal training configuration uses BF16 on a supported CUDA device.
There is no silent precision fallback. No timm, torchvision or custom CUDA
extension is required by this standalone implementation.

## Data

Download the four public datasets from their original providers, following
[DATASETS.md](DATASETS.md). Extract each release into a source directory.
Full datasets are not redistributed with this repository; the credited BUS-BRA
illustrations in `assets/` are provided only to explain the model and predictions.

```bash
python -m asni_net prepare --dataset bus_bra --source-root /downloads/BUSBRA --data-root /datasets/asni
python -m asni_net prepare --dataset tn3k --source-root /downloads/TN3K --data-root /datasets/asni
python -m asni_net prepare --dataset mmotu --source-root /downloads/OTU_2d --data-root /datasets/asni
python -m asni_net prepare --dataset bus_uclm --source-root /downloads/BUS-UCLM --data-root /datasets/asni
python -m asni_net verify-data --dataset bus_bra --data-root /datasets/asni
```

Preparation locates files by SHA-256, converts the original labels, and checks
the resulting mask pixels against the frozen manifest. It reads source images
without changing them. Paths may be Windows paths; quote paths containing spaces.
Outputs must be outside the source directory. See [the split manifest](../splits/manifest.json).

## Train, resume and evaluate

The nine complete YAML files in `configs/` correspond to the three seeds
20260928, 20260929 and 20260930 on BUS-BRA, TN3K and MMOTU. Example:

```bash
python -m asni_net train --config configs/bus_bra_20260928.yaml --data-root /datasets/asni --output runs/bus_bra_20260928 --device cuda
```

Training saves `checkpoints/last.pt` (full state for trusted local resume),
`checkpoints/best.pt` (tensor-only model weights), the complete configuration,
source hashes, environment, history and actual/best epochs. Selection uses the
strict maximum of validation Dice, never test results. To pause reproducibly,
use `--stop-after N` successful optimizer updates. Resume a paused run with:

```bash
python -m asni_net train --config configs/bus_bra_20260928.yaml --data-root /datasets/asni --output runs/bus_bra_20260928 --device cuda --resume runs/bus_bra_20260928/checkpoints/last.pt
python -m asni_net evaluate --config configs/bus_bra_20260928.yaml --weights weights/bus_bra_20260928.pt --data-root /datasets/asni --output runs/test_bus_bra_20260928 --device cuda
```

Use the provided `weights/` to evaluate the published checkpoints directly, or
substitute your run's `checkpoints/best.pt`. Output directories must be new to
protect previous results. Public weights load with `weights_only=True`; full
training resume files use Python serialization and must be your trusted files.

The external evaluation uses the fixed **BUS-BRA / 20260928** checkpoint:

```bash
python -m asni_net evaluate --config configs/bus_bra_20260928.yaml --weights weights/bus_bra_20260928.pt --dataset bus_uclm --role external_test --data-root /datasets/asni --output runs/external_bus_uclm --device cuda
```

Evaluation produces original-resolution masks, probabilities, per-image/per-group
metrics, bootstrap summaries, threshold and checkpoint provenance. Normal images
use nonempty-prediction rate and predicted area fraction, with no invented lesion
Dice. BUS-BRA and positive BUS-UCLM results use equal-weight patient/group means;
TN3K/MMOTU use image means. See [PROTOCOL.md](PROTOCOL.md).

## Single-image use and analysis

```bash
python -m asni_net predict --weights weights/bus_bra_20260928.pt --image image.png --output runs/prediction --device cpu
python -m asni_net inspect --weights weights/bus_bra_20260928.pt --image image.png --output runs/inspection --device cpu
python -m asni_net benchmark --weights weights/bus_bra_20260928.pt --output runs/timing --device cuda
python -m asni_net summarize reference_results/bus_bra_20260928_test.csv reference_results/bus_bra_20260929_test.csv reference_results/bus_bra_20260930_test.csv --output runs/bus_bra_summary.json
```

Inspection exports offset maps, node aggregation/redistribution matrices and
intermediate feature summaries as NPZ plus documented statistics. Timing saves
every raw sample; the default is 50 warmups followed by 5×200 synchronized forward
measurements. It is a standalone forward benchmark, not a recreation of the
historical multi-model profiler's memory lifecycle.

## Validation and reproducibility

Version 2 uses explicit SIP, NI and channel mapping components and descriptive
checkpoint keys. See [ARCHITECTURE.md](ARCHITECTURE.md) for the source layout
and checkpoint compatibility. All nine packaged weight files use the v2 schema;
their tensor values and the paper model's computation are unchanged.

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic images and CPU only. They cover sampling borders, geometry,
empty-target/boundary metrics, fixed membership, exact interrupted/resumed
training and reference seed summaries. The exported model was additionally
compared with the frozen research implementation: all three initializations,
random-state consumption, train/eval outputs, gradients and optimizer updates matched, and all nine
checkpoints matched on real 512×512 validation inputs. See `validation/` for the
release validation record and its limits.

This is an extraction of the actual trained architecture, not a new architecture
or a rerun of the paper's full training campaign. Hardware/software changes can
change floating-point results and timing. [PROTOCOL.md](PROTOCOL.md) separates
model/checkpoint equivalence from full training reproducibility.

The [GitHub CPU workflow](https://github.com/asni-review-code/ASNI-Net-review/actions/workflows/tests.yml)
runs the same test suite on a fresh Linux runner with Python 3.12. A passing
workflow checks the software paths above; it is not a repeat of the paper's
full training or GPU evaluation.

