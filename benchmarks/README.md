# AutoBench, MIL Benchmark Suite

Benchmark suite for evaluating and improving Multiple Instance Learning (MIL)
models in computational pathology. Demonstrates **autoMIL** across multiple
datasets.

## Datasets

| Dataset | Tasks | Classes | Slides | Config |
|---------|-------|---------|--------|--------|
| **Ovarian** | BRCA, HRD | 2 (binary) | ~400 | `datasets/other/ovarian.yaml` |
| **CLWD** | Subtype (7-class, 6-class) | 6-7 | 408 | `datasets/other/clwd.yaml` |
| **Placeholder** | TBD | TBD | TBD | `datasets/templates/placeholder.yaml` |

## Setup

```bash
# From the repo root
cp benchmarks/.env.example benchmarks/.env
# Edit .env with your paths and tokens

# Install TRIDENT (required for feature extraction)
pip install -e benchmarks/lib/TRIDENT

# Install autobench
uv sync
```

`benchmarks/.env` is local-only and should never be committed. The new
`benchmarks/.gitignore` excludes it along with caches and benchmark outputs.

## Scope

`benchmarks/src` and `benchmarks/scripts` are the first-party benchmark layer.
`benchmarks/lib` vendors upstream research code for reproducibility under their
original licenses; treat it as third-party integration code rather than the
public autoMIL framework API.

## Usage

### Data Preparation

```bash
# Prepare ovarian dataset (task CSVs, splits, H5->PT conversion)
python benchmarks/scripts/run_benchmark.py --dataset ovarian --prep_only

# Prepare CLWD dataset
python benchmarks/scripts/run_benchmark.py --dataset clwd --prep_only
```

### Feature Extraction

```bash
# Extract features using all 7 foundation models
python benchmarks/scripts/run_feature_extraction.py --dataset ovarian --all_gpus

# Specific models only
python benchmarks/scripts/run_feature_extraction.py --dataset clwd --models conch_v15 hoptimus1
```

### Running Benchmarks

```bash
# Full benchmark on a single GPU
python benchmarks/scripts/run_benchmark.py --dataset ovarian --gpu 0

# Multi-GPU with specific frameworks and strategies
python benchmarks/scripts/run_benchmark.py --dataset ovarian \
    --frameworks clam nnmil --strategies a b c --all_gpus

# CLWD benchmark
python benchmarks/scripts/run_benchmark.py --dataset clwd --gpu 0
```

### Using with autoMIL

Each dataset has a pre-configured autoMIL overlay in `experiments/`:

```bash
cd benchmarks/experiments/ovarian_hrd
automil init   # if not already initialized
automil orchestrator start
```

## Adding a New Dataset

1. Copy `datasets/templates/placeholder.yaml` and fill in your dataset's paths, tasks, cohorts, and encoders.
2. Create a new experiment directory: `experiments/your_dataset/automil/config.yaml`.
3. Run preparation: `python benchmarks/scripts/run_benchmark.py --dataset your_dataset --prep_only`.

## Architecture

```
benchmarks/
├── datasets/         # Per-dataset YAML configs (ovarian, clwd, placeholder)
├── src/autobench/    # Reusable benchmark code
│   ├── config.py     # DatasetConfig loader (YAML → dataclass)
│   ├── data.py       # Generic data loading and filtering
│   ├── encoders/     # Custom encoder wrappers
│   └── pipeline/     # Experiment execution engine
│       ├── config.py       # ExperimentConfig, registries, grid generation
│       ├── prepare.py      # H5→PT conversion, task CSVs, splits
│       ├── train.py        # CLAM training loop
│       ├── runner.py       # Single-experiment runner (all folds)
│       ├── evaluate.py     # Metrics and confidence intervals
│       ├── orchestrator.py # Multi-GPU scheduling
│       ├── nnmil/          # nnMIL framework adapter
│       └── smmile/         # SMMILe framework adapter
├── scripts/          # CLI entry points
├── experiments/      # autoMIL overlays per dataset
├── lib/              # External dependencies (CLAM, nnMIL, SMMILe, TRIDENT)
└── tests/            # Test suite
```
