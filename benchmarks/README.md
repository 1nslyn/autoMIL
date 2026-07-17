# AutoBench, MIL Benchmark Suite

Benchmark suite for evaluating and improving Multiple Instance Learning (MIL)
models in computational pathology, and the empirical layer that demonstrates
**autoMIL**. autobench varies both benchmark axes — the **encoder** (patch and
slide foundation models) and the **MIL aggregator** — and gives every
(encoder × aggregator) cell an equal-effort, agentic recipe search on a **frozen
data substrate**: splits, folds, and extracted features are held constant, and
the agent may change only architecture and training recipe. That recipe-bias
control, together with a validation-firewall so test never drives selection, is
what lets the encoder-vs-aggregator comparison be read cleanly instead of being
confounded by uneven per-cell tuning effort.

## Datasets

The preprint roster is a 5-cohort TCGA slate, each pinned to one distinct
classification gene plus an overall-survival (OS) task, run across 4 MIL
aggregators (`clam_mb`, `simple_mil`, `ab_mil`, `dtfd_mil`) plus a TITAN
slide-encoder arm:

| Dataset | Classification gene | + Survival | Config |
|---------|--------------------|:----------:|--------|
| **TCGA-LUAD** | KRAS | OS | `datasets/tcga/tcga_luad.yaml` |
| **TCGA-LGG** | IDH1 | OS | `datasets/tcga/tcga_lgg.yaml` |
| **TCGA-SKCM** | NRAS | OS | `datasets/tcga/tcga_skcm.yaml` |
| **TCGA-BLCA** | PIK3CA | OS | `datasets/tcga/tcga_blca.yaml` |
| **TCGA-COAD** | BRAF | OS | `datasets/tcga/tcga_coad.yaml` |

Additional non-slate cohorts also ship configs: **Ovarian** (BRCA/HRD,
`datasets/other/ovarian.yaml`), **CLWD** (lung subtype, `datasets/other/clwd.yaml`),
**HANCOCK** (`datasets/other/hancock.yaml`), and **CPTAC-CCRCC/GBM**
(`datasets/cptac/`). `datasets/templates/` holds the `tcga_`, `cptac_`, and
`placeholder` templates for adding your own. The full Phase-2 (journal) scope is
the 16 TCGA + 10 CPTAC inventory — see [`../paper/`](../paper/).

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
├── datasets/         # Per-dataset YAML configs, grouped by program:
│   ├── tcga/         #   5-member preprint slate (luad/lgg/skcm/blca/coad)
│   ├── cptac/        #   CPTAC cohorts (ccrcc, gbm)
│   ├── other/        #   ovarian, clwd, hancock
│   └── templates/    #   tcga_/cptac_/placeholder templates
├── src/autobench/    # Reusable benchmark code
│   ├── config.py     # DatasetConfig loader (YAML → dataclass)
│   ├── data.py       # Generic data loading and filtering
│   ├── encoders/     # Custom encoder wrappers
│   └── pipeline/     # Experiment execution engine
│       ├── config.py       # ExperimentConfig, registries, grid generation
│       ├── prepare.py      # H5→PT conversion, task CSVs
│       ├── splits.py       # Fold / split generation
│       ├── evaluate.py     # Metrics and confidence intervals
│       ├── orchestrator.py # Multi-GPU scheduling
│       ├── clam/           # CLAM adapter (train loop + runner + survival)
│       ├── nnmil/          # nnMIL framework adapter
│       ├── abmil/          # AB-MIL framework adapter
│       ├── dtfd/           # DTFD-MIL framework adapter
│       ├── titan/          # TITAN slide-encoder arm
│       └── smmile/         # SMMILe framework adapter
├── scripts/          # CLI entry points (run_benchmark.py, run_feature_extraction.py, slurm/)
├── experiments/      # autoMIL overlays per dataset
├── lib/              # External dependencies (CLAM, nnMIL, SMMILe, TRIDENT)
└── tests/            # Test suite
```
