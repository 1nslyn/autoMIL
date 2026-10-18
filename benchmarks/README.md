# AutoBench, MIL Benchmark Suite

Benchmark suite for evaluating Multiple Instance Learning (MIL) models in
computational pathology, and the experiment layer for **autoMIL**.

Each benchmark cell is one dataset task, one tile encoder and one MIL aggregator
(CLAM, nnMIL, ABMIL or DTFD), plus a TITAN slide-encoder arm. A cell is trained
with its native recipe, then an agent searches for a better recipe under an
equal budget (30 launched attempts) on frozen splits, folds and features. The
agent changes only the training recipe. Selection uses validation metrics, and
test results stay sealed until the end of the campaign. The campaign measures
how the ranking of the four tile aggregators changes between the native
baselines and the agent-tuned winners, per dataset, task and encoder (see
[the campaign README](campaigns/preprint_130/README.md) and
`campaigns/preprint_130/analysis_plan.json`).

## Datasets

The active campaign roster has 3 cohorts and 78 cells. Each cohort has one
classification task and an overall-survival (OS) task, run across 4 MIL
aggregators (`clam_mb`, `simple_mil`, `abmil`, `dtfd_mil`) and 3 tile encoders
(`virchow2`, `hoptimus1`, `uni_v2`), plus a TITAN arm: 26 cells per cohort.

| Dataset | Classification task | Task type | + Survival | Config |
|---------|---------------------|-----------|:----------:|--------|
| **TCGA-LUAD** | KRAS (mut/wt) | binary | OS | `datasets/tcga/tcga_luad.yaml` |
| **TCGA-HNSC** | tumor grade | 3-class (G1/G2/G3) | OS | `datasets/tcga/tcga_hnsc.yaml` |
| **CPTAC-PDAC** | immune_class | 3-class (low/med/high) | OS | `datasets/cptac/cptac_pdac.yaml` |

`campaigns/preprint_130/active_roster.json` is the authority for the roster. The
campaign manifest keeps 130 rows because it was frozen with 5 cohorts. TCGA-LGG
and CPTAC-GBM were dropped from the roster on 2026-08-23, but their configs
remain in `datasets/`.

Other configs ship outside the roster: **Ovarian** (BRCA/HRD, `datasets/other/ovarian.yaml`),
**CLWD** (lung subtype, `datasets/other/clwd.yaml`), **HANCOCK**
(`datasets/other/hancock.yaml`) and **CPTAC-CCRCC** (`datasets/cptac/`). Ovarian,
CLWD, HANCOCK and `ccrcc.yaml` configure 7 foundation-model encoders; the roster
datasets and `cptac_ccrcc.yaml` configure 3. `datasets/templates/` holds the `tcga_`, `cptac_` and
`placeholder` templates for adding your own. The dataset loader finds a config by
name, for example `--dataset tcga_luad`, wherever it sits under `datasets/`.

## Setup

```bash
# From the repo root
cp benchmarks/.env.example benchmarks/.env
# Edit .env with your paths and tokens

# Install the full uv workspace, including autobench and its editable TRIDENT dependency
uv sync --all-packages
```

`benchmarks/.env` is local-only and is never committed; `benchmarks/.gitignore`
excludes it along with caches and benchmark outputs. `benchmarks/src` and
`benchmarks/scripts` are the first-party benchmark layer; `benchmarks/lib`
vendors upstream research code (CLAM, nnMIL, TRIDENT) under its original
licenses.

## Usage

Source `benchmarks/.env` first (`set -a && source benchmarks/.env && set +a`).
The step-by-step workflow is in the [tutorials](../docs/tutorials/):
[TCGA](../docs/tutorials/tcga_feature_extraction_tutorial.md) or
[CPTAC](../docs/tutorials/cptac_feature_extraction_tutorial.md) feature
extraction, then the [benchmark experiments tutorial](../docs/tutorials/benchmark_experiments_tutorial.md).
Each script has `--help`.

```bash
# Data preparation only (task CSVs, splits, H5->PT conversion)
uv run --package autobench python benchmarks/scripts/run_benchmark.py --dataset tcga_luad --prep_only

# Feature extraction with every encoder in the dataset config, or a subset
uv run --package autobench python benchmarks/scripts/run_feature_extraction.py --dataset ovarian --all_gpus
uv run --package autobench python benchmarks/scripts/run_feature_extraction.py --dataset clwd --models conch_v15 hoptimus1

# Benchmark on a single GPU
uv run --package autobench python benchmarks/scripts/run_benchmark.py --dataset tcga_luad --gpu 0

# All GPUs, CLAM and nnMIL only
uv run --package autobench python benchmarks/scripts/run_benchmark.py --dataset tcga_luad \
    --frameworks clam nnmil --all_gpus
```

`--strategies` selects split strategies. Every shipped config defines only
`standard`, which is the default.

### Using with autoMIL

`experiments/` holds an autoMIL overlay for these datasets: `tcga_luad`,
`tcga_hnsc`, `tcga_lgg`, `cptac_pdac`, `cptac_gbm`, `ccrcc`, `ovarian_hrd` and
`placeholder`. CLWD and HANCOCK have none. Each overlay is a directory with
`automil/config.yaml`:

```bash
cd benchmarks/experiments/tcga_luad
uv run automil check
uv run automil orchestrator start
```

The preprint campaign does not use these overlays. It materializes one project
per cell under `campaigns/preprint_130/runtime/`; see
[the operator runbook](../docs/tutorials/run_agentic_campaign.md).

## Adding a New Dataset

1. Copy `datasets/templates/placeholder.yaml` and fill in your dataset's paths, tasks, cohorts, and encoders.
2. Optionally create an overlay in `experiments/your_dataset/automil/config.yaml`, modeled on an existing one.
3. Run preparation: `uv run --package autobench python benchmarks/scripts/run_benchmark.py --dataset your_dataset --prep_only`.

## Architecture

```
benchmarks/
├── datasets/         # Per-dataset YAML configs, grouped by program:
│   ├── tcga/         #   tcga_luad, tcga_hnsc (roster), tcga_lgg
│   ├── cptac/        #   cptac_pdac (roster), cptac_gbm, cptac_ccrcc, ccrcc
│   ├── other/        #   ovarian, clwd, hancock
│   └── templates/    #   tcga_/cptac_/placeholder templates
├── src/autobench/    # Reusable benchmark code
│   ├── config.py     # DatasetConfig loader (YAML → dataclass)
│   ├── data.py       # Generic data loading and filtering
│   ├── campaign*.py  # Campaign manifest, launch, staging and analysis
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
│       └── titan/          # TITAN slide-encoder arm
├── scripts/          # CLI entry points (run_benchmark.py, run_feature_extraction.py, campaign_*.py, slurm/)
├── campaigns/        # preprint_130/: campaign manifest, roster, protocol and operator docs
├── experiments/      # autoMIL overlays per dataset
├── lib/              # External dependencies (CLAM, nnMIL, TRIDENT)
└── tests/            # Test suite
```
