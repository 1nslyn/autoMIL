# Benchmark Experiments Tutorial

End-to-end guide for running MIL benchmark experiments on TCGA (and other) datasets using the autobench pipeline. This is the natural next step after [feature extraction](tcga_feature_extraction_tutorial.md).

## Overview

**Goal:** Train and evaluate Multiple Instance Learning (MIL) models across combinations of:
- **Encoders**, foundation models used for feature extraction (Virchow2, H-optimus-1, UNI2-h)
- **MIL architectures**, CLAM-MB, Simple MIL, DTFD-MIL, and ABMIL (attention-based and baseline aggregators)
- **Tasks**, biomarker prediction targets (e.g., KRAS mutation) plus a shared `os` (overall survival) task
- **Frameworks**, CLAM, nnMIL, DTFD, and ABMIL (TITAN is a separate slide-level arm — see [run_preprint_benchmark.md](run_preprint_benchmark.md))

**Recommended reference** (from `submit_benchmark.sh`'s default `FRAMEWORKS="clam nnmil dtfd abmil"`):
- Encoders: `hoptimus1`, `uni_v2`, `virchow2`
- CLAM model: `clam_mb`
- nnMIL model: `simple_mil`
- DTFD model: `dtfd_mil`
- ABMIL model: `abmil`

This gives a focused benchmark grid of **tasks × 3 encoders × 4 models** (plus extra fan-out from survival-loss variants on the `os` task), enough to compare encoder quality and establish baselines without burning excessive GPU hours.

**Pipeline:** Data preparation → Experiment grid generation → Multi-GPU training → Cross-fold aggregation → Results export

**Output:** Per-experiment `summary.json` with mean/std/95% CI across 5-fold patient-stratified CV, plus aggregated CSV tables.

> **Methodology note.** The wrapper matches the published CLAM README
> invocation for `--lr 2e-4`, `--early_stopping`, and `--weighted_sample`.
> The fold count is the **lab-standard 5-fold** (2026-07) — a deliberate
> deviation from CLAM's `--k 10` (fewer folds give larger, more stable
> per-fold test sets for the imbalanced mutation tasks). Splits are
> **patient-level stratified k-fold** on
> `case_id`. All slides from one case are forced into the same partition.
> Earlier campaign artifacts (pre-2026-05) used slide-level splits, which
> leaks same-patient signal across train/val/test and inflates reported
> AUCs on cohorts with multi-slide cases. Numbers from those runs are not
> directly comparable to current runs.

## Prerequisites

Before running benchmarks, you must have:

1. **Completed feature extraction**, `.h5` feature files for each encoder in `{dataset_root}/trident_output/20x_224px_0px_overlap/features_{encoder}/`
2. **Dataset YAML config**, in `benchmarks/datasets/` (created during the feature extraction tutorial)
3. **Environment variables**, dataset root paths in `benchmarks/.env`
4. **Repository set up**, `automil` and `autobench` packages installed

If you haven't done these, follow the [feature extraction tutorial](tcga_feature_extraction_tutorial.md) first.

> **Important for TCGA datasets:** Your dataset YAML must have `slide_id_transform: "strip_svs"`. The GOLDMARK manifest's `slide_name` column includes the `.svs` extension, but H5/PT feature files do not. Without this transform, the H5→PT conversion silently produces zero files.

### Verify Your Setup

```bash
cd ~/scratch/autoMIL
set -a && source benchmarks/.env && set +a

# Verify dataset config loads
uv run python -c "
from autobench.config import load_dataset_config
ds = load_dataset_config('tcga_{code}')
print(f'Dataset:  {ds.name}')
print(f'Tasks:    {list(ds.tasks.keys())}')
print(f'Encoders: {list(ds.encoder_dims.keys())}')
print(f'WSI dir:  {ds.wsi_dir}')
print(f'Features: {ds.features_base_dir}')
"

# Verify features exist
for encoder in virchow2 hoptimus1 uni_v2; do
    count=$(ls ${AUTOBENCH_TCGA_XXX_ROOT}/trident_output/20x_224px_0px_overlap/features_${encoder}/*.h5 2>/dev/null | wc -l)
    echo "$encoder: $count H5 files"
done
```

Replace `{code}` with your TCGA cancer type (e.g., `luad`) and `TCGA_XXX` with your env var name throughout this guide.

## Understanding the Pipeline

The benchmark pipeline has four phases, all handled automatically by the SLURM script:

```
Phase 1: Data Preparation (automatic)
  mapping CSV → task CSVs (case_id, slide_id, label)
                → patient-stratified k-fold splits on case_id (5-fold default)
                → H5 features → PyTorch .pt tensors

Phase 2: Experiment Grid Generation
  frameworks × strategies × tasks × encoders × models
  → list of ExperimentConfig objects (one per unique combination)

Phase 3: Training
  For each experiment, for each fold:
    → train model on train split
    → evaluate on val + test splits
    → save per-fold metrics

Phase 4: Aggregation
  Per-fold metrics → mean, std, 95% CI (t-distribution)
  → summary.json per experiment
  → aggregated CSV tables
```

### Recommended Model Selection

For the standard benchmark, we use **one model per framework** to keep the grid manageable:

| Framework | Model | Key | Why |
|-----------|-------|-----|-----|
| CLAM | CLAM Multi-Branch | `clam_mb` | Best-performing CLAM variant; attention-based with multiple branches |
| nnMIL | Simple MIL | `simple_mil` | Lightweight baseline; fast to train, establishes a floor |
| DTFD | DTFD MIL | `dtfd_mil` | Two-tier pseudo-bag distillation |
| ABMIL | Attention MIL | `abmil` | Attention-based aggregation, non-gated variant (Ilse et al., 2018) |

This is the default roster: each dataset YAML's `clam_models` / `nnmil_models` / `dtfd_models` / `abmil_models` lists pin exactly this one-model-per-framework selection, and `submit_benchmark.sh` runs all four frameworks by default (`FRAMEWORKS="clam nnmil dtfd abmil"`). A separate TITAN arm (a frozen slide-level foundation model — no tile-encoder sweep) runs via `submit_titan_extract.sh` + `submit_titan.sh`; see [run_preprint_benchmark.md](run_preprint_benchmark.md).

#### All Available Models (for extended benchmarks)

<details>
<summary>CLAM Framework (3 models)</summary>

| Model | Key | Description |
|-------|-----|-------------|
| CLAM Single-Branch | `clam_sb` | Attention-based MIL with single attention branch |
| CLAM Multi-Branch | `clam_mb` | Attention-based MIL with multiple attention branches |
| Standard MIL | `mil` | Basic MIL baseline (mean pooling + classifier) |

</details>

<details>
<summary>nnMIL Framework (up to 7 models)</summary>

| Model | Key | Description |
|-------|-----|-------------|
| Transformer MIL | `trans_mil` | Transformer-based aggregation |
| Deep Sets MIL | `ds_mil` | Permutation-invariant aggregation |
| ILRA MIL | `ilra_mil` | Independent Learned Region Aggregation |
| WiKG MIL | `wikg_mil` | Weighted Instance Knowledge Graph |
| Simple MIL | `simple_mil` | Minimal baseline |
| Vision Transformer | `vision_transformer` | ViT-based bag aggregation |
| RRT | `rrt` | Recurrent Relational Transformer |

> **Note:** Available nnMIL models depend on your dataset YAML's `nnmil_models` list. Not all datasets enable all 7 models. Check your YAML to see which are configured. Attention MIL and DTFD MIL used to be listed here but are now their own frameworks — see below.
>
> **Note:** `vision_transformer`, `rrt`, `trans_mil`, and `ilra_mil` are memory-intensive. The pipeline automatically caps their batch size at 4 and sequence length at 4096.

</details>

<details>
<summary>DTFD Framework (1 model)</summary>

| Model | Key | Description |
|-------|-----|-------------|
| DTFD MIL | `dtfd_mil` | Two-tier pseudo-bag distillation |

Select via `--frameworks dtfd --dtfd_models dtfd_mil` (default: all from the dataset YAML's `dtfd_models`).

</details>

<details>
<summary>ABMIL Framework (2 models)</summary>

| Model | Key | Description |
|-------|-----|-------------|
| Attention MIL | `abmil` | Non-gated attention aggregation (Ilse et al., 2018) |
| Attention MIL (gated) | `abmil_gated` | Gated attention aggregation |

Select via `--frameworks abmil --abmil_models abmil` (default: all from the dataset YAML's `abmil_models`).

</details>

### Metrics Computed

For each fold, the pipeline computes:
- **AUC-ROC**, area under receiver operating characteristic curve
- **Accuracy**, standard classification accuracy
- **Balanced accuracy**, per-class recall averaged (robust to class imbalance)
- **F1 score**, binary F1 or weighted multiclass F1
- **Sensitivity**, true positive rate (CLAM only)
- **Specificity**, true negative rate (CLAM only)

Cross-fold aggregation reports **mean**, **standard deviation**, and **95% confidence intervals** (via t-distribution) for each metric.

## Step-by-Step Guide

### Step 1: Run Benchmark Experiments

> **Note:** Data preparation (task CSVs, stratified splits, H5→PT conversion) runs automatically as Phase 1 inside the SLURM job, no separate step needed. The pipeline is idempotent: re-running skips files that already exist.

#### Option A: Interactive (single GPU, small runs)

For quick tests or small subsets, run interactively on a compute node:

```bash
# Request an interactive session (adjust account and resources)
salloc --account=YOUR_ACCOUNT --gpus-per-node=1 --cpus-per-task=8 --mem=32G --time=4:00:00

# Inside the allocation
cd ~/scratch/autoMIL
set -a && source benchmarks/.env && set +a

# Run a single experiment to test (one encoder, one model, one task)
uv run python benchmarks/scripts/run_benchmark.py \
    --dataset tcga_{code} \
    --gpu 0 \
    --frameworks clam \
    --encoders hoptimus1 \
    --models clam_mb \
    --tasks kras \
    --no_wandb
```

This runs 1 experiment (5 folds by default) and takes ~20-60 minutes depending on dataset size.

#### Option B: SLURM Batch Job (recommended)

For the standard benchmark, use the SLURM submission script. Its interface is **positional** (`<dataset> [n_folds]`), not env-var driven — it always runs the recommended model selection (`clam_mb` + `simple_mil` + `dtfd_mil` + `abmil`) across the 3 encoders (`hoptimus1`, `uni_v2`, `virchow2`) and all of the dataset's tasks, reading the roster straight from the dataset YAML. The only env var it honors is `FRAMEWORKS`, to run a subset of the 4 frameworks.

**First, configure the script for your account:**

The script resolves its project directory from `SLURM_SUBMIT_DIR`, which SLURM sets automatically — so as long as you `sbatch` from the repo root, no path edit is needed. It only falls back to a hardcoded default if `SLURM_SUBMIT_DIR` is unset:

```bash
# Only needed if you don't submit from the repo root:
sed -i "s|/home/yinshuol/scratch/autoMIL/autoMIL|$HOME/scratch/autoMIL/autoMIL|" benchmarks/scripts/slurm/submit_benchmark.sh

# Update the SLURM account:
sed -i "s|--account=rrg-jma|--account=YOUR_ACCOUNT|" benchmarks/scripts/slurm/submit_benchmark.sh
```

The script has no `--mail-user` directive by default (only `--mail-type=BEGIN,END,FAIL`); add one yourself if you want job-completion emails.

**Submit the job with recommended settings:**

```bash
mkdir -p logs

# Standard benchmark: clam_mb + simple_mil + dtfd_mil + abmil, 3 encoders, all tasks, 5-fold
sbatch benchmarks/scripts/slurm/submit_benchmark.sh tcga_{code}

# Or a 10-fold comparison run (2nd positional arg)
sbatch benchmarks/scripts/slurm/submit_benchmark.sh tcga_{code} 10

# Or run only a subset of frameworks via FRAMEWORKS
FRAMEWORKS="clam" sbatch benchmarks/scripts/slurm/submit_benchmark.sh tcga_{code}
FRAMEWORKS="nnmil" sbatch benchmarks/scripts/slurm/submit_benchmark.sh tcga_{code}
FRAMEWORKS="clam nnmil" sbatch benchmarks/scripts/slurm/submit_benchmark.sh tcga_{code}
```

> **Subsetting encoders, models, or tasks?** The SLURM script doesn't expose `ENCODERS`/`MODELS`/`NNMIL_MODELS`/`TASKS`/`SEED` overrides — it always runs the dataset YAML's full roster. To run a narrower slice (e.g., one task only), use [Option A](#option-a-interactive-single-gpu-small-runs) or [Option C](#option-c-multi-gpu-interactive) with `run_benchmark.py` directly.

**Monitor the job:**

```bash
# Check job status
squeue -u $USER

# Watch the output log (job name is "mil_bench")
tail -f logs/bench_mil_bench_*.out
```

The SLURM script:
1. Validates your dataset config and counts total experiments
2. Runs data preparation (Phase 1)
3. Distributes experiments across 4 H100 GPUs with memory-budget scheduling
4. Auto-resubmits on time limit (idempotent, completed experiments are skipped)

#### Option C: Multi-GPU Interactive

If you have a multi-GPU allocation:

```bash
# Use all available GPUs with recommended model selection
uv run python benchmarks/scripts/run_benchmark.py \
    --dataset tcga_{code} \
    --all_gpus \
    --frameworks clam nnmil \
    --encoders hoptimus1 uni_v2 virchow2 \
    --models clam_mb \
    --nnmil_models simple_mil \
    --no_wandb

# Or specify GPU indices
uv run python benchmarks/scripts/run_benchmark.py \
    --dataset tcga_{code} \
    --gpus 0 1 2 3 \
    --frameworks clam nnmil \
    --encoders hoptimus1 uni_v2 virchow2 \
    --models clam_mb \
    --nnmil_models simple_mil \
    --no_wandb
```

### Step 2: Understand the Experiment Grid

The pipeline generates a near-Cartesian product: **frameworks × strategies × tasks × encoders × models**, pruned by feasibility — a survival task's loss variants (`cox`, `nllsurv`) each become a separate experiment, and some frameworks restrict which loss/model combinations are valid. With the recommended settings, the grid is focused and manageable:

**Standard grid for a TCGA dataset with 2 tasks (a mutation task + `os` survival):**

| Framework | Model | Grid | Experiments |
|-----------|-------|------|-------------|
| CLAM | `clam_mb` | 3 encoders × (1 mutation + 1 survival-loss variant) | 6 |
| nnMIL | `simple_mil` | 3 encoders × (1 mutation + 2 survival-loss variants) | 9 |
| DTFD | `dtfd_mil` | 3 encoders × (1 mutation + 1 survival-loss variant) | 6 |
| ABMIL | `abmil` | 3 encoders × (1 mutation + 2 survival-loss variants) | 9 |
| **Total** | | | **30 experiments, 150 fold trainings (5-fold)** |

CLAM and DTFD only run `nllsurv` for `os` here (CLAM's `cox` needs `clam_sb`, not the recommended `clam_mb`; DTFD's pseudo-bag distillation only supports `nllsurv`), while nnMIL and ABMIL run both `cox` and `nllsurv`, so they fan out further. Add the separate TITAN arm (3 more experiments — see [run_preprint_benchmark.md](run_preprint_benchmark.md)) and the full grid is 33 experiments.

To verify the exact count for your dataset:

```bash
uv run python -c "
from autobench.config import load_dataset_config
from autobench.pipeline.config import BenchmarkConfig, Framework, build_registries, generate_all_experiments

ds = load_dataset_config('tcga_{code}')
registries = build_registries(ds)

# Standard grid with the recommended (default) model selection
cfg = BenchmarkConfig.from_dataset_config(
    ds,
    frameworks=[Framework.CLAM, Framework.NNMIL, Framework.DTFD, Framework.ABMIL],
)
exps = generate_all_experiments(cfg, registries)

print(f'Total experiments: {len(exps)}')
print(f'Total fold trainings: {len(exps) * 5}')
print()

# Break down by framework
from collections import Counter
fw_counts = Counter(e.framework.value for e in exps)
for fw, count in fw_counts.items():
    print(f'  {fw}: {count} experiments ({count * 5} folds)')
print()

# List all experiments
for e in exps:
    print(f'  {e.experiment_id}')
"
```

> **Extended benchmarks:** If you want to run all available models, omit the `--models`, `--nnmil_models`, `--dtfd_models`, and `--abmil_models` flags. This expands the grid significantly (3 CLAM + up to 7 nnMIL + 1 DTFD + 2 ABMIL models), so plan for longer wall times. See [All Available Models](#all-available-models-for-extended-benchmarks) above.

### Step 3: Understanding the Output

#### Data Preparation Output (Phase 1)

The SLURM script's Phase 1 creates:

```
{benchmark_dir}/
├── dataset_csv/
│   ├── kras.csv               # slide_id, case_id, label
│   └── os.csv                 # slide_id, case_id, status, time (survival task)
├── splits/
│   └── standard/
│       ├── kras/
│       │   ├── splits_0.csv  # fold 0: train/val/test slide IDs
│       │   └── ...           # splits_1.csv through splits_4.csv
│       └── os/
│           └── ...
└── features/
    ├── virchow2/
    │   └── pt_files/         # .pt tensors converted from .h5
    ├── hoptimus1/
    │   └── pt_files/
    └── uni_v2/
        └── pt_files/
```

#### Training Results (Phase 3-4)

After training completes, the results directory looks like:

```
{benchmark_dir}/results/
├── _completed.json                            # List of completed experiment IDs
├── _failed.json                               # Failed experiments with error details
├── clam/
│   └── standard/
│       ├── kras/
│       │   ├── hoptimus1/
│       │   │   └── clam_mb/
│       │   │       ├── config.json            # Experiment configuration
│       │   │       ├── summary.json           # Aggregated metrics (this is what you want)
│       │   │       ├── fold_0/
│       │   │       │   ├── metrics.json       # Per-fold test + val metrics
│       │   │       │   ├── predictions.csv    # Per-slide predictions
│       │   │       │   └── s_0_checkpoint.pt  # Model checkpoint
│       │   │       ├── fold_1/
│       │   │       └── ...
│       │   ├── virchow2/
│       │   │   └── clam_mb/
│       │   │       └── ...
│       │   └── uni_v2/
│       │       └── clam_mb/
│       │           └── ...
│       └── os/                                 # survival task: one extra level per loss variant
│           └── hoptimus1/
│               └── clam_mb/
│                   └── nllsurv/                 # CLAM+os is nllsurv-only here (cox needs clam_sb)
│                       └── ...
└── nnmil/
    └── standard/
        ├── kras/
        │   ├── hoptimus1/
        │   │   └── simple_mil/
        │   │       └── ...
        │   ├── virchow2/
        │   └── uni_v2/
        └── os/
            └── hoptimus1/
                └── simple_mil/
                    ├── cox/
                    └── nllsurv/
```

#### The summary.json File

This is the key output. Each experiment produces one:

```json
{
  "experiment_id": "clam__standard__kras__hoptimus1__clam_mb__s42",
  "task": "kras",
  "encoder": "hoptimus1",
  "embed_dim": 1536,
  "model_type": "clam_mb",
  "framework": "clam",
  "strategy": "standard",
  "n_folds": 5,
  "seed": 42,
  "test": {
    "auc_roc":           {"mean": 0.72, "std": 0.08, "ci_low": 0.62, "ci_high": 0.82},
    "accuracy":          {"mean": 0.85, "std": 0.03, "ci_low": 0.81, "ci_high": 0.89},
    "balanced_accuracy": {"mean": 0.68, "std": 0.07, "ci_low": 0.59, "ci_high": 0.77},
    "f1":                {"mean": 0.45, "std": 0.10, "ci_low": 0.33, "ci_high": 0.57},
    "sensitivity":       {"mean": 0.55, "std": 0.12, "ci_low": 0.40, "ci_high": 0.70},
    "specificity":       {"mean": 0.90, "std": 0.03, "ci_low": 0.86, "ci_high": 0.94}
  },
  "val": { ... },
  "per_fold_test": [ ... ],
  "per_fold_val": [ ... ]
}
```

#### The predictions.csv File (CLAM only)

Per-fold, per-slide predictions for detailed analysis. **Only CLAM** saves `predictions.csv`; nnMIL saves `metrics.json` per fold but not per-slide predictions.

```csv
slide_id,y_true,y_prob_0,y_prob_1,y_hat
TCGA-05-4244-01Z-00-DX1.abc123,0,0.82,0.18,0
TCGA-05-4249-01Z-00-DX1.def456,1,0.35,0.65,1
...
```

### Step 4: Analyze Results

#### Quick Summary

```bash
# View all completed experiments
uv run python -c "
import json, pathlib

results_dir = pathlib.Path('${AUTOBENCH_TCGA_XXX_ROOT}/benchmark/results')

# Collect all summaries
summaries = []
for p in results_dir.rglob('summary.json'):
    summaries.append(json.loads(p.read_text()))

# Sort by test AUC
summaries.sort(key=lambda s: s['test']['auc_roc']['mean'], reverse=True)

# Print leaderboard
print(f'{'Experiment':<60} {'AUC':>8} {'BAcc':>8} {'F1':>8}')
print('=' * 88)
for s in summaries:
    t = s['test']
    print(f'{s[\"experiment_id\"]:<60} {t[\"auc_roc\"][\"mean\"]:>7.3f} {t[\"balanced_accuracy\"][\"mean\"]:>7.3f} {t[\"f1\"][\"mean\"]:>7.3f}')
"
```

#### Check for Failed Experiments

```bash
uv run python -c "
import json, pathlib

failed_path = pathlib.Path('${AUTOBENCH_TCGA_XXX_ROOT}/benchmark/results/_failed.json')
if failed_path.exists():
    failed = json.loads(failed_path.read_text())
    print(f'Failed experiments: {len(failed)}')
    for exp_id, info in failed.items():
        print(f'  {exp_id}: {info[\"reason\"]}, {info.get(\"detail\", \"\")[:80]}')
else:
    print('No failures recorded.')
"
```

#### Compare Encoders

```bash
uv run python -c "
import json, pathlib
from collections import defaultdict

results_dir = pathlib.Path('${AUTOBENCH_TCGA_XXX_ROOT}/benchmark/results')
by_encoder = defaultdict(list)

for p in results_dir.rglob('summary.json'):
    s = json.loads(p.read_text())
    by_encoder[s['encoder']].append(s['test']['auc_roc']['mean'])

print('Encoder Performance (mean test AUC across all experiments):')
for enc, aucs in sorted(by_encoder.items(), key=lambda x: -sum(x[1])/len(x[1])):
    avg = sum(aucs) / len(aucs)
    print(f'  {enc:<15} {avg:.3f}  (n={len(aucs)} experiments)')
"
```

#### Compare Models

```bash
uv run python -c "
import json, pathlib
from collections import defaultdict

results_dir = pathlib.Path('${AUTOBENCH_TCGA_XXX_ROOT}/benchmark/results')
by_model = defaultdict(list)

for p in results_dir.rglob('summary.json'):
    s = json.loads(p.read_text())
    key = f'{s[\"framework\"]}/{s[\"model_type\"]}'
    by_model[key].append(s['test']['auc_roc']['mean'])

print('Model Performance (mean test AUC across all experiments):')
for model, aucs in sorted(by_model.items(), key=lambda x: -sum(x[1])/len(x[1])):
    avg = sum(aucs) / len(aucs)
    print(f'  {model:<25} {avg:.3f}  (n={len(aucs)} experiments)')
"
```

### Step 5: Update the Tracking Sheet

After benchmarks complete, update your row in the [tracking sheet](https://docs.google.com/spreadsheets/d/1DVzgG7EfkQwOw-hjWqI8gwagAzdG9jG-fR8z7-IDbEk/edit?usp=sharing):

- Mark **Benchmark:CLAM** and/or **Benchmark:nnMIL** as complete
- Record best AUC per task in the **Results** column
- Note any failed experiments or issues in **Notes**

## CLI Reference

### Full Argument List

```
uv run python benchmarks/scripts/run_benchmark.py

Required:
  --dataset DATASET         Dataset config name (e.g., 'tcga_luad') or path to YAML

GPU Selection (mutually exclusive):
  --gpu GPU                 Single GPU index (default: 0)
  --all_gpus                Use all available GPUs
  --gpus GPU [GPU ...]      Specific GPU indices

Path Overrides:
  --benchmark_dir DIR       Override benchmark directory from YAML
  --mapping_csv PATH        Override mapping CSV path
  --features_base_dir DIR   Override features base directory

Experiment Grid:
  --encoders E [E ...]      Encoder keys (default: all from dataset config)
  --models M [M ...]        CLAM model types (default: all from dataset config)
  --tasks T [T ...]         Task names (default: all from dataset config)
  --strategies S [S ...]    Split strategies (default: first from dataset config)
  --frameworks {clam,nnmil,dtfd,titan,abmil} [...]   Model frameworks (default: clam)
  --nnmil_models M [M ...]  nnMIL model types (default: all from dataset config)
  --dtfd_models M [M ...]   DTFD model types (default: all from dataset config)
  --abmil_models M [M ...]  ABMIL model types (default: all from dataset config)

Training:
  --max_epochs N            Maximum training epochs (default: 200)
  --lr RATE                 Learning rate (default: 2e-4, matches CLAM README)
  --seed N                  Random seed (default: 42)
  --n_folds N               Number of CV folds (default: 5, lab standard — deviates from CLAM README --k 10)
  --no_early_stopping       Disable early stopping (default: on, matches CLAM README)
  --patience N              Early stopping patience (default: 20)
  --stop_epoch N            Minimum epochs before early stopping (default: 50)
  --no_weighted_sample      Disable class-weighted sampling (default: on, matches CLAM README)

Logging:
  --wandb_project NAME      W&B project (default: {dataset}-benchmark)
  --no_wandb                Disable W&B logging

Other:
  --experiments_per_gpu N   Max concurrent worker processes per GPU (default: auto-detect).
                            VRAM-budget scheduling still gates actual submission;
                            this only binds for small-VRAM model sweeps.
  --prep_only               Only run data preparation, skip training (used internally by SLURM script)
```

### SLURM Environment Variables

`submit_benchmark.sh` takes the dataset and fold count as **positional** arguments, not env vars — it always runs the dataset YAML's full recommended roster (all encoders, all tasks, `clam_mb` + `simple_mil` + `dtfd_mil` + `abmil`). The only env var it reads is `FRAMEWORKS`:

```bash
# Positional: <dataset> [n_folds]
sbatch benchmarks/scripts/slurm/submit_benchmark.sh tcga_luad        # 5-fold (default)
sbatch benchmarks/scripts/slurm/submit_benchmark.sh tcga_luad 10     # 10-fold comparison run

# FRAMEWORKS (optional, default: "clam nnmil dtfd abmil")
FRAMEWORKS="clam" sbatch benchmarks/scripts/slurm/submit_benchmark.sh tcga_luad
```

There's no `ENCODERS`/`MODELS`/`NNMIL_MODELS`/`TASKS`/`SEED` override — the script doesn't pass those flags through to `run_benchmark.py`. If you need to subset encoders, models, or tasks, run `run_benchmark.py` directly ([Option A](#option-a-interactive-single-gpu-small-runs) / [Option C](#option-c-multi-gpu-interactive)).

The legacy `DATASET=... sbatch ...` env-var form still works as a fallback (so does `N_FOLDS=`), but the positional form is preferred.

**Example submissions:**

```bash
# Standard benchmark (recommended, 5-fold, all 4 frameworks)
sbatch benchmarks/scripts/slurm/submit_benchmark.sh tcga_luad

# CLAM only
FRAMEWORKS="clam" sbatch benchmarks/scripts/slurm/submit_benchmark.sh tcga_luad

# 10-fold comparison run, all 4 frameworks
sbatch benchmarks/scripts/slurm/submit_benchmark.sh tcga_luad 10
```

## Timing Estimates

These are rough estimates based on a single H100 GPU. Multi-GPU (4× H100) divides wall time roughly by 4.

| Model | Time per fold | VRAM | Notes |
|-------|--------------|------|-------|
| `clam_mb` | 5-15 min | ~3-4 GB | Fast, recommended CLAM model |
| `simple_mil` | 10-20 min | ~3 GB | Fast, recommended nnMIL baseline |
| `ds_mil` | 10-30 min | ~3-4 GB | Extended benchmark |
| `trans_mil`, `vision_transformer`, `rrt` | 30-90 min | ~8-16 GB | Extended benchmark, memory-intensive |

**Standard benchmark** (recommended, default 4-framework roster): 30 experiments × 5 folds = 150 fold trainings for a 2-task dataset (mutation + `os` survival). On 4× H100: **~2-6 hours**.

**Extended benchmark** (all models — 3 CLAM + 7 nnMIL + 1 DTFD + 2 ABMIL model types): 93 experiments × 5 folds = 465 fold trainings for the same dataset. On 4× H100: **~12-24 hours**.

For large datasets (>800 slides), increase the SLURM time limit:

```bash
DATASET=tcga_brca sbatch --time=2-00:00:00 benchmarks/scripts/slurm/submit_benchmark.sh
```

## Resuming Interrupted Runs

The pipeline is fully idempotent. If a job times out or fails:

1. **Completed experiments** are tracked in `results/_completed.json` and skipped on re-run
2. **Per-fold checkpoints**, if fold 0-2 finished but fold 3 failed, folds 0-2 are skipped
3. **Auto-continuation**, the SLURM script detects time limits and resubmits automatically

To manually resume:

```bash
# Just resubmit, same command, same args
DATASET=tcga_{code} sbatch benchmarks/scripts/slurm/submit_benchmark.sh
```

To check progress:

```bash
uv run python -c "
import json, pathlib

results_dir = pathlib.Path('${AUTOBENCH_TCGA_XXX_ROOT}/benchmark/results')

completed = json.loads((results_dir / '_completed.json').read_text()) if (results_dir / '_completed.json').exists() else []
failed = json.loads((results_dir / '_failed.json').read_text()) if (results_dir / '_failed.json').exists() else {}

print(f'Completed: {len(completed)}')
print(f'Failed:    {len(failed)}')
"
```

## Troubleshooting

### CUDA Out of Memory (OOM)

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

The multi-GPU scheduler handles OOM automatically by retrying with a bumped VRAM estimate (1.5× multiplier, up to 3 retries). A single failed experiment does **not** kill the run — the orchestrator marks it failed in `_failed.json` and continues. If persistent:

- Reduce the model set: skip `vision_transformer` and `rrt` (highest VRAM)
- Run memory-intensive models separately on a single GPU (sequential):
  ```bash
  uv run python benchmarks/scripts/run_benchmark.py \
      --dataset tcga_{code} \
      --gpu 0 \
      --frameworks nnmil \
      --nnmil_models vision_transformer rrt \
      --no_wandb
  ```

> **Non-OOM error behavior.** Only `torch.cuda.OutOfMemoryError` is treated as retriable. Any other exception in a worker (CUDA assertion, missing file, dataloader crash, etc.) raises `RuntimeError` inside the orchestrator and **aborts the entire multi-GPU run**. Check `logs/bench_*.err` and `_failed.json` for details, then resubmit (the pipeline is idempotent and resumes from where it left off).

### Missing Feature Files

```
Warning: X slides have no .pt file, skipping
```

This happens when some slides failed during feature extraction. Check:
1. Compare H5 count to slide count: `ls features_{encoder}/*.h5 | wc -l`
2. Check `trident_output/skipped_slides.txt` for extraction failures
3. Re-extract missing slides if needed (see [feature extraction tutorial](tcga_feature_extraction_tutorial.md#troubleshooting))

The pipeline continues with available slides, a few missing slides won't invalidate results.

### Config Loading Errors

```
FileNotFoundError: Dataset config not found
```

- Ensure YAML is in `benchmarks/datasets/` with correct filename
- Ensure env vars are set in `benchmarks/.env`
- Ensure `.env` is sourced: `set -a && source benchmarks/.env && set +a`

### nnMIL Plan Generation Fails

```
KeyError: 'slide_id' not in dataset CSV
```

Check your dataset YAML column mappings:
- `slide_id_column` must match the CSV column name exactly
- `case_id_column` must match the CSV column for patient IDs
- For TCGA/GOLDMARK: `slide_id_column: "slide_name"`, `case_id_column: "sample_names"`

### Job Timeout

If 24 hours isn't enough (large dataset + many models):

```bash
# Increase wall time
DATASET=tcga_{code} sbatch --time=2-00:00:00 benchmarks/scripts/slurm/submit_benchmark.sh

# Or split into two jobs: CLAM first, then nnMIL
DATASET=tcga_{code} FRAMEWORKS="clam" sbatch benchmarks/scripts/slurm/submit_benchmark.sh
DATASET=tcga_{code} FRAMEWORKS="nnmil" sbatch benchmarks/scripts/slurm/submit_benchmark.sh
```

### W&B Logging Issues

If W&B causes problems, disable it:
```bash
--no_wandb
```

The SLURM script disables W&B by default. For interactive runs, add `--no_wandb` or ensure `WANDB_API_KEY` is set in `benchmarks/.env`.

## Quick Reference

| Step | Command |
|------|---------|
| Verify setup | `uv run python -c "from autobench.config import load_dataset_config; print(load_dataset_config('tcga_{code}').name)"` |
| Single experiment (interactive) | `uv run python benchmarks/scripts/run_benchmark.py --dataset tcga_{code} --gpu 0 --encoders hoptimus1 --models clam_mb --tasks kras --no_wandb` |
| Standard benchmark (SLURM, 5-fold, all 4 frameworks) | `sbatch benchmarks/scripts/slurm/submit_benchmark.sh tcga_{code}` |
| CLAM only (SLURM) | `FRAMEWORKS="clam" sbatch benchmarks/scripts/slurm/submit_benchmark.sh tcga_{code}` |
| 10-fold comparison run (SLURM) | `sbatch benchmarks/scripts/slurm/submit_benchmark.sh tcga_{code} 10` |
| Check job status | `squeue -u $USER` |
| Monitor logs | `tail -f logs/bench_mil_bench_*.out` |
| Resume after timeout | Resubmit the same command (idempotent) |
| Count completed | `uv run python -c "import json; print(len(json.loads(open('results/_completed.json').read())))"` |

## Questions?

Ask me directly :)
