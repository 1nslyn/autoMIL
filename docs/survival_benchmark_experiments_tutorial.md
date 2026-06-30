# Survival Benchmark Experiments Tutorial

End-to-end guide for running survival (time-to-event) MIL benchmark experiments on TCGA, CPTAC, and other datasets using the autobench pipeline. This is the survival counterpart of the [classification benchmark experiments tutorial](benchmark_experiments_tutorial.md).

Replace `{dataset}` with your dataset config name (e.g., `cptac_ccrcc`, `tcga_luad`) and `AUTOBENCH_{DATASET}_ROOT` with your env-var name throughout this guide.

## Overview

**Goal:** Train and evaluate Multiple Instance Learning (MIL) models for **overall survival (OS)** across combinations of:
- **Encoders**, foundation models used for feature extraction (e.g. Virchow2, H-optimus-1, UNI2-h)
- **Frameworks + MIL architectures**, CLAM (`clam_sb`/`clam_mb`) and nnMIL (e.g. `ab_mil`/`trans_mil`)
- **Survival losses**, Cox proportional-hazards (`cox`) and discrete-time negative-log-likelihood (`nllsurv`)

**Valid model × loss grid.** Cox emits a single risk score, so `clam_mb`'s per-class branches are degenerate at n=1 — Cox is `clam_sb`-only. `nllsurv` emits `nll_bins` hazard logits and works for both:

| Framework | Model                                       | `cox` |         `nllsurv`          |
| --------- | ------------------------------------------- | :---: | :------------------------: |
| CLAM      | `clam_sb`                                   |   ✅   |             ✅              |
| CLAM      | `clam_mb`                                   |   —   | ✅ (per-time-bin attention) |
| nnMIL     | attention models (`ab_mil`, `trans_mil`, …) |   ✅   |             ✅              |

> The nnMIL survival models that actually run come from your dataset YAML's `nnmil_models` list — adjust it per dataset.

**Pipeline:** OS labels → Data preparation → Experiment grid → Multi-GPU training → Cross-fold aggregation → Results export

**Output:** Per-experiment `summary.json` with the **concordance index (c-index)** — mean/std/95% CI across **5-fold patient-stratified CV** — plus aggregated CSV tables.

> **Methodology note (survival ≠ classification).** Survival experiments deliberately differ from the classification pipeline:
> - **5-fold CV**, not 10. With few events, 10-fold leaves only a couple of events per test fold and the c-index becomes near-random; 5-fold doubles the per-fold events and matches nnMIL's native survival convention.
> - **Model selection on validation loss**, not the validation c-index. With few val events the val c-index is a coin flip, so maximizing it overfits to noise; the survival loss uses every sample and is stable.
> - **`nllsurv` time bins from event (uncensored) times only** (the PORPOISE/MCAT convention), so censored outliers — including any corrupted negative follow-up times — do not shift the bin boundaries.
> - **Patient-level c-index** via `scikit-survival`'s `concordance_index_censored`, NaN-safe (undefined folds are dropped from the cross-fold mean, never counted as 0.0).

## Prerequisites

Before running survival benchmarks, you must have:

1. **Completed feature extraction**, `.h5` feature files for each encoder in `{dataset_root}/trident_output/20x_224px_0px_overlap/features_{encoder}/`
2. **Environment variables**, dataset root paths in `benchmarks/.env`
3. **`scikit-survival` installed**, provides the validated c-index estimator (declared in `benchmarks/pyproject.toml`; `uv sync` installs it)

### Verify your setup

```bash
cd ~/scratch/autoMIL
set -a && source benchmarks/.env && set +a

# Dataset config loads and features are discoverable
uv run python -c "
from autobench.config import load_dataset_config
ds = load_dataset_config('{dataset}')
print(f'Dataset:  {ds.name}')
print(f'Tasks:    {list(ds.tasks.keys())}')
print(f'Encoders: {list(ds.encoder_dims.keys())}')
"

# scikit-survival is importable (else the c-index silently falls back to a buggy estimator)
uv run --package autobench python -c "import sksurv.metrics; print('scikit-survival OK')"
```

## Part 1: Prepare survival (OS) labels

A one-time step per cohort, done after the manifest exists. Joins OS columns from a GDC clinical export into `normalized_manifest.csv`.

### Step 1.1: Download clinical.tsv from GDC

Each dataset's clinical data lives on the GDC. How you reach the GDC page differs by program:

- **CPTAC:** open the dataset's **TCIA** page and follow the **GDC** link in its **External Resources** section.
- **TCGA:** open the dataset's **GDC** page directly.

On the GDC page, choose **Clinical → TSV** and download. The download arrives as an archive — **unzip it**, then place the resulting `clinical.tsv` inside that dataset's own folder, as `datasets/{DATASET}/clinical.tsv` (every dataset keeps its own copy):

```
datasets/CPTAC-CCRCC/clinical.tsv
datasets/TCGA-LUAD/clinical.tsv
# ... one clinical.tsv per dataset folder
```

**Columns used for survival:**

| GDC column                         | Meaning                                                     |
| ---------------------------------- | ----------------------------------------------------------- |
| `cases.index_date`                 | Reference event for all day offsets — typically `Diagnosis` |
| `demographic.vital_status`         | `Alive` or `Dead`                                           |
| `demographic.days_to_death`        | Days from index date to death (Dead only)                   |
| `diagnoses.days_to_last_follow_up` | Days from index date to last contact (Alive)                |

### Step 1.2: Join OS columns to the manifest

```bash
uv run python benchmarks/scripts/add_os_to_manifest.py \
    --manifest datasets/{DATASET}/normalized_manifest.csv \
    --clinical datasets/{DATASET}/clinical.tsv
```

This overwrites the manifest in place. Counts are per slide and vary by dataset — example for CPTAC-CCRCC (245 slides):

```
Slides: 245
  OS_event=1 (Dead):         52
  OS_event=0 (Alive):        193
  OS_event=NaN (Not Rep.):   0  <- dropped by survival task dropna
Saved -> datasets/{DATASET}/normalized_manifest.csv
```

### Step 1.3: Verify the labels

```bash
uv run python -c "
import pandas as pd
df = pd.read_csv('datasets/{DATASET}/normalized_manifest.csv')
print(df[['case_id', 'OS_event', 'OS_time']].head())
print('Event dist:', df['OS_event'].value_counts().to_dict())
print(f'Time range: {df.OS_time.min():.0f} - {df.OS_time.max():.0f} days')
print(f'NaN rows:   {df.OS_event.isna().sum()}')
"
```

### Step 1.4: Enable the survival task in the dataset YAML

In `benchmarks/datasets/{dataset}.yaml`, add an `os` entry under `tasks` and to `task_strategy_feasibility`:

```yaml
tasks:
  # ... existing classification tasks ...
  os:
    task_type: survival
    event_col: OS_event
    time_col: OS_time
    survival_losses: [cox, nllsurv]
    nll_bins: 4

task_strategy_feasibility:
  # ... other tasks ...
  os: ["standard"]
```

### Step 1.5: Record the OS task counts for the benchmark sheet

In the benchmark tracking [sheet](https://docs.google.com/spreadsheets/d/1DVzgG7EfkQwOw-hjWqI8gwagAzdG9jG-fR8z7-IDbEk/edit?gid=0#gid=0), record the **OS** task in the format **`OS (total: event, non-event, not reported)`** — the total number of patients followed by the per-bucket headcounts, always in the order **event, non-event, not reported**:

| Bucket           | Condition         | Meaning                                                |
| ---------------- | ----------------- | ------------------------------------------------------ |
| **event**        | `OS_event == 1`   | Death observed (Dead)                                  |
| **non-event**    | `OS_event == 0`   | Censored — alive at last follow-up                     |
| **not reported** | `OS_event` is NaN | Missing vital status / follow-up (dropped by the task) |

Count at the **patient (case) level**, not slides — survival is patient-stratified and the c-index is patient-level, so one patient with several slides is one event:

```bash
uv run --package autobench python -c "
import pandas as pd
df = pd.read_csv('datasets/{DATASET}/normalized_manifest.csv')
pt  = df.drop_duplicates('case_id')
ev   = int((pt.OS_event == 1).sum())
non  = int((pt.OS_event == 0).sum())
miss = int(pt.OS_event.isna().sum())
print(f'OS (total {len(pt)}: event {ev}, non-event {non}, not reported {miss})')
"
```

Example for CPTAC-CCRCC (103 patients / 245 slides):

```
OS (total 103: event 21, non-event 82, not reported 0)
```

So the sheet's OS cell reads **`OS (total 103: event 21, non-event 82, not reported 0)`**.

> Patient- and slide-level counts differ — CPTAC-CCRCC is `event 21, non-event 82, not reported 0` per patient but `52, 193, 0` per slide (Step 1.2's manifest output is slide-level). **Record the patient-level totals in the sheet.** The not-reported count is patients dropped by the survival task's `dropna` on event/time, so the trained cohort is event + non-event.

## Understanding the Pipeline

The benchmark pipeline has four phases, all handled automatically by the runner:

```
Phase 1: Data Preparation (automatic, idempotent)
  manifest → os task CSV (case_id, slide_id, status, time)
           → patient-stratified 5-fold splits on case_id, stratified on status
           → H5 features → PyTorch .pt tensors
           → (nnMIL) dataset_plan.json per (encoder, loss)

Phase 2: Experiment Grid Generation
  frameworks × tasks × encoders × (model, loss) valid combos
  → list of ExperimentConfig objects (one per unique combination)

Phase 3: Training (per experiment, per fold)
  → train survival head on the train split (cox risk set / nllsurv NLL)
  → select the best checkpoint on validation LOSS
  → score test + val concordance index (patient-level)

Phase 4: Aggregation
  Per-fold c-index → mean, std, 95% CI (NaN folds dropped)
  → summary.json per experiment
  → aggregated CSV tables (with a survival_loss column)
```

### Survival model selection

| Framework | Model            | Key                      | Losses       | Why                                                                      |
| --------- | ---------------- | ------------------------ | ------------ | ------------------------------------------------------------------------ |
| CLAM      | Single-Branch    | `clam_sb`                | cox, nllsurv | Shared attention → single risk score (cox) or hazard bins (nllsurv)      |
| CLAM      | Multi-Branch     | `clam_mb`                | nllsurv      | One attention branch per time-bin (degenerate for the single-output cox) |
| nnMIL     | attention models | `ab_mil`, `trans_mil`, … | cox, nllsurv | From the dataset YAML's `nnmil_models`                                   |

> CLAM survival reuses CLAM's attention model trained by an **adapter-side survival loop** (CLAM's vendored classification loop is untouched); instance-level clustering is disabled (no classes for survival).

### Metric computed

The single survival metric is the **concordance index (c-index)** — the fraction of comparable patient pairs the model ranks correctly (higher risk ↔ earlier event). 0.5 is chance, 1.0 is perfect. Computed at the **patient level** (slide risks averaged per case) via `scikit-survival`. Classification metrics (AUC, accuracy, …) are **not** applicable and appear as `NaN` in any shared summary table — this is expected, not a failure.

## Step-by-Step Guide

### Step 1: Run survival experiments

> Data preparation (task CSV, 5-fold stratified splits, H5→PT, nnMIL plans) runs automatically as Phase 1. The pipeline is idempotent: re-running skips files that already exist.

#### Option A: Single experiment (interactive, one GPU)

For a quick test — one framework, one encoder, the `os` task:

```bash
cd ~/scratch/autoMIL
set -a && source benchmarks/.env && set +a

uv run python benchmarks/scripts/run_benchmark.py \
    --dataset {dataset} \
    --gpu 0 \
    --tasks os \
    --frameworks clam \
    --encoders uni_v2 \
    --no_wandb
```

This trains the valid CLAM survival combos for that encoder (`clam_sb`×{cox,nllsurv}, `clam_mb`×nllsurv), 5 folds each.

#### Option B: Multi-GPU (recommended for a full run)

Run the whole survival grid across all GPUs. nnMIL and CLAM are separate `--frameworks` invocations:

```bash
# nnMIL survival
uv run python benchmarks/scripts/run_benchmark.py \
    --dataset {dataset} --tasks os --frameworks nnmil --all_gpus --no_wandb

# CLAM survival (clam_sb×{cox,nllsurv}, clam_mb×nllsurv)
uv run python benchmarks/scripts/run_benchmark.py \
    --dataset {dataset} --tasks os --frameworks clam --all_gpus --no_wandb
```

To run detached and keep a log:

```bash
mkdir -p logs
setsid bash -c '
  cd ~/scratch/autoMIL
  set -a; source benchmarks/.env; set +a
  uv run python benchmarks/scripts/run_benchmark.py \
    --dataset {dataset} --tasks os --frameworks clam --all_gpus --no_wandb
' >> logs/{dataset}_surv_clam.out 2>&1 &
```

#### Option C: SLURM batch job

On a cluster, use the generic survival submission script `submit_survival_benchmark.sh` (env-var driven, mirrors `submit_benchmark.sh`; update the `PROJECT_DIR`, `--account`, and `--mail-user` for your environment), then:

```bash
mkdir -p logs
# DATASET defaults to cptac_ccrcc; override per dataset. FRAMEWORKS defaults to "clam nnmil".
DATASET={dataset} sbatch benchmarks/scripts/submit_survival_benchmark.sh

# Subset overrides (env vars): FRAMEWORKS, ENCODERS, MODELS, NNMIL_MODELS, TASKS, SEED
DATASET={dataset} FRAMEWORKS="clam" ENCODERS="virchow2" sbatch benchmarks/scripts/submit_survival_benchmark.sh

squeue -u $USER
tail -f logs/bench_autobench_surv_*.out
```

### Step 2: Understand the experiment grid

The survival grid is **frameworks × encoders × valid (model, loss) combos** for the `os` task. Per encoder: nnMIL = (each attention model × {cox, nllsurv}); CLAM = `clam_sb`×{cox,nllsurv} + `clam_mb`×nllsurv (3). Multiply by the number of encoders.

Verify the exact grid for your dataset:

```bash
uv run python -c "
from autobench.config import load_dataset_config
from autobench.pipeline.config import BenchmarkConfig, Framework, build_registries, generate_all_experiments

ds = load_dataset_config('{dataset}')
cfg = BenchmarkConfig.from_dataset_config(ds, frameworks=[Framework.CLAM, Framework.NNMIL])
exps = [e for e in generate_all_experiments(cfg, build_registries(ds)) if e.task.name == 'os']
print(f'os experiments: {len(exps)} ({len(exps) * 5} fold trainings)')
for e in sorted(set((e.framework.value, e.model.model_type, e.survival_loss) for e in exps)):
    print(' ', e)
"
```

### Step 3: Understand the output

After training, results live under `{benchmark_dir}/results/{framework}/standard/os/{encoder}/{model}/{loss}/`:

```
results/
├── _completed.json          # completed experiment IDs (gates re-runs)
├── nnmil/standard/os/uni_v2/trans_mil/cox/
│   ├── config.json
│   ├── summary.json         # aggregated c-index (this is what you want)
│   └── fold_0/ … fold_4/    # per-fold metrics.json (+ checkpoints)
└── clam/standard/os/uni_v2/clam_sb/cox/
    └── ...
```

Aggregated tables are written to `{benchmark_dir}/aggregated/{framework}/standard.csv`, with a **`survival_loss`** column so cox and nllsurv rows are distinguishable, and `test_c_index_mean/std/ci_low/ci_high` columns.

`summary.json` for a survival experiment:

```json
{
  "experiment_id": "clam__standard__os__uni_v2__clam_sb__s42__cox",
  "task": "os",
  "encoder": "uni_v2",
  "model_type": "clam_sb",
  "survival_loss": "cox",
  "n_folds": 5,
  "test": { "c_index": {"mean": 0.62, "std": 0.13, "ci_low": 0.49, "ci_high": 0.74} },
  "val":  { "c_index": {"mean": 0.70, "std": 0.10, "ci_low": 0.60, "ci_high": 0.80} },
  "per_fold_test": [ ... ],
  "per_fold_val": [ ... ]
}
```

### Step 4: Analyze results

```bash
# c-index leaderboard across both frameworks
uv run --package autobench python -c "
import os, pandas as pd
root = os.environ['AUTOBENCH_{DATASET}_ROOT']
rows = []
for fw in ('clam', 'nnmil'):
    df = pd.read_csv(f'{root}/benchmark/aggregated/{fw}/standard.csv')
    o = df[df.task == 'os']
    for _, r in o.iterrows():
        rows.append((fw, r['model_type'], r.get('survival_loss'), r['test_c_index_mean'], r['test_c_index_std']))
res = pd.DataFrame(rows, columns=['fw','model','loss','test_cidx','std']).sort_values('test_cidx', ascending=False)
print(res.to_string(index=False, float_format=lambda x: f'{x:.3f}'))
"
```

Compare frameworks/losses by averaging `test_c_index_mean` over encoders. On small, low-event cohorts the c-index tends to cluster near 0.5–0.55 with a few configs reaching ~0.6–0.65 and wide per-fold variance (std ~0.15–0.25) — this reflects event scarcity, not a pipeline issue.

## CLI Reference (survival-relevant)

```
uv run python benchmarks/scripts/run_benchmark.py --dataset {dataset} --tasks os ...

  --frameworks {clam,nnmil} [...]   Frameworks to run (survival supported in both)
  --tasks os                        Restrict to the survival task
  --encoders E [...]                Subset encoders (default: all from YAML)
  --gpu / --all_gpus / --gpus       GPU selection
  --no_wandb                        Disable W&B logging
  --prep_only                       Only run data preparation (splits/CSV/plans)
```

> **n_folds is pinned for survival.** `--n_folds` (default 10) only affects classification; survival is forced to 5 by `resolve_n_folds()` regardless of the flag. The survival loss set and `nll_bins` come from the task's YAML entry, not the CLI.

## Timing Estimates

Rough per-experiment estimates (5 folds, single 48 GB GPU; multi-GPU divides wall time):

| Framework / model        | Time per experiment | Notes                         |
| ------------------------ | ------------------- | ----------------------------- |
| CLAM `clam_sb`/`clam_mb` | ~1–4 min            | Tiny model (~0.4 GB); fast    |
| nnMIL `ab_mil`           | ~5–20 min           | Lightweight                   |
| nnMIL `trans_mil`        | ~20–90 min          | Transformer, memory-intensive |

A full OS grid (e.g. ~20 experiments × 5 folds) on 3 GPUs typically finishes in **~2–4 hours**.

## Resuming and Re-running

The pipeline is idempotent. Completed experiment IDs are tracked in `results/_completed.json` and skipped on re-run; per-fold `metrics.json` lets a partially-done experiment resume.

**To force a re-run after changing survival code** (e.g. a new loss recipe), the existing results count as "completed" and are skipped — clear them first:

```bash
uv run --package autobench python -c "
import os, json, pathlib
root = pathlib.Path(os.environ['AUTOBENCH_{DATASET}_ROOT']) / 'benchmark' / 'results'
p = root / '_completed.json'
kept = [e for e in json.loads(p.read_text()) if '__os__' not in e]
p.write_text(json.dumps(sorted(kept), indent=2))
print('cleared os entries from _completed.json')
"
# Then delete results/{framework}/standard/os and re-run.
```

> **Changing the split count or task labels** also requires regenerating splits/plans: delete `splits/standard/os/` and the `nnmil/standard/os_*` plan dirs, then re-run (Phase 1 regenerates them).

## Troubleshooting

### c-index is suspiciously low / exactly 0.0

If `scikit-survival` is **not** installed, the c-index silently falls back to a hand-rolled estimator that scores tied risk scores as discordant (0.0 instead of 0.5), biasing results downward. Verify: `uv run --package autobench python -c "import sksurv.metrics"`. If it errors, run `uv sync` (the dep is declared in `benchmarks/pyproject.toml`).

### Wide per-fold c-index variance (std 0.2–0.3)

Expected on low-event cohorts: with few events over 5 folds, each test fold has only a handful of event patients, so the per-fold c-index is high-variance. This is a power limitation of the data, not a bug. 5-fold (vs 10) already mitigates it.

### Missing feature files for one encoder

```
Skipping N slides without H5 features for {encoder}
```

A slide failed/stalled during feature extraction for that encoder only. nnMIL prep prints the message above and excludes those slides from the plan; the CLAM survival path drops them silently. Either way the run continues on the available slides, so that encoder is then scored on fewer slides than the others. Re-extract the missing slide (see the feature extraction tutorial), then delete and regenerate that encoder's nnMIL plan (`nnmil/standard/os_{encoder}_*`) so it picks the slide back up.

### "Non-OOM experiment failure" aborts the run

Only `torch.cuda.OutOfMemoryError` is retried. Any other worker exception raises `RuntimeError` and aborts the whole multi-GPU run; check the per-experiment log under `{benchmark_dir}/logs/`, fix the cause, and re-run (idempotent).

### AUC / balanced-accuracy columns are NaN for os rows

Expected. Survival has no classification metrics; the c-index lives in the `test_c_index_*` columns of the aggregated CSV.

## Quick Reference

| Step                 | Command                                                                                                                                                                                                                                                                                                      |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Add OS labels        | `uv run python benchmarks/scripts/add_os_to_manifest.py --manifest datasets/{DATASET}/normalized_manifest.csv --clinical datasets/{DATASET}/clinical.tsv`                                                                                                                                                    |
| OS counts for sheet  | `uv run --package autobench python -c "import pandas as pd; p=pd.read_csv('datasets/{DATASET}/normalized_manifest.csv').drop_duplicates('case_id'); print(f'OS (total {len(p)}: event {int((p.OS_event==1).sum())}, non-event {int((p.OS_event==0).sum())}, not reported {int(p.OS_event.isna().sum())})')"` |
| Verify survival task | `uv run python -c "from autobench.config import load_dataset_config as L; print(L('{dataset}').tasks['os'].task_type)"`                                                                                                                                                                                      |
| Single experiment    | `uv run python benchmarks/scripts/run_benchmark.py --dataset {dataset} --tasks os --frameworks clam --encoders uni_v2 --gpu 0 --no_wandb`                                                                                                                                                                    |
| Full nnMIL run       | `uv run python benchmarks/scripts/run_benchmark.py --dataset {dataset} --tasks os --frameworks nnmil --all_gpus --no_wandb`                                                                                                                                                                                  |
| Full CLAM run        | `uv run python benchmarks/scripts/run_benchmark.py --dataset {dataset} --tasks os --frameworks clam --all_gpus --no_wandb`                                                                                                                                                                                   |
| c-index leaderboard  | read `aggregated/{framework}/standard.csv`, `test_c_index_mean` for `task == os`                                                                                                                                                                                                                             |
| Re-run os            | clear `__os__` from `_completed.json` + delete `results/*/standard/os`, then re-run                                                                                                                                                                                                                          |

## Questions?

Ask me directly :)
