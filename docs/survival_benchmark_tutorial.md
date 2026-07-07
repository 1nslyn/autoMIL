# Survival Benchmark Tutorial

End-to-end guide for adding overall survival (OS) labels to a manifest and running survival benchmark experiments.

This tutorial covers two phases:
1. **Label preparation** — join OS columns from a GDC clinical export into `normalized_manifest.csv`
2. **Running experiments** — *(coming soon)*

Label preparation is a one-time step per cohort, done after the manifest has been created by [`prepare_cptac_manifest.py`](cptac_feature_extraction_tutorial.md).

**Pipeline:** GDC `clinical.tsv` → `add_os_to_manifest.py` → manifest with `OS_event` / `OS_time`

**Script:** `benchmarks/scripts/add_os_to_manifest.py`

## Prerequisites

- `normalized_manifest.csv` already exists (produced by `prepare_cptac_manifest.py`)
- `clinical.tsv` downloaded from the GDC Data Portal (Step 1 below)

## Step 1: Download clinical.tsv from GDC

> Skip this if you already have `clinical.tsv`.

1. Go to the [GDC Data Portal](https://portal.gdc.cancer.gov/repository)
2. Under **Files**, filter by **Program: CPTAC** and **Data Type: Clinical Supplement**
3. Select all files for your project (e.g., `CPTAC-3`) and click **Download → TSV**

Place the file in your dataset directory:

```bash
mv ~/Downloads/clinical.tsv datasets/{DATASET}/clinical.tsv
# Example: mv ~/Downloads/clinical.tsv datasets/CPTAC-CCRCC/clinical.tsv
```

**Columns used for survival:**

| GDC column | Meaning |
|---|---|
| `cases.index_date` | Reference event for all day offsets — typically `Diagnosis` |
| `demographic.vital_status` | `Alive` or `Dead` |
| `demographic.days_to_death` | Days from index date to death (Dead only) |
| `diagnoses.days_to_last_follow_up` | Days from index date to last contact (Alive) |

## Step 2: Join OS columns to the manifest

```bash
cd ~/scratch/autoMIL

uv run python benchmarks/scripts/add_os_to_manifest.py \
    --manifest datasets/{DATASET}/normalized_manifest.csv \
    --clinical datasets/{DATASET}/clinical.tsv

# Example for CCRCC:
uv run python benchmarks/scripts/add_os_to_manifest.py \
    --manifest datasets/CPTAC-CCRCC/normalized_manifest.csv \
    --clinical datasets/CPTAC-CCRCC/clinical.tsv
```

This overwrites the manifest in place. Use `--output <path>` to write a separate file.

Expected output for CPTAC-CCRCC:

```
Slides: 110
  OS_event=1 (Dead):         21
  OS_event=0 (Alive):        89
  OS_event=NaN (Not Rep.):   0  <- dropped by survival task dropna
Saved -> datasets/CPTAC-CCRCC/normalized_manifest.csv
```

## Step 3: Verify

```bash
uv run python -c "
import pandas as pd
df = pd.read_csv('datasets/{DATASET}/normalized_manifest.csv')
print(df[['case_id', 'OS_event', 'OS_time']].head())
print()
print('Event dist: ', df['OS_event'].value_counts().to_dict())
print(f'Time range:  {df.OS_time.min():.0f} – {df.OS_time.max():.0f} days')
print(f'NaN rows:    {df.OS_event.isna().sum()}')
"
```

## Step 4: Enable survival task in the dataset YAML

In `benchmarks/datasets/cptac_{code}.yaml`, add an `os` entry under `tasks`:

```yaml
tasks:
  # ... existing classification tasks ...
  os:
    task_type: survival
    event_col: OS_event
    time_col: OS_time
    survival_losses: [cox, nllsurv]
    nll_bins: 4
```

Add `os` to `task_strategy_feasibility`:

```yaml
task_strategy_feasibility:
  bap1: ["standard"]
  # ... other tasks ...
  os: ["standard"]
```

## Data quality note (CPTAC-CCRCC)

Six patients in CPTAC-CCRCC have `days_to_last_follow_up ≤ 0` in the GDC export — a data entry artifact where the recorded follow-up date falls on or before the diagnosis date:

| Patient | OS_time |
|---|---|
| C3N-00148 | −9 |
| C3N-00149 | −8 |
| C3N-00150 | −7 |
| C3N-00154 | −7 |
| C3N-00646 | −6 |
| C3L-00792 | 0 |

All six are `OS_event=0` (censored / Alive). The script passes them through as-is.

**Impact:**
- **Cox regression** — unaffected; uses relative ordering only.
- **NLL survival** — the `nll_bins` quantile edges are computed from event-only times (`OS_event=1`), so these censored rows do not shift the bin boundaries.
