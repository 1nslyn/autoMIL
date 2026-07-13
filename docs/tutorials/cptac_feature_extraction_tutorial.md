# CPTAC Feature Extraction Tutorial

End-to-end guide for extracting pathology features from CPTAC whole-slide images using the autobench pipeline with **Patho-Bench** labels.

## Overview

**Goal:** Extract patch-level features from CPTAC WSIs using 3 pathology foundation models:
- **Virchow2** (2560-dim), `paige-ai/Virchow2`
- **H-optimus-1** (1536-dim), `bioptimus/H-optimus-1`
- **UNI2-h** (1536-dim), `MahmoodLab/UNI2-h`

**Pipeline:** TCIA slide download → Patho-Bench splits → YAML config → TRIDENT feature extraction

**Key difference from TCGA:** Labels and splits come from **Patho-Bench** (HuggingFace), not GOLDMARK. A helper script (`benchmarks/scripts/manifests/prepare_cptac_manifest.py`) downloads and converts them. Everything else — TRIDENT extraction, YAML config, SLURM scripts — is identical to the TCGA workflow.

**Output:** Per-slide `.h5` tensors in `{output_dir}/20x_224px_0px_overlap/features_{encoder}/`

**Tracking sheet:** `datasets/TCGA-CPTAC-Datasets - CPTAC-10.csv`, update your row as you complete each step.

**Reference:**
- `benchmarks/datasets/templates/cptac_template.yaml`, template config to copy
- `benchmarks/datasets/tcga_luad.yaml`, Leo's completed TCGA reference (structure is identical)

## Available CPTAC Cohorts

| Source name (Patho-Bench) | Cancer type |
|---------------------------|-------------|
| `cptac_brca` | Breast Cancer |
| `cptac_ccrcc` | Clear Cell Renal Cell Carcinoma |
| `cptac_coad` | Colorectal Adenocarcinoma |
| `cptac_gbm` | Glioblastoma |
| `cptac_hnsc` | Head & Neck Squamous Cell Carcinoma |
| `cptac_lscc` | Laryngeal Squamous Cell Carcinoma |
| `cptac_luad` | Lung Adenocarcinoma |
| `cptac_pdac` | Pancreatic Ductal Adenocarcinoma |
| `cptac_ucec` | Uterine Corpus Endometrial Cancer |
| `cptac_ov` | Ovarian Cancer |

## Prerequisites

### 1. Cluster Access
Same as TCGA — your own Compute Canada account, SLURM familiarity, sufficient storage quota (estimate ~200–400 GB per cohort for WSIs + features).

### 2. Clone the Repository and Set Up the Environment

```bash
cd ~/scratch
git clone https://github.com/leoyin1127/autoMIL.git
cd autoMIL

module load cuda/12.2
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

uv sync --all-packages
uv run python -c "from autobench.config import load_dataset_config; print('autobench OK')"
```

### 3. HuggingFace Access

Virchow2, H-optimus-1, and UNI2-h are **gated models**:
1. Create a HuggingFace account at https://huggingface.co/
2. Request access to `paige-ai/Virchow2`, `bioptimus/H-optimus-1`, `MahmoodLab/UNI2-h`
3. Generate a token at https://huggingface.co/settings/tokens
4. Add to `benchmarks/.env`: `HF_TOKEN=your_hf_token_here`

## Step-by-Step Guide

Throughout this guide, replace `{CODE}` with your CPTAC cohort code in **lowercase**
(e.g., `ccrcc`, `brca`) and `{DATASET}` with the directory name (e.g., `CPTAC-CCRCC`).

### Step 1: Prepare the Dataset Directory

```bash
cd ~/scratch/autoMIL

mkdir -p datasets/{DATASET}/wsi
# Example: mkdir -p datasets/CPTAC-CCRCC/wsi
```

### Step 2: Find Available Tasks

Use `--list-tasks` to see what tasks are available for your cohort:

```bash
uv run python benchmarks/scripts/manifests/prepare_cptac_manifest.py \
    --source cptac_{code} \
    --list-tasks

# Example for CCRCC:
uv run python benchmarks/scripts/manifests/prepare_cptac_manifest.py \
    --source cptac_ccrcc \
    --list-tasks
```

### Step 3: Download Patho-Bench Splits (replaces GOLDMARK)

Pass the task names found in the previous step to `--tasks` to download the splits and convert them to an autobench-compatible `normalized_manifest.csv`:

```bash
uv run python benchmarks/scripts/manifests/prepare_cptac_manifest.py \
    --source cptac_{code} \
    --tasks TASK1_mutation TASK2_mutation \
    --saveto datasets/{DATASET}

# Example for CCRCC:
uv run python benchmarks/scripts/manifests/prepare_cptac_manifest.py \
    --source cptac_ccrcc \
    --tasks BAP1_mutation VHL_mutation Immune_class PBRM1_mutation \
    --saveto datasets/CPTAC-CCRCC
```

This produces:
- `datasets/{DATASET}/normalized_manifest.csv` — the mapping file autobench reads
- `datasets/{DATASET}/pb_splits/` — raw Patho-Bench TSVs (kept for reference)

Inspect the output:

```bash
head -3 datasets/{DATASET}/normalized_manifest.csv
# case_id,slide_id,BAP1_binary,VHL_binary
# C3L-00001,C3L-00001-21,1,0
# C3L-00002,C3L-00002-21,0,1
```

The `slide_id` column is a **filename stem with no extension**. Verify it matches your WSI filenames:

```bash
ls datasets/CPTAC-GBM/wsi/ | head -3
# C3L-00001-21.svs
# C3L-00002-21.svs
```

If there is a mismatch (e.g., extra prefix, different naming convention), resolve it before proceeding — autobench uses `slide_id` to look up both WSI files and TRIDENT `.h5` features.

### Step 4: Get WSI Files from TCIA

CPTAC slides are distributed via TCIA (The Cancer Imaging Archive), not GDC. If you have already downloaded the WSIs, place them in `datasets/{DATASET}/wsi/` as a flat directory of `.svs` files.

If you still need to download:

1. Go to https://www.cancerimagingarchive.net/
2. Search for your CPTAC project (e.g., `CPTAC-CCRCC`)
3. Download using the NBIA Data Retriever or the `tcia_utils` Python package

The resulting slides should be placed flat in `datasets/{DATASET}/wsi/`:
```
datasets/{DATASET}/wsi/
├── C3L-00001-21.svs
├── C3L-00002-21.svs
└── ...
```

Verify slide count:
```bash
echo "WSI count: $(ls datasets/{DATASET}/wsi/*.svs | wc -l)"
```

### Step 5: Update the Tracking Sheet

Fill in your row in the [TCGA-CPTAC tracking sheet](https://docs.google.com/spreadsheets/d/1DVzgG7EfkQwOw-hjWqI8gwagAzdG9jG-fR8z7-IDbEk/edit?gid=994979686#gid=994979686):
- Fill in **DOI**, **Radiology** and **Pathology** columns, **License**, other dataset metadata
- In the **Tasks** cell, list each task with its class distribution. Format: `task_name (total: N vs N vs ...)`, one per line. Example:

```
BAP1_mutation (103: 83 vs 20)
VHL_mutation (103: 51 vs 52)
IMMUNE_CLASS (243: 71 vs 90 vs 82)
```

Compute from the manifest:

```bash
uv run python -c "
import pandas as pd
df = pd.read_csv('datasets/{DATASET}/normalized_manifest.csv')
for col in df.columns:
    if col.endswith('_binary'):
        task = col.replace('_binary', '')
        total = int(df[col].notna().sum())
        counts = df[col].value_counts(dropna=True).sort_index()
        breakdown = ' vs '.join(f'{c}' for v, c in counts.items())
        print(f'{task} ({total}: {breakdown})')
"
```

### Step 6: Create Dataset YAML Config

```bash
cp benchmarks/datasets/templates/cptac_template.yaml benchmarks/datasets/cptac/cptac_{code}.yaml
# Example: cp benchmarks/datasets/templates/cptac_template.yaml benchmarks/datasets/cptac/cptac_ccrcc.yaml
```

Edit the YAML file. Here is a completed example for `cptac_ccrcc`:

```yaml
name: cptac_ccrcc
description: "CPTAC-CCRCC — Clear Cell Renal Cell Carcinoma (BAP1, VHL mutation)"

paths:
  data_root: "${AUTOBENCH_CPTAC_CCRCC_ROOT}"
  wsi_dir: "${data_root}/wsi"
  mapping_csv: "${data_root}/normalized_manifest.csv"   # from prepare_cptac_manifest.py
  output_dir: "${data_root}/trident_output"
  benchmark_dir: "${data_root}/benchmark"
  features_base_dir: "${output_dir}/20x_224px_0px_overlap"

tasks:
  bap1:
    label_col: "BAP1_binary"
    label_map:
      0: "wildtype"
      1: "mutant"
    n_classes: 2
  vhl:
    label_col: "VHL_binary"
    label_map:
      0: "wildtype"
      1: "mutant"
    n_classes: 2

split_strategies:
  standard:
    train_cohorts: []
    test_cohorts: []

task_strategy_feasibility:
  bap1: ["standard"]
  vhl: ["standard"]

# Patho-Bench slide_id is already a stem (no extension) — no transform needed.
slide_id_column: "slide_id"
slide_id_transform: null
wsi_extension: ".svs"
case_id_column: "case_id"
status_column: null
status_value: null

encoders:
  models:
    "paige-ai/Virchow2": "virchow2"
    "bioptimus/H-optimus-1": "hoptimus1"
    "MahmoodLab/UNI2-h": "uni_v2"
  dims:
    virchow2: 2560
    hoptimus1: 1536
    uni_v2: 1536

nnmil_models:
  - ab_mil
  - trans_mil

extraction:
  magnification: 20
  patch_size: 224
  batch_size: 64
```

**What to customize per cohort:**
- `name` and `description`
- `paths.data_root` env var name (e.g., `AUTOBENCH_CPTAC_BRCA_ROOT`)
- `tasks`: one entry per task; `label_col` must match `{GENE}_binary` from the manifest
- `task_strategy_feasibility`: list all task names

**Notable difference from TCGA:**
- `slide_id_transform: null` (Patho-Bench gives stems directly; TCGA used `strip_svs`)
- `wsi_extension: ".svs"` (appended for WSI file lookups; TCGA set this to `null`)

### Step 7: Configure Environment

```bash
# Add to benchmarks/.env (use an absolute path)
echo 'AUTOBENCH_CPTAC_CCRCC_ROOT=/home/$USER/scratch/autoMIL/datasets/CPTAC-CCRCC' >> benchmarks/.env
```

Verify the config loads:

```bash
cd ~/scratch/autoMIL
set -a && source benchmarks/.env && set +a

uv run python -c "
from autobench.config import load_dataset_config
ds = load_dataset_config('cptac_{code}')
print(f'Name: {ds.name}')
print(f'WSI dir: {ds.wsi_dir}')
print(f'Mapping CSV: {ds.mapping_csv}')
print(f'Encoders: {list(ds.encoder_models.values())}')
"
```

### Step 8: Run Feature Extraction via SLURM

The Fir cluster has H100 MIG GPU slices; a `3g.40gb` slice is sufficient.

```bash
mkdir -p logs

# MIG slice (recommended)
sbatch benchmarks/scripts/slurm/submit_feature_extraction_mig.sh cptac_{code} virchow2 hoptimus1 uni_v2

# Full H100 (if MIG queue is busy)
sbatch benchmarks/scripts/slurm/submit_feature_extraction.sh cptac_{code}
```

Monitor:
```bash
squeue -u $USER
tail -f logs/extract_wsi_extract_*.out
```

For larger cohorts (BRCA, LUAD — >200 slides), increase wall time:
```bash
sbatch --time=2-00:00:00 benchmarks/scripts/slurm/submit_feature_extraction_mig.sh cptac_{code} virchow2 hoptimus1 uni_v2
```

### Step 9: Verify and Report

```bash
DATASET_ROOT=datasets/{DATASET}

for model in virchow2 hoptimus1 uni_v2; do
    echo "$model: $(ls $DATASET_ROOT/trident_output/20x_224px_0px_overlap/features_$model/*.h5 2>/dev/null | wc -l)"
done

echo "Input slides: $(tail -n +2 $DATASET_ROOT/trident_output/slide_list.csv | wc -l)"

cat $DATASET_ROOT/trident_output/skipped_slides.txt 2>/dev/null

grep -i "error\|failed\|oom" logs/extract_wsi_extract_*.out
```

Update the tracking sheet: mark **FE:virchow2**, **FE:hoptimus1**, **FE:univ2** as complete.

## Expected Output Structure

```
datasets/{DATASET}/
├── normalized_manifest.csv               # From prepare_cptac_manifest.py
├── pb_splits/                            # Raw Patho-Bench TSVs (reference)
│   └── cptac_{code}/
│       └── {task}/
│           ├── k=all.tsv
│           └── config.yaml
├── wsi/                                  # TCIA .svs files (flat)
│   ├── C3L-00001-21.svs
│   └── ...
└── trident_output/
    ├── slide_list.csv
    ├── thumbnails/
    ├── contours_geojson/
    └── 20x_224px_0px_overlap/
        ├── patches/
        ├── features_virchow2/            # 2560-dim, one .h5 per slide
        ├── features_hoptimus1/           # 1536-dim
        └── features_uni_v2/             # 1536-dim
```

Each `.h5` contains:
- `features`: `(num_patches, embedding_dim)` float32
- `coords`: `(num_patches, 2)` int64
