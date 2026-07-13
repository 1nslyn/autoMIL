# TCGA Feature Extraction Tutorial

End-to-end guide for extracting pathology features from TCGA whole-slide images using the autobench pipeline.

## Overview

**Goal:** Extract patch-level features from TCGA WSIs using 3 pathology foundation models:
- **Virchow2** (2560-dim), `paige-ai/Virchow2`
- **H-optimus-1** (1536-dim), `bioptimus/H-optimus-1`
- **UNI2-h** (1536-dim), `MahmoodLab/UNI2-h`

**Pipeline:** GDC slide download → GOLDMARK metadata → YAML config → TRIDENT feature extraction

**Output:** Per-slide `.pt` tensors in `{output_dir}/{encoder_key}/pt_files/`

**Tracking sheet:** `datasets/TCGA-CPTAC-Datasets - TCGA-16.csv`, update your row as you complete each step.

**Reference:**
- `benchmarks/datasets/tcga_luad.yaml`, Leo's completed config (checked into repo)
- `benchmarks/datasets/tcga_template.yaml`, template to copy for your dataset

## Prerequisites

### 1. Cluster Access
- Your own Compute Canada account with SSH access
- Basic SLURM familiarity (`sbatch`, `squeue`, `scancel`)
- Sufficient storage quota (estimate ~300 GB per dataset for WSIs + features)

### 2. Clone the Repository and Set Up the Environment

```bash
# Clone the repo to your scratch space
cd ~/scratch
git clone https://github.com/leoyin1127/autoMIL.git
cd autoMIL

# Load required modules
module load cuda/12.2
module load python/3.11   # or whatever version is available (>=3.10)

# Install uv (if not already available)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart shell to get uv in PATH

# Install dependencies (uv manages the virtual environment automatically)
uv sync

# Verify installation
uv run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
uv run python -c "from autobench.config import load_dataset_config; print('autobench OK')"
```

### 3. GDC Data Transfer Tool
Download and install `gdc-client`:
```bash
# Download the tool
wget https://gdc.cancer.gov/files/public/file/gdc-client_v1.6.1_Ubuntu_x64.zip
unzip gdc-client_v1.6.1_Ubuntu_x64.zip
chmod +x gdc-client

# Add to PATH or move to a bin directory
mkdir -p ~/bin
mv gdc-client ~/bin/
export PATH="$HOME/bin:$PATH"  # add to ~/.bashrc to make permanent
```

### 4. HuggingFace Access
Virchow2, Hoptimus-1 and UNI2-h are **gated models**, you need to request access:
1. Create a HuggingFace account at https://huggingface.co/
2. Go to https://huggingface.co/paige-ai/Virchow2 and request access
3. Go to https://huggingface.co/MahmoodLab/UNI2-h and request access
4. Generate a token at https://huggingface.co/settings/tokens
5. Refer to the `benchmarks/.env.example` file, copy to `benchmarks/.env`, and add your `HF_TOKEN=your_hf_token_here`

### 5. Configure the SLURM Script
  
The default SLURM script (`benchmarks/scripts/slurm/submit_feature_extraction.sh`) references Leo's project directory and email. Update it for your account:

```bash
# Update the project directory path
sed -i "s|/home/yinshuol/scratch/autoMIL/autoMIL|$HOME/scratch/autoMIL|" benchmarks/scripts/slurm/submit_feature_extraction.sh

# Update the SLURM account and email
sed -i "s|--account=def-wanglab|--account=YOUR_ACCOUNT|" benchmarks/scripts/slurm/submit_feature_extraction.sh
sed -i "s|--mail-user=leo.yin@mail.utoronto.ca|--mail-user=YOUR_EMAIL|" benchmarks/scripts/slurm/submit_feature_extraction.sh
```

## Step-by-Step Guide

Throughout this guide, replace `{CODE}` with your TCGA cancer type code in **lowercase** (e.g., `luad`, `brca`, `gbm`) and `{DATASET}` with the full name (e.g., `TCGA-LUAD`).

### Step 1: Prepare the Dataset Directory

```bash
cd ~/scratch/autoMIL

# Create the dataset directory
mkdir -p datasets/{DATASET}/wsi
# Example: mkdir -p datasets/TCGA-LUAD/wsi
```

### Step 2: Download Metadata from GOLDMARK

1. Go to https://artificialintelligencepathology.org/
2. Click **Open Download Center**
3. Find your TCGA project (e.g., TCGA-LUAD)
4. Download the **Sanitized Manifest** (normalized manifest CSV)
5. Save to `datasets/{DATASET}/normalized_manifest.csv`

Also download the per-task split files (e.g., `EGFR_all_splits_*.csv`) for reference.

The GOLDMARK manifest contains everything the pipeline needs:
- `slide_name`, actual GDC filenames (e.g., `TCGA-05-4244-01Z-00-DX1.d4ff32cd-...svs`)
- `sample_names`, patient/case ID (e.g., `TCGA-05-4244`)
- `{GENE}_binary` columns, 0/1 labels for each biomarker (e.g., `EGFR_binary`, `KRAS_binary`)
- `split_1` through `split_5`, pre-defined 5-fold cross-validation splits

**No transformation needed**, the YAML config maps these columns directly to what the pipeline expects.

### Step 3: Download the GDC Manifest and Filter for Slides

#### 3a. Download the manifest from GDC

1. Go to https://portal.gdc.cancer.gov/
2. Click the **Projects** tab
3. Search for your TCGA project (e.g., `TCGA-LUAD`)
4. Click into the project page
5. Click the **Manifest** button at the top

![GDC Project Page, Manifest download](public/Screenshot%202026-03-31%20at%2001.56.01.png)

Save the manifest file to `datasets/{DATASET}/gdc_manifest.txt`.

#### 3b. Filter the manifest to only include slides used by GOLDMARK

The full GDC manifest includes ALL files (genomic, clinical, imaging, e.g., 36,224 files for TCGA-LUAD). We only need the ~465 slides that GOLDMARK actually uses. Run this to create a matched manifest:

```bash
cd ~/scratch/autoMIL

python3 -c "
import csv, sys

dataset_dir = 'datasets/{DATASET}'

# Load GOLDMARK slide names
goldmark_slides = set()
with open(f'{dataset_dir}/normalized_manifest.csv') as f:
    for row in csv.DictReader(f):
        goldmark_slides.add(row['slide_name'])

# Filter GDC manifest to match
matched = 0
with open(f'{dataset_dir}/gdc_manifest.txt') as f_in, \
     open(f'{dataset_dir}/gdc_manifest_matched.txt', 'w') as f_out:
    header = f_in.readline()
    f_out.write(header)
    for line in f_in:
        filename = line.strip().split('\t')[1]
        if filename in goldmark_slides:
            f_out.write(line)
            matched += 1

print(f'GOLDMARK slides: {len(goldmark_slides)}')
print(f'Matched in GDC manifest: {matched}')
print(f'Written to: {dataset_dir}/gdc_manifest_matched.txt')
"
```

#### 3c. Download the slides via SLURM

Submit the download as a SLURM job (runs on CPU, no GPU needed):

```bash
mkdir -p logs
sbatch benchmarks/scripts/slurm/submit_gdc_download.sh \
    datasets/{DATASET} \
    datasets/{DATASET}/gdc_manifest_matched.txt
```

This script downloads all matched slides, flattens them from GDC's nested UUID directories into `datasets/{DATASET}/wsi/`, and cleans up. Monitor with:

```bash
tail -f logs/gdc_download_*.out
```

After the job completes, verify:

```bash
echo "Downloaded: $(ls datasets/{DATASET}/wsi/*.svs | wc -l) slides"
```

### Step 4: Update the Tracking Sheet

Fill in your row in `https://docs.google.com/spreadsheets/d/1DVzgG7EfkQwOw-hjWqI8gwagAzdG9jG-fR8z7-IDbEk/edit?usp=sharing`:
- Fill in the **DOI**, **Radiology** columns, **Pathology** columns, **License**, and other dataset metadata from GDC/TCIA.
- In the **Tasks** cell, record the class distribution for each biomarker task. Format: `task_name (total: positive vs negative)`. For example, CPTAC-CCRCC's BAP1 task is recorded as `BAP1_mutation (103: 20 vs 83)`, meaning 103 cases with labels, 20 positive (mutant), 83 negative (wildtype). You can compute this from the manifest's binary columns. List each task on a new line.

### Step 5: Create Dataset YAML Config

```bash
# Copy the template
cp benchmarks/datasets/tcga_template.yaml benchmarks/datasets/tcga_{code}.yaml
# Example: cp benchmarks/datasets/tcga_template.yaml benchmarks/datasets/tcga_luad.yaml
```

Edit the YAML file. The GOLDMARK normalized manifest has standardized column names across all TCGA projects, so the column mappings below work for every dataset. Here is the TCGA-LUAD example:

```yaml
name: tcga_luad
description: "TCGA-LUAD, Lung Adenocarcinoma (EGFR, KRAS mutation prediction)"

paths:
  data_root: "${AUTOBENCH_TCGA_LUAD_ROOT}"
  wsi_dir: "${data_root}/wsi"
  mapping_csv: "${data_root}/normalized_manifest.csv"   # <-- GOLDMARK manifest
  output_dir: "${data_root}/trident_output"
  benchmark_dir: "${data_root}/benchmark"
  features_base_dir: "${output_dir}/20x_224px_0px_overlap"

tasks:
  egfr:
    label_col: "EGFR_binary"   # <-- GOLDMARK uses {GENE}_binary columns
    label_map:
      0: "wildtype"
      1: "mutant"
    n_classes: 2
  kras:
    label_col: "KRAS_binary"
    label_map:
      0: "wildtype"
      1: "mutant"
    n_classes: 2

split_strategies:
  standard:
    train_cohorts: []
    test_cohorts: []

task_strategy_feasibility:
  egfr: ["standard"]
  kras: ["standard"]

# These column names are consistent across all GOLDMARK TCGA manifests:
#   slide_name  = GDC filename (matches .svs files on disk after download)
#   sample_names = TCGA case/patient ID
slide_id_column: "slide_name"
slide_id_transform: null
wsi_extension: null              # slide_name already includes .svs
case_id_column: "sample_names"
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

**What to customize per dataset:**
- `name` and `description`
- `paths.data_root` env var name (e.g., `AUTOBENCH_TCGA_BRCA_ROOT`)
- `tasks`, one entry per biomarker. Check your GOLDMARK manifest for the `{GENE}_binary` column names
- `task_strategy_feasibility`, list all your task names

### Step 6: Configure Environment

Add your dataset's root path to `benchmarks/.env`:

```bash
# Example for TCGA-LUAD (use absolute path)
echo 'AUTOBENCH_TCGA_LUAD_ROOT=/home/$USER/scratch/autoMIL/datasets/TCGA-LUAD' >> benchmarks/.env
```

Verify the config loads:

```bash
cd ~/scratch/autoMIL
set -a && source benchmarks/.env && set +a

uv run python -c "
from autobench.config import load_dataset_config
ds = load_dataset_config('tcga_{code}')
print(f'Name: {ds.name}')
print(f'WSI dir: {ds.wsi_dir}')
print(f'Mapping CSV: {ds.mapping_csv}')
print(f'Encoders: {list(ds.encoder_models.values())}')
"
```

### Step 7: Run Feature Extraction via SLURM

The Fir cluster has H100 MIG GPU slices, a **3g.40gb** (40 GB) slice is sufficient for feature extraction and more resource-efficient than a full H100.

```bash
mkdir -p logs

# Submit using MIG GPU slice (recommended)
sbatch benchmarks/scripts/slurm/submit_feature_extraction_mig.sh tcga_{code} virchow2 hoptimus1 uni_v2

# Or submit using full H100 (if MIG queue is busy)
sbatch benchmarks/scripts/slurm/submit_feature_extraction.sh tcga_{code}
```

**Monitor the job:**
```bash
squeue -u $USER
tail -f logs/extract_wsi_extract_*.out
```

For large datasets (>800 slides, e.g., TCGA-BRCA), increase the time limit:
```bash
sbatch --time=2-00:00:00 benchmarks/scripts/slurm/submit_feature_extraction_mig.sh tcga_brca virchow2 hoptimus1 uni_v2
```

### Step 8: Verify and Report

```bash
DATASET_ROOT=datasets/{DATASET}

# Count .h5 feature files per model
for model in virchow2 hoptimus1 uni_v2; do
    echo "$model: $(ls $DATASET_ROOT/trident_output/20x_224px_0px_overlap/features_$model/*.h5 2>/dev/null | wc -l)"
done

# Compare to input slide count
echo "Input slides: $(tail -n +2 $DATASET_ROOT/trident_output/slide_list.csv | wc -l)"

# Check for skipped/corrupted slides
cat $DATASET_ROOT/trident_output/skipped_slides.txt 2>/dev/null

# Check logs for errors
grep -i "error\|failed\|oom" logs/extract_wsi_extract_*.out
```

The `.pt` file count should match (or be close to) the input slide count.

#### Update Tracking Sheet

Update your row in `datasets/TCGA-CPTAC-Datasets - TCGA-16.csv`:
- Mark **FE:virchow2**, **FE:hoptimus1**, **FE:univ2** as complete (or note failures)
- Ensure all dataset info columns are filled (DOI, subjects, license, etc.)
- Ensure **Tasks** cell has class distributions for each biomarker, e.g.: `EGFR (465: 51 vs 414)`
- Add any issues to **Notes**

## Expected Output Structure

After successful extraction, your dataset directory should look like:

```
datasets/{DATASET}/
├── gdc_manifest.txt                          # Full GDC manifest (from portal)
├── gdc_manifest_matched.txt                  # Filtered to GOLDMARK slides only
├── normalized_manifest.csv                   # From GOLDMARK Download Center
├── wsi/                                      # Downloaded .svs files (flattened)
│   ├── TCGA-XX-XXXX-01Z-00-DX1.{uuid}.svs
│   └── ...
└── trident_output/
    ├── slide_list.csv                        # Generated by extraction script
    ├── _logs_segmentation.txt                # Segmentation log
    ├── contours/                             # Tissue contour visualizations (.jpg)
    ├── contours_geojson/                     # GeoJSON contours
    ├── thumbnails/                           # Slide thumbnails (.jpg)
    └── 20x_224px_0px_overlap/
        ├── _config_coords.json
        ├── _logs_coords.txt
        ├── patches/                          # Patch coordinate files
        │   ├── TCGA-XX-XXXX-..._patches.h5
        │   └── ...
        ├── visualization/                    # Patch overlay visualizations (.jpg)
        ├── features_virchow2/                # Virchow2 features (2560-dim)
        │   ├── TCGA-XX-XXXX-....h5
        │   └── ...
        ├── features_hoptimus1/               # H-optimus-1 features (1536-dim)
        │   └── ...
        └── features_uni_v2/                  # UNI2-h features (1536-dim)
            └── ...
```

Each `.h5` feature file contains two datasets:
- `features`: `(num_patches, embedding_dim)` float32, the patch embeddings
- `coords`: `(num_patches, 2)` int64, the (x, y) patch coordinates

The benchmark pipeline automatically converts these to `.pt` tensors during experiment preparation.

## Troubleshooting

### HuggingFace 403 / Gated Model Error
```
ValueError: Gated repo. You must be authenticated to access it.
```
**Fix:** Ensure `HF_TOKEN` is set in `benchmarks/.env` and you've been granted access to the model on HuggingFace.

### CUDA Out of Memory (OOM)
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```
**Fix:** Reduce batch size. Virchow2 (2560-dim) is the largest model.
```bash
uv run python benchmarks/scripts/run_feature_extraction.py \
    --dataset tcga_xxx \
    --models virchow2 \
    --batch_size 32 \
    --gpu 0
```

### Corrupted Slides
Some GDC slides may be corrupted. The pipeline validates slides upfront and lists failures in `skipped_slides.txt`. If many slides fail:
1. Re-download the specific UUIDs from GDC
2. Try opening with OpenSlide locally to confirm corruption

### Disk Space
Estimate storage needs per dataset:

| Component | Size per dataset |
|-----------|-----------------|
| WSI slides (.svs) | 50-300 GB (varies by cohort) |
| Patch coordinates | ~1 GB |
| virchow2 features | ~135 GB |
| hoptimus1 features | ~81 GB |
| uni_v2 features | ~81 GB |
| **Total** | **~350-600 GB** |

Check quota before starting: `quota` or `df -h /scratch/`

### Config Loading Errors
```
FileNotFoundError: Dataset config not found
```
**Fix:**
- Ensure YAML file is in `benchmarks/datasets/` with the correct name
- Ensure the env var is set in `benchmarks/.env`
- Ensure `.env` is sourced (the extraction script does this automatically)

### Job Timeout on SLURM
For large datasets (>800 cases), 24 hours may not be enough:
```bash
sbatch --time=2-00:00:00 benchmarks/scripts/slurm/submit_feature_extraction.sh tcga_xxx
```

If a job times out mid-extraction, you can resume by running with `--skip_seg` (if segmentation/patching already completed) and specifying only the incomplete models:
```bash
uv run python benchmarks/scripts/run_feature_extraction.py \
    --dataset tcga_xxx \
    --models uni_v2 \
    --skip_seg \
    --gpu 0
```

## Quick Reference

| Step | Command |
|------|---------|
| Download slides | `sbatch benchmarks/scripts/slurm/submit_gdc_download.sh datasets/{DATASET} datasets/{DATASET}/gdc_manifest_matched.txt` |
| Extract features (MIG) | `sbatch benchmarks/scripts/slurm/submit_feature_extraction_mig.sh tcga_{code} virchow2 hoptimus1 uni_v2` |
| Extract features (full H100) | `sbatch benchmarks/scripts/slurm/submit_feature_extraction.sh tcga_{code}` |
| Check jobs | `squeue -u $USER` |
| Monitor extraction | `tail -f logs/extract_wsi_extract_*.out` |
| Verify config | `uv run python -c "from autobench.config import load_dataset_config; print(load_dataset_config('tcga_{code}').wsi_dir)"` |
| Count results | `ls datasets/{DATASET}/trident_output/20x_224px_0px_overlap/features_virchow2/*.h5 \| wc -l` |

## Questions?

Ask me directly :)

---

## Handling GDC UUID Rewrites (Stale GOLDMARK Manifests)

> **TL;DR, when Step 3b returns 0 (or very few) matches, GDC has re-uploaded
> the slides with new file UUIDs. The patient/slide mapping is still correct;
> only the file UUIDs are stale. Fix it with
> `benchmarks/scripts/manifests/refresh_goldmark_uuids.py`.**

### Symptom

After running Step 3b, you see something like:

```
GOLDMARK slides: 149
Matched in GDC manifest: 0
```

Or your subsequent download produces 0 `.svs` files in `wsi/`.

### Why this happens

The matching code shown earlier in this tutorial filters by **exact filename
equality**, including the file UUID embedded in the GDC filename:

```
TCGA-2G-AAEW-01Z-00-DX1.<UUID>.svs
                        ^^^^^^^^
                        this part can change
```

GDC sometimes re-uploads slides, same patient, same physical specimen, new
file UUID. When that happens, GOLDMARK's `slide_name` column points at file
UUIDs that no longer exist in GDC, and the exact-match filter returns 0.

Confirmed in the wild for **TCGA-TGCT** (100% UUID rewrite rate at time of
writing). Any future cohort can hit the same issue silently.

### The fix

Match on the **TCGA barcode prefix** (the part before the first `.`),
which is the stable per-slide identifier. Verified unique per slide on
both the GOLDMARK and GDC sides for every cohort tested.

A helper script handles this end-to-end:

```bash
cd ~/scratch/autoMIL
python benchmarks/scripts/manifests/refresh_goldmark_uuids.py datasets/{DATASET}
```

It expects `datasets/{DATASET}/` to contain:
- `normalized_manifest.csv`, from GOLDMARK
- `gdc_manifest.txt`, from the GDC portal

It produces:
- `gdc_manifest_matched.txt`, same purpose as before, but with **current**
  GDC UUIDs (use this for the SLURM download in Step 3c)
- `normalized_manifest.refreshed.csv`, GOLDMARK manifest with `slide_name`
  rewritten to match GDC's current UUIDs (use this as `mapping_csv` in your
  YAML)
- `uuid_rewrites.tsv`, per-row audit log: `unchanged`, `rewritten`,
  `not_found_in_gdc`, or `ambiguous`

The script aborts if either input has a duplicate barcode (it never guesses)
and exits non-zero if any row is unmatched or ambiguous so you cannot
silently proceed with a partial dataset.

### Things to change in the workflow

When you suspect or confirm UUID rewrites, swap the following:

1. **Replace the inline Python in Step 3b** with a single command:
   ```bash
   python benchmarks/scripts/manifests/refresh_goldmark_uuids.py datasets/{DATASET}
   ```
   (You no longer need to write the matching logic by hand.)

2. **Step 3c (download) is unchanged**, it still reads
   `gdc_manifest_matched.txt`, which now contains current UUIDs.

3. **Step 5 (YAML config)**, point `mapping_csv` at the refreshed file:
   ```yaml
   paths:
     mapping_csv: "${data_root}/normalized_manifest.refreshed.csv"
     #                                          ^^^^^^^^^^^
     # was: normalized_manifest.csv (only valid when no UUIDs were rewritten)
   ```
   This is required whenever the audit shows `rewritten` rows; otherwise
   `validate_slides()` will look for files that no longer exist on disk.
   For cohorts with zero rewrites, both files are semantically identical
   so pointing at the refreshed one is always safe.

### Sanity check

After running the helper script, verify the audit:

```bash
cd datasets/{DATASET}
echo "rewritten: $(awk -F'\t' 'NR>1 && $3==\"rewritten\"' uuid_rewrites.tsv | wc -l)"
echo "unchanged: $(awk -F'\t' 'NR>1 && $3==\"unchanged\"' uuid_rewrites.tsv | wc -l)"
echo "missing:   $(awk -F'\t' 'NR>1 && $3==\"not_found_in_gdc\"' uuid_rewrites.tsv | wc -l)"
echo "ambiguous: $(awk -F'\t' 'NR>1 && $3==\"ambiguous\"' uuid_rewrites.tsv | wc -l)"
```

Healthy outcome: `missing == 0` and `ambiguous == 0`. Any non-zero value in
those rows means the refreshed manifest is incomplete, review
`uuid_rewrites.tsv` before proceeding.
