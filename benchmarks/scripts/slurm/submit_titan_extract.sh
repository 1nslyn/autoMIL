#!/bin/bash
# SLURM: extract TITAN slide features for ANY dataset at TITAN's NATIVE recipe
# (CONCH v1.5 tiles @ 20x/512px -> TITAN pool -> 768-d slide embedding + coords).
# FOLD-INDEPENDENT: run ONCE per dataset; both the 5-fold and 10-fold TITAN arms reuse it.
#
# Two TRIDENT passes: (1) --task all --patch_encoder conch_v15 @512px (the heavy
# pass), (2) --task feat --slide_encoder titan. Then symlinks the 224px base's
# features_titan -> the new 512px slide_features_titan so the benchmark arm finds them.
#
# Usage: sbatch benchmarks/scripts/slurm/submit_titan_extract.sh <dataset>
#   e.g. sbatch benchmarks/scripts/slurm/submit_titan_extract.sh tcga_lgg
# Run from the repo root. After it completes: sbatch submit_titan.sh <dataset> [n_folds]

#SBATCH --job-name=titan_extract
#SBATCH --account=rrg-jma
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=128G
#SBATCH --output=logs/titanextract_%x_%j.out
#SBATCH --error=logs/titanextract_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

set -uo pipefail
DATASET="${1:-${DATASET:-}}"
[ -n "$DATASET" ] || { echo "usage: sbatch submit_titan_extract.sh <dataset>"; exit 1; }
PROJECT_DIR="${SLURM_SUBMIT_DIR:-/home/yinshuol/scratch/autoMIL/autoMIL}"
TRIDENT="$PROJECT_DIR/benchmarks/lib/TRIDENT/run_batch_of_slides.py"

echo "================================================"
echo "${DATASET} TITAN extraction (CONCH v1.5 @20x/512px -> TITAN 768-d)"
echo "Job ${SLURM_JOB_ID:-N/A} | $(hostname) | $(date)"
echo "================================================"

module load cuda/12.2 2>/dev/null || true
cd "$PROJECT_DIR" || { echo "ERROR: project dir not found: $PROJECT_DIR"; exit 1; }
mkdir -p logs
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv is required on the compute node"; exit 1; }
UV_RUN=(uv run --frozen --no-sync --package autobench)
set -a; source benchmarks/.env; set +a

# Resolve WSI dir + trident output dir from the dataset config.
WSI_DIR=$("${UV_RUN[@]}" python -c "from autobench.config import load_dataset_config as L; print(L('${DATASET}').wsi_dir)") \
    || { echo "ERROR: cannot load dataset config '${DATASET}' (check name + AUTOBENCH_*_ROOT in benchmarks/.env)"; exit 1; }
JOB_DIR=$("${UV_RUN[@]}" python -c "from autobench.config import load_dataset_config as L; print(L('${DATASET}').output_dir)")
[ -n "$WSI_DIR" ] && [ -n "$JOB_DIR" ] || { echo "ERROR: could not resolve wsi_dir/output_dir for ${DATASET}"; exit 1; }
echo "WSI dir:  $WSI_DIR"
echo "Job dir:  $JOB_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true

# HuggingFace login — TITAN is gated (conch_v15 already cached).
"${UV_RUN[@]}" python -c "import os
from huggingface_hub import login
t=os.environ.get('HF_TOKEN')
login(token=t, add_to_git_credential=False) if t else print('WARN: no HF_TOKEN')" 2>&1 | tail -2

echo ""
echo "===== Pass 1: seg (skip existing) + coords@512 + conch_v15 tiles@512 ====="
"${UV_RUN[@]}" python "$TRIDENT" --task all \
    --wsi_dir "$WSI_DIR" --job_dir "$JOB_DIR" \
    --patch_encoder conch_v15 --mag 20 --patch_size 512 \
    --gpu 0 --skip_errors
P1=$?
if [ $P1 -ne 0 ]; then echo "ERROR: Pass 1 (conch_v15 tiles) failed with exit $P1"; exit $P1; fi

echo ""
echo "===== Pass 2: TITAN slide features (reads conch_v15 tiles) ====="
"${UV_RUN[@]}" python "$TRIDENT" --task feat --slide_encoder titan \
    --wsi_dir "$WSI_DIR" --job_dir "$JOB_DIR" \
    --mag 20 --patch_size 512 \
    --gpu 0 --skip_errors
P2=$?
if [ $P2 -ne 0 ]; then echo "ERROR: Pass 2 (TITAN slide) failed with exit $P2"; exit $P2; fi

echo ""
echo "===== Expose TITAN features to the benchmark (symlink into 224px base) ====="
SLIDE512="$JOB_DIR/20x_512px_0px_overlap/slide_features_titan"
LINK224="$JOB_DIR/20x_224px_0px_overlap/features_titan"
if [ -d "$SLIDE512" ]; then
    n=$(ls "$SLIDE512"/*.h5 2>/dev/null | wc -l | tr -d ' ')
    echo "TITAN slide features: $n .h5 files in $SLIDE512"
    mkdir -p "$JOB_DIR/20x_224px_0px_overlap"
    [ -e "$LINK224" ] || ln -s ../20x_512px_0px_overlap/slide_features_titan "$LINK224"
    echo "symlink: $LINK224 -> $(readlink "$LINK224" 2>/dev/null)"
    echo "Next: sbatch benchmarks/scripts/slurm/submit_titan.sh ${DATASET} [n_folds]"
else
    echo "WARNING: $SLIDE512 not found — TITAN extraction did not produce slide features."
    exit 1
fi
echo "=== Done: $(date) ==="
