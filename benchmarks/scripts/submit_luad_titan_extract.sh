#!/bin/bash
# SLURM: extract TCGA-LUAD TITAN slide features at TITAN's NATIVE recipe
# (CONCH v1.5 tiles @ 20x/512px -> TITAN pool -> 768-d slide embedding + coords).
# FOLD-INDEPENDENT: run this ONCE; both the 5-fold and 10-fold TITAN arms reuse it.
#
# Two TRIDENT passes (run_batch_of_slides.py):
#   Pass 1 (--task all, --patch_encoder conch_v15 @512px): seg (skips existing) ->
#           coords@512 -> conch_v15 tile features.  <-- the heavy pass.
#   Pass 2 (--task feat, --slide_encoder titan): reads the conch_v15 tiles ->
#           TITAN 768-d slide features.
# Then symlinks 20x_224px_0px_overlap/features_titan -> the new 512px
# slide_features_titan so the benchmark TITAN arm (default 224px base) finds them.
#
# Output: ${LUAD_ROOT}/trident_output/20x_512px_0px_overlap/{features_conch_v15, slide_features_titan}
# Resumable (TRIDENT skips done slides). After it completes: submit_luad_titan.sh.
#
# Usage: sbatch benchmarks/scripts/submit_luad_titan_extract.sh

#SBATCH --job-name=luad_titan_extract
#SBATCH --account=rrg-jma
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=128G
#SBATCH --output=/scratch/yinshuol/autoMIL/logs/titanextract_%x_%j.out
#SBATCH --error=/scratch/yinshuol/autoMIL/logs/titanextract_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leo.yin@mail.utoronto.ca

set -uo pipefail
PROJECT_DIR="/home/yinshuol/scratch/autoMIL/autoMIL"
TRIDENT="$PROJECT_DIR/benchmarks/lib/TRIDENT/run_batch_of_slides.py"

echo "================================================"
echo "LUAD TITAN extraction (CONCH v1.5 @20x/512px -> TITAN 768-d)"
echo "Job ${SLURM_JOB_ID:-N/A} | $(hostname) | $(date)"
echo "================================================"

module load cuda/12.2 2>/dev/null || true
cd "$PROJECT_DIR" || { echo "ERROR: project dir not found"; exit 1; }
source .venv/bin/activate
set -a; source benchmarks/.env; set +a
[ -z "${AUTOBENCH_TCGA_LUAD_ROOT:-}" ] && { echo "ERROR: AUTOBENCH_TCGA_LUAD_ROOT unset"; exit 1; }

WSI_DIR="${AUTOBENCH_TCGA_LUAD_ROOT}/wsi"
JOB_DIR="${AUTOBENCH_TCGA_LUAD_ROOT}/trident_output"
echo "WSI dir:  $WSI_DIR"
echo "Job dir:  $JOB_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true

# HuggingFace login — TITAN is gated (conch_v15 already cached).
python -c "import os
from huggingface_hub import login
t=os.environ.get('HF_TOKEN')
login(token=t, add_to_git_credential=False) if t else print('WARN: no HF_TOKEN')" 2>&1 | tail -2

echo ""
echo "===== Pass 1: seg (skip existing) + coords@512 + conch_v15 tiles@512 ====="
python "$TRIDENT" --task all \
    --wsi_dir "$WSI_DIR" --job_dir "$JOB_DIR" \
    --patch_encoder conch_v15 --mag 20 --patch_size 512 \
    --gpu 0 --skip_errors
P1=$?
if [ $P1 -ne 0 ]; then echo "ERROR: Pass 1 (conch_v15 tiles) failed with exit $P1"; exit $P1; fi

echo ""
echo "===== Pass 2: TITAN slide features (reads conch_v15 tiles) ====="
python "$TRIDENT" --task feat --slide_encoder titan \
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
    [ -e "$LINK224" ] || ln -s ../20x_512px_0px_overlap/slide_features_titan "$LINK224"
    echo "symlink: $LINK224 -> $(readlink "$LINK224" 2>/dev/null)"
    echo "Next: sbatch benchmarks/scripts/submit_luad_titan.sh [n_folds]"
else
    echo "WARNING: $SLIDE512 not found — TITAN extraction did not produce slide features."
    exit 1
fi
echo "=== Done: $(date) ==="
