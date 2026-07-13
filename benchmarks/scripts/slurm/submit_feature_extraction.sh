#!/bin/bash
# SLURM job script: WSI feature extraction on Fir HPC
#
# Extracts patch-level features from whole-slide images using pathology
# foundation models via TRIDENT. Runs all encoders sequentially on 1 GPU.
#
# Usage:
#   sbatch benchmarks/scripts/slurm/submit_feature_extraction.sh <dataset> [extra args...]
#   sbatch benchmarks/scripts/slurm/submit_feature_extraction.sh clwd
#   sbatch benchmarks/scripts/slurm/submit_feature_extraction.sh ccrcc
#   sbatch benchmarks/scripts/slurm/submit_feature_extraction.sh hancock
#   sbatch benchmarks/scripts/slurm/submit_feature_extraction.sh tcga_luad --models virchow2 --skip_seg

#SBATCH --job-name=wsi_extract
#SBATCH --account=rrg-jma
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=h100:1
#SBATCH --mem=128G
#SBATCH --output=logs/extract_%x_%j.out
#SBATCH --error=logs/extract_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leo.yin@mail.utoronto.ca
#SBATCH --exclude=fc10512

set -uo pipefail

# ==================== DATASET ARG ====================
DATASET="${1:?Usage: sbatch $0 <dataset>  (e.g., clwd, ccrcc, ovarian)}"

# ==================== CONFIG ====================
PROJECT_DIR="/home/yinshuol/scratch/autoMIL/autoMIL"

# ==================== JOB INFO ====================
echo "================================================"
echo "AutoBench Feature Extraction — ${DATASET}"
echo "================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Dataset:   $DATASET"
echo "Node:      $(hostname)"
echo "GPUs:      $SLURM_GPUS_PER_NODE"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "Start:     $(date)"
echo "================================================"

# ==================== ENVIRONMENT ====================
module load cuda/12.2

cd "$PROJECT_DIR" || { echo "ERROR: Project directory not found"; exit 1; }
source .venv/bin/activate

# Load env vars from .env (HF_TOKEN, WANDB_API_KEY, dataset roots)
set -a
source benchmarks/.env
set +a

echo "Python:    $(which python)"
echo "CUDA:      $(nvcc --version 2>/dev/null | grep release || echo 'N/A')"

# ==================== GPU INFO ====================
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# ==================== CUDA CHECK ====================
echo ""
echo "PyTorch CUDA check..."
python -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Driver: {torch.version.cuda}')
else:
    print('  ERROR: CUDA not available — extraction will be extremely slow on CPU!')
    print('  Check: module load cuda version vs PyTorch CUDA version')
    import sys; sys.exit(1)
" || { echo "ERROR: CUDA not available. Aborting."; exit 1; }

# ==================== RUN ====================
echo ""
echo "Starting feature extraction..."
echo "================================================"

python benchmarks/scripts/run_feature_extraction.py \
    --dataset "$DATASET" \
    --gpu 0 \
    "${@:2}"

EXIT_CODE=$?

# ==================== DONE ====================
echo ""
echo "================================================"
echo "Feature extraction finished"
echo "Exit code: $EXIT_CODE"
echo "End time:  $(date)"
echo "================================================"

exit $EXIT_CODE
