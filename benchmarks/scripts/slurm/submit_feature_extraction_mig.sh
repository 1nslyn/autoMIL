#!/bin/bash
# SLURM job script: WSI feature extraction on Fir HPC (MIG GPU)
#
# Uses a 3g.40gb MIG slice of H100 instead of a full H100 — sufficient
# for inference-only feature extraction and more resource-efficient.
#
# Usage:
#   sbatch benchmarks/scripts/slurm/submit_feature_extraction_mig.sh <dataset> [models...]
#   sbatch benchmarks/scripts/slurm/submit_feature_extraction_mig.sh tcga_luad
#   sbatch benchmarks/scripts/slurm/submit_feature_extraction_mig.sh tcga_luad virchow2 hoptimus1 uni_v2

#SBATCH --job-name=wsi_extract
#SBATCH --account=rrg-jma
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --mem=64G
#SBATCH --output=logs/extract_%x_%j.out
#SBATCH --error=logs/extract_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leo.yin@mail.utoronto.ca

set -uo pipefail

# ==================== ARGS ====================
DATASET="${1:?Usage: sbatch $0 <dataset> [models...]}"
shift
MODELS="$@"

# ==================== CONFIG ====================
PROJECT_DIR="/home/yinshuol/scratch/autoMIL/autoMIL"

# ==================== JOB INFO ====================
echo "================================================"
echo "AutoBench Feature Extraction — ${DATASET}"
echo "================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Dataset:   $DATASET"
echo "Models:    ${MODELS:-all from config}"
echo "Node:      $(hostname)"
echo "GPUs:      $SLURM_GPUS_PER_NODE"
echo "CPUs:      $SLURM_CPUS_PER_TASK"
echo "Start:     $(date)"
echo "================================================"

# ==================== ENVIRONMENT ====================
module load cuda/12.2

cd "$PROJECT_DIR" || { echo "ERROR: Project directory not found"; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv is required on the compute node"; exit 1; }
UV_RUN=(uv run --frozen --no-sync --package autobench)

# Load env vars from .env (HF_TOKEN, WANDB_API_KEY, dataset roots)
set -a
source benchmarks/.env
set +a

echo "Python:    $("${UV_RUN[@]}" python -c 'import sys; print(sys.executable)')"
echo "CUDA:      $(nvcc --version 2>/dev/null | grep release || echo 'N/A')"

# ==================== GPU INFO ====================
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# ==================== CUDA CHECK ====================
echo ""
echo "PyTorch CUDA check..."
"${UV_RUN[@]}" python -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Driver: {torch.version.cuda}')
else:
    print('  ERROR: CUDA not available!')
    import sys; sys.exit(1)
" || { echo "ERROR: CUDA not available. Aborting."; exit 1; }

# ==================== RUN ====================
echo ""
echo "Starting feature extraction..."
echo "================================================"

CMD=("${UV_RUN[@]}" python benchmarks/scripts/run_feature_extraction.py --dataset "$DATASET" --gpu 0)
if [ -n "$MODELS" ]; then
    read -r -a MODEL_ARGS <<< "$MODELS"
    CMD+=(--models "${MODEL_ARGS[@]}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

EXIT_CODE=$?

# ==================== DONE ====================
echo ""
echo "================================================"
echo "Feature extraction finished"
echo "Exit code: $EXIT_CODE"
echo "End time:  $(date)"
echo "================================================"

exit $EXIT_CODE
