#!/bin/bash
# SLURM job: SINGLE-GPU LUAD benchmark (clam_mb + simple_mil) — fix-validation run.
#
# Purpose: exercise the in-process global-torch-state isolation fix
# (orchestrator._isolated_torch_state).  The single-GPU path runs every
# experiment back-to-back in ONE process, so the nnMIL -> CLAM transition that
# previously crashed ("element 0 of tensors does not require grad") is hit for
# real here, unlike the multi-GPU path which isolates each experiment in its
# own process.
#
# Grid: frameworks {clam, nnmil} x models {clam_mb, simple_mil}
#       x encoders {hoptimus1, uni_v2, virchow2} x tasks {egfr, kras}  = 12 exps
#
# Idempotent: completed experiments/folds are skipped, so the 24h time-limit
# auto-resubmit below resumes a long (~30h) single-GPU run across jobs.
#
# Usage:  sbatch benchmarks/scripts/submit_benchmark_luad_singlegpu.sh

#SBATCH --job-name=autobench_luad_sgpu
#SBATCH --account=rrg-jma
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=96G
#SBATCH --output=/scratch/yinshuol/autoMIL/logs/bench_%x_%j.out
#SBATCH --error=/scratch/yinshuol/autoMIL/logs/bench_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leo.yin@mail.utoronto.ca

set -uo pipefail

# ==================== CONFIG ====================
DATASET="tcga_luad"
FRAMEWORKS="clam nnmil"
ENCODERS="hoptimus1 uni_v2 virchow2"
TASKS="egfr kras"
CLAM_MODELS="clam_mb"
NNMIL_MODELS="simple_mil"
# LUAD uses 10-fold patient-stratified CV (post-fairness re-run, May 2026).
# Must match existing splits or fold-loading fails on splits_<N>.csv.
N_FOLDS=10
PROJECT_DIR="/scratch/yinshuol/autoMIL/autoMIL"
SELF="$PROJECT_DIR/benchmarks/scripts/submit_benchmark_luad_singlegpu.sh"

# ==================== JOB INFO ====================
echo "================================================"
echo "AutoBench LUAD — SINGLE-GPU fix-validation run"
echo "================================================"
echo "Job ID:      ${SLURM_JOB_ID:-N/A}"
echo "Dataset:     $DATASET"
echo "Frameworks:  $FRAMEWORKS   (CLAM: $CLAM_MODELS | nnMIL: $NNMIL_MODELS)"
echo "Encoders:    $ENCODERS"
echo "Tasks:       $TASKS"
echo "Node:        $(hostname)"
echo "GPUs:        ${SLURM_GPUS_PER_NODE:-N/A}"
echo "Start:       $(date)"
echo "================================================"

# ==================== ENVIRONMENT ====================
module load cuda/12.2 2>/dev/null || true

cd "$PROJECT_DIR" || { echo "ERROR: Project directory not found"; exit 1; }
source .venv/bin/activate

# Load dataset-specific env vars (AUTOBENCH_TCGA_LUAD_ROOT, ...)
set -a
source benchmarks/.env
set +a

echo "Python:      $(which python)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || true

# ==================== DATA PREP ====================
echo ""
echo "================ Phase 1: Data Preparation ================"
python benchmarks/scripts/run_benchmark.py \
    --dataset "$DATASET" \
    --prep_only \
    --encoders $ENCODERS \
    --tasks $TASKS \
    --n_folds $N_FOLDS
PREP_EXIT=$?
if [ $PREP_EXIT -ne 0 ]; then
    echo "ERROR: Data preparation failed (exit $PREP_EXIT)"
    exit $PREP_EXIT
fi

# ==================== BENCHMARK (SINGLE GPU) ====================
echo ""
echo "================ Phase 2: Benchmark (single GPU) ================"
CMD=(python benchmarks/scripts/run_benchmark.py
    --dataset "$DATASET"
    --gpu 0
    --frameworks $FRAMEWORKS
    --models $CLAM_MODELS
    --nnmil_models $NNMIL_MODELS
    --encoders $ENCODERS
    --tasks $TASKS
    --n_folds $N_FOLDS
    --no_wandb
)
echo "Command: ${CMD[*]}"
echo ""
"${CMD[@]}"
EXIT_CODE=$?

# ==================== AUTO-CONTINUATION ====================
echo ""
echo "================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Benchmark completed successfully!"
elif [ $EXIT_CODE -eq 143 ] || [ $EXIT_CODE -eq 137 ]; then
    echo "Time limit reached (exit $EXIT_CODE) — auto-resubmitting (idempotent resume)..."
    NEW_JOB_ID=$(sbatch --parsable "$SELF")
    if [ $? -eq 0 ]; then
        echo "New job submitted: $NEW_JOB_ID"
    else
        echo "ERROR: resubmit failed. Manually run: sbatch $SELF"
    fi
else
    echo "Benchmark exited with code $EXIT_CODE — non-recoverable. Check logs."
fi
echo "End time: $(date)"
echo "================================================"
exit $EXIT_CODE
