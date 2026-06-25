#!/bin/bash
# SLURM job script: CPTAC-CCRCC OS survival benchmark (nnMIL only)
#
# Runs the nnMIL survival benchmark for CPTAC-CCRCC (os task, cox + nllsurv).
# The pipeline is idempotent — resubmitting resumes from where it left off.
#
# Usage:
#   sbatch benchmarks/scripts/submit_cptac_ccrcc_survival.sh
#
# Overrides:
#   ENCODERS="virchow2" sbatch benchmarks/scripts/submit_cptac_ccrcc_survival.sh

#SBATCH --job-name=ccrcc_surv
#SBATCH --account=def-jma-ab
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=0
#SBATCH --output=logs/bench_%x_%j.out
#SBATCH --error=logs/bench_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yeonwoo.seo@mail.utoronto.ca

# ==================== CONFIG ====================
DATASET="cptac_ccrcc"
FRAMEWORKS="nnmil"
TASKS="os"
PROJECT_DIR="/home/yws0322/scratch/autoMIL"

# ==================== JOB INFO ====================
echo "================================================"
echo "AutoBench — CPTAC-CCRCC OS Survival (nnMIL)"
echo "================================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Dataset:     $DATASET"
echo "Tasks:       $TASKS"
echo "Frameworks:  $FRAMEWORKS"
echo "Node:        $(hostname)"
echo "GPUs:        $SLURM_GPUS_PER_NODE"
echo "CPUs:        $SLURM_CPUS_PER_TASK"
echo "Start:       $(date)"
echo "================================================"

# ==================== ENVIRONMENT ====================
module load cuda/12.2

cd "$PROJECT_DIR" || { echo "ERROR: Project directory not found"; exit 1; }
source .venv/bin/activate

set -a
source benchmarks/.env
set +a

echo "Python:      $(which python)"
echo "CUDA:        $(nvcc --version 2>/dev/null | grep release || echo 'N/A')"

# ==================== GPU INFO ====================
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# ==================== DATA PREP ====================
echo ""
echo "================================================"
echo "Phase 1: Data Preparation"
echo "================================================"

PREP_ARGS=(
    --dataset "$DATASET"
    --tasks "$TASKS"
    --prep_only
)

if [ -n "$ENCODERS" ]; then
    PREP_ARGS+=(--encoders $ENCODERS)
fi

python benchmarks/scripts/run_benchmark.py "${PREP_ARGS[@]}"
PREP_EXIT=$?

if [ $PREP_EXIT -ne 0 ]; then
    echo "ERROR: Data preparation failed (exit $PREP_EXIT)"
    exit $PREP_EXIT
fi

# ==================== BENCHMARK ====================
echo ""
echo "================================================"
echo "Phase 2: Survival Training"
echo "================================================"

BENCH_ARGS=(
    --dataset "$DATASET"
    --tasks "$TASKS"
    --frameworks $FRAMEWORKS
    --all_gpus
    --no_wandb
)

if [ -n "$ENCODERS" ]; then
    BENCH_ARGS+=(--encoders $ENCODERS)
fi

if [ -n "$MODELS" ]; then
    BENCH_ARGS+=(--nnmil_models $MODELS)
fi

if [ -n "$SEED" ]; then
    BENCH_ARGS+=(--seed "$SEED")
fi

echo "Command: python benchmarks/scripts/run_benchmark.py ${BENCH_ARGS[*]}"
echo ""

python benchmarks/scripts/run_benchmark.py "${BENCH_ARGS[@]}"
EXIT_CODE=$?

# ==================== AUTO-CONTINUATION ====================
echo ""
echo "================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Survival benchmark completed successfully!"
else
    echo "Benchmark exited with code $EXIT_CODE"

    if [ $EXIT_CODE -eq 143 ] || [ $EXIT_CODE -eq 137 ]; then
        echo ""
        echo "Time limit reached — auto-resubmitting..."
        echo "Pipeline is idempotent: completed experiments will be skipped."

        cd "$PROJECT_DIR"
        NEW_JOB_ID=$(sbatch --parsable benchmarks/scripts/submit_cptac_ccrcc_survival.sh)

        if [ $? -eq 0 ]; then
            echo "New job submitted: $NEW_JOB_ID"
            echo "Monitor: squeue -u $USER"
            echo "Logs:    tail -f logs/bench_ccrcc_surv_${NEW_JOB_ID}.out"
        else
            echo "ERROR: Failed to resubmit. Manually run:"
            echo "  sbatch benchmarks/scripts/submit_cptac_ccrcc_survival.sh"
        fi
    else
        echo "Non-recoverable error. Check logs."
    fi
fi

echo ""
echo "End time: $(date)"
echo "================================================"

exit $EXIT_CODE
