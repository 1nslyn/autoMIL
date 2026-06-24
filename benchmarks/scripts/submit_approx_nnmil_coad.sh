#!/bin/bash
# SLURM job: TCGA-COAD BRAF — nnMIL-method approximation of the nnMIL-paper number.
#
# Why this exists: the nnMIL paper evaluates colorectal BRAF by training on SURGEN
# and using TCGA-CRC (COAD+READ) as a WHOLE-cohort external test set. We can't
# reproduce that protocol (no SURGEN cohort, and we have COAD only — no READ), so
# this is an APPROXIMATION: in-cohort 5-fold patient-stratified CV using nnMIL's
# own split recipe (seed 42, val_frac 0.125) on TCGA-COAD.
#
# BRAF is rare in COAD (~55 positive cases after feature-filtering) so AUC will be
# high-variance — treat the number as a rough comparator, not a precise match.
#
# Outputs go to Leo's scratch; features read from the shared trident H5 store via
# symlinks under $BENCHMARK_DIR/features (forced CLAM H5->PT step no-ops).
#
# Grid: framework {nnmil} x model {simple_mil} x encoders {hoptimus1,uni_v2,virchow2}
#       x task {braf} x 5 folds = 15 fold-runs.
#
# Idempotent: completed folds are skipped, so the time-limit auto-resubmit resumes.
#
# Usage:  sbatch benchmarks/scripts/submit_approx_nnmil_coad.sh

#SBATCH --job-name=approx_nnmil_coad
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
DATASET="tcga_coad"
TASKS="braf"
FRAMEWORKS="nnmil"
NNMIL_MODELS="simple_mil"
ENCODERS="hoptimus1 uni_v2 virchow2"
# nnMIL's native planner default is 5-fold.
N_FOLDS=5
PROJECT_DIR="/scratch/yinshuol/autoMIL/autoMIL"
BENCHMARK_DIR="/scratch/yinshuol/autoMIL/approx_nnmil/tcga_coad/benchmark"
SELF="$PROJECT_DIR/benchmarks/scripts/submit_approx_nnmil_coad.sh"

# ==================== JOB INFO ====================
echo "================================================"
echo "AutoBench TCGA-COAD BRAF — nnMIL-method approximation (5-fold)"
echo "================================================"
echo "Job ID:        ${SLURM_JOB_ID:-N/A}"
echo "Dataset:       $DATASET   Task: $TASKS"
echo "Framework:     $FRAMEWORKS ($NNMIL_MODELS)   Encoders: $ENCODERS   Folds: $N_FOLDS"
echo "Benchmark dir: $BENCHMARK_DIR"
echo "Node:          $(hostname)   Start: $(date)"
echo "================================================"

# ==================== ENVIRONMENT ====================
module load cuda/12.2 2>/dev/null || true
cd "$PROJECT_DIR" || { echo "ERROR: Project directory not found"; exit 1; }
source .venv/bin/activate
set -a; source benchmarks/.env; set +a
echo "Python: $(which python)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || true

# ==================== DATA PREP ====================
echo ""
echo "================ Phase 1: Data Preparation ================"
python benchmarks/scripts/run_benchmark.py \
    --dataset "$DATASET" \
    --benchmark_dir "$BENCHMARK_DIR" \
    --prep_only \
    --encoders $ENCODERS \
    --tasks $TASKS \
    --n_folds $N_FOLDS
PREP_EXIT=$?
if [ $PREP_EXIT -ne 0 ]; then
    echo "ERROR: Data preparation failed (exit $PREP_EXIT)"; exit $PREP_EXIT
fi

# ==================== BENCHMARK (SINGLE GPU) ====================
echo ""
echo "================ Phase 2: Benchmark (nnMIL, single GPU) ================"
CMD=(python benchmarks/scripts/run_benchmark.py
    --dataset "$DATASET"
    --benchmark_dir "$BENCHMARK_DIR"
    --gpu 0
    --frameworks $FRAMEWORKS
    --nnmil_models $NNMIL_MODELS
    --encoders $ENCODERS
    --tasks $TASKS
    --n_folds $N_FOLDS
    --no_wandb
)
echo "Command: ${CMD[*]}"; echo ""
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
    [ $? -eq 0 ] && echo "New job submitted: $NEW_JOB_ID" || echo "ERROR: resubmit failed. Run: sbatch $SELF"
else
    echo "Benchmark exited with code $EXIT_CODE — non-recoverable. Check logs."
fi
echo "End time: $(date)"
echo "================================================"
exit $EXIT_CODE
