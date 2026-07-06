#!/bin/bash
# SLURM job script: survival (time-to-event) MIL benchmark training.
#
# Runs the CLAM + nnMIL survival grid for a dataset's survival task(s).
# Experiments are distributed across GPUs with memory-budget scheduling.
# The pipeline is idempotent — resubmitting resumes from where it left off.
#
# Usage:
#   DATASET=cptac_ccrcc sbatch benchmarks/scripts/submit_survival_benchmark.sh
#
# Overrides (env vars):
#   DATASET=tcga_luad             # dataset config name (default: cptac_ccrcc)
#   TASKS="os"                    # survival task name(s) (default: os)
#   FRAMEWORKS="clam nnmil"       # frameworks (default: both)
#   ENCODERS="virchow2 uni_v2"    # encoder subset (default: all from YAML)
#   MODELS="clam_sb"              # CLAM model subset
#   NNMIL_MODELS="ab_mil"         # nnMIL model subset
#   SEED=42
#
# Note: survival CV is pinned to 5 folds in code (resolve_n_folds); --n_folds
# has no effect on survival tasks.

#SBATCH --job-name=autobench_surv
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

set -uo pipefail

# ==================== CONFIG ====================
DATASET="${DATASET:-cptac_ccrcc}"
TASKS="${TASKS:-os}"
FRAMEWORKS="${FRAMEWORKS:-clam nnmil}"
PROJECT_DIR="${PROJECT_DIR:-/home/yws0322/scratch/autoMIL}"

# Optional overrides — default empty so `set -u` doesn't trip.
ENCODERS="${ENCODERS:-}"
MODELS="${MODELS:-}"
NNMIL_MODELS="${NNMIL_MODELS:-}"
SEED="${SEED:-}"

SCRIPT="benchmarks/scripts/submit_survival_benchmark.sh"

# ==================== JOB INFO ====================
echo "================================================"
echo "AutoBench — Survival Benchmark"
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

cd "$PROJECT_DIR" || { echo "ERROR: Project directory not found: $PROJECT_DIR"; exit 1; }
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

# ==================== VALIDATION ====================
echo ""
echo "Validating dataset config..."
python -c "
from autobench.config import load_dataset_config
from autobench.pipeline.config import build_registries, generate_all_experiments, BenchmarkConfig, Framework
ds = load_dataset_config('${DATASET}')
registries = build_registries(ds)
print(f'  Dataset:  {ds.name} — {ds.description}')
print(f'  Tasks:    {list(ds.tasks.keys())}')
print(f'  Encoders: {list(ds.encoder_dims.keys())}')
frameworks = [Framework.CLAM if f == 'clam' else Framework.NNMIL for f in '${FRAMEWORKS}'.split()]
cfg = BenchmarkConfig.from_dataset_config(ds, frameworks=frameworks)
tasks = '${TASKS}'.split()
exps = [e for e in generate_all_experiments(cfg, registries) if e.task.name in tasks]
print(f'  Survival experiments ({tasks}): {len(exps)}')
" || { echo "ERROR: Failed to load dataset config"; exit 1; }

# ==================== DATA PREP ====================
echo ""
echo "================================================"
echo "Phase 1: Data Preparation"
echo "================================================"

PREP_ARGS=( --dataset "$DATASET" --tasks $TASKS --prep_only )
[ -n "$ENCODERS" ] && PREP_ARGS+=(--encoders $ENCODERS)

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

BENCH_ARGS=( --dataset "$DATASET" --tasks $TASKS --frameworks $FRAMEWORKS --all_gpus --no_wandb )
[ -n "$ENCODERS" ]     && BENCH_ARGS+=(--encoders $ENCODERS)
[ -n "$MODELS" ]       && BENCH_ARGS+=(--models $MODELS)
[ -n "$NNMIL_MODELS" ] && BENCH_ARGS+=(--nnmil_models $NNMIL_MODELS)
[ -n "$SEED" ]         && BENCH_ARGS+=(--seed "$SEED")

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
        echo "Time limit reached — auto-resubmitting (idempotent)..."
        cd "$PROJECT_DIR"
        NEW_JOB_ID=$(DATASET="$DATASET" TASKS="$TASKS" FRAMEWORKS="$FRAMEWORKS" \
            ENCODERS="$ENCODERS" MODELS="$MODELS" NNMIL_MODELS="$NNMIL_MODELS" SEED="$SEED" \
            sbatch --parsable "$SCRIPT")
        if [ $? -eq 0 ]; then
            echo "New job submitted: $NEW_JOB_ID"
            echo "Monitor: squeue -u $USER"
        else
            echo "ERROR: Failed to resubmit. Manually run: sbatch $SCRIPT"
        fi
    else
        echo "Non-recoverable error. Check logs."
    fi
fi

echo ""
echo "End time: $(date)"
echo "================================================"
exit $EXIT_CODE
