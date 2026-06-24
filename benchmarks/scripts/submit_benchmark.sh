#!/bin/bash
# SLURM job script: MIL benchmark training on Fir HPC
#
# Runs the full CLAM + nnMIL benchmark grid for a dataset.
# Experiments are distributed across GPUs with memory-budget scheduling.
# The pipeline is idempotent — resubmitting resumes from where it left off.
#
# Usage:
#   sbatch benchmarks/scripts/submit_benchmark.sh
#
# To override dataset (default: clwd):
#   DATASET=ovarian sbatch benchmarks/scripts/submit_benchmark.sh
#
# To run only specific frameworks:
#   FRAMEWORKS="clam" sbatch benchmarks/scripts/submit_benchmark.sh
#   FRAMEWORKS="nnmil" sbatch benchmarks/scripts/submit_benchmark.sh
#   FRAMEWORKS="clam nnmil" sbatch benchmarks/scripts/submit_benchmark.sh
#
# To run a subset:
#   ENCODERS="conch_v15 hibou_l" TASKS="brca" sbatch benchmarks/scripts/submit_benchmark.sh

#SBATCH --job-name=autobench_train
#SBATCH --account=rrg-jma
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=0
#SBATCH --output=logs/bench_%x_%j.out
#SBATCH --error=logs/bench_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leo.yin@mail.utoronto.ca

set -uo pipefail

# ==================== CONFIG ====================
DATASET="${DATASET:-clwd}"
FRAMEWORKS="${FRAMEWORKS:-clam nnmil}"
PROJECT_DIR="/home/yinshuol/scratch/autoMIL/autoMIL"

# ==================== JOB INFO ====================
echo "================================================"
echo "AutoBench MIL Benchmark"
echo "================================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Dataset:     $DATASET"
echo "Frameworks:  $FRAMEWORKS"
echo "Node:        $(hostname)"
echo "GPUs:        $SLURM_GPUS_PER_NODE"
echo "CPUs:        $SLURM_CPUS_PER_TASK"
echo "Wall time:   $SLURM_TIMELIMIT"
echo "Start:       $(date)"
echo "================================================"

# ==================== ENVIRONMENT ====================
module load cuda/12.2

cd "$PROJECT_DIR" || { echo "ERROR: Project directory not found"; exit 1; }
source .venv/bin/activate

# Load dataset-specific env vars
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
print(f'  Dataset:    {ds.name} — {ds.description}')
print(f'  Tasks:      {list(ds.tasks.keys())}')
print(f'  Strategies: {list(ds.split_strategies.keys())}')
print(f'  Encoders:   {list(ds.encoder_dims.keys())}')
print(f'  CLAM models:  {ds.clam_models}')
print(f'  nnMIL models:  {ds.nnmil_models}')
# Count experiments
frameworks = []
for f in '${FRAMEWORKS}'.split():
    frameworks.append(Framework.CLAM if f == 'clam' else Framework.NNMIL)
cfg = BenchmarkConfig.from_dataset_config(ds, frameworks=frameworks)
exps = generate_all_experiments(cfg, registries)
print(f'  Total experiments: {len(exps)}')
" || { echo "ERROR: Failed to load dataset config"; exit 1; }

# ==================== DATA PREP ====================
echo ""
echo "================================================"
echo "Phase 1: Data Preparation"
echo "================================================"

PREP_ARGS=(
    --dataset "$DATASET"
    --prep_only
)

if [ -n "$ENCODERS" ]; then
    PREP_ARGS+=(--encoders $ENCODERS)
fi

if [ -n "$TASKS" ]; then
    PREP_ARGS+=(--tasks $TASKS)
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
echo "Phase 2: Benchmark Training"
echo "================================================"

BENCH_ARGS=(
    --dataset "$DATASET"
    --all_gpus
    --frameworks $FRAMEWORKS
    --no_wandb
)

if [ -n "$ENCODERS" ]; then
    BENCH_ARGS+=(--encoders $ENCODERS)
fi

if [ -n "$TASKS" ]; then
    BENCH_ARGS+=(--tasks $TASKS)
fi

if [ -n "$MODELS" ]; then
    BENCH_ARGS+=(--models $MODELS)
fi

if [ -n "$NNMIL_MODELS" ]; then
    BENCH_ARGS+=(--nnmil_models $NNMIL_MODELS)
fi

if [ -n "$SEED" ]; then
    BENCH_ARGS+=(--seed "$SEED")
fi

if [ -n "$N_FOLDS" ]; then
    BENCH_ARGS+=(--n_folds "$N_FOLDS")
fi

echo "Command: python benchmarks/scripts/run_benchmark.py ${BENCH_ARGS[*]}"
echo ""

python benchmarks/scripts/run_benchmark.py "${BENCH_ARGS[@]}"
EXIT_CODE=$?

# ==================== AUTO-CONTINUATION ====================
echo ""
echo "================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Benchmark completed successfully!"
else
    echo "Benchmark exited with code $EXIT_CODE"

    # Check if it was a time limit (SIGTERM from SLURM)
    if [ $EXIT_CODE -eq 143 ] || [ $EXIT_CODE -eq 137 ]; then
        echo ""
        echo "Time limit reached — auto-resubmitting..."
        echo "Pipeline is idempotent: completed experiments will be skipped."

        cd "$PROJECT_DIR"
        NEW_JOB_ID=$(sbatch --parsable benchmarks/scripts/submit_benchmark.sh)

        if [ $? -eq 0 ]; then
            echo "New job submitted: $NEW_JOB_ID"
            echo "Monitor: squeue -u $USER"
            echo "Logs:    tail -f logs/bench_autobench_train_${NEW_JOB_ID}.out"
        else
            echo "ERROR: Failed to resubmit. Manually run:"
            echo "  sbatch benchmarks/scripts/submit_benchmark.sh"
        fi
    else
        echo "Non-recoverable error. Check logs."
    fi
fi

echo ""
echo "End time: $(date)"
echo "================================================"

exit $EXIT_CODE
