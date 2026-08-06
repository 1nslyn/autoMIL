#!/bin/bash
# SLURM: TITAN arm for ANY dataset, 1x H100. Fold count optional (default 5).
# TITAN is a slide-level foundation model (768-d) -> linear probe.
#
# Requires the shared TITAN features (extract ONCE via submit_titan_extract.sh
# <dataset>, which is fold-independent). Writes results/titan into the phase-2
# benchmark_<n>fold, reusing that run's <n>-fold splits (auto-prepped if absent).
#
# Usage:
#   sbatch benchmarks/scripts/slurm/submit_titan.sh <dataset>        # 5-fold (standard)
#   sbatch benchmarks/scripts/slurm/submit_titan.sh <dataset> 10     # 10-fold (comparison)
# Chain after extraction:
#   sbatch --dependency=afterok:<extract_jobid> submit_titan.sh <dataset> [n_folds]

#SBATCH --job-name=titan_arm
#SBATCH --account=rrg-jma
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=96G
#SBATCH --output=logs/titan_%x_%j.out
#SBATCH --error=logs/titan_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

set -uo pipefail
DATASET="${1:-${DATASET:-}}"
N_FOLDS="${2:-${N_FOLDS:-5}}"
[ -n "$DATASET" ] || { echo "usage: sbatch submit_titan.sh <dataset> [n_folds]"; exit 1; }
[[ "$N_FOLDS" =~ ^[0-9]+$ ]] && [ "$N_FOLDS" -ge 2 ] || { echo "ERROR: n_folds must be an integer >= 2 (got '$N_FOLDS')"; exit 1; }
PROJECT_DIR="${SLURM_SUBMIT_DIR:-/home/yinshuol/scratch/autoMIL/autoMIL}"
SELF="$PROJECT_DIR/benchmarks/scripts/slurm/submit_titan.sh"

echo "=== ${DATASET} TITAN arm — ${N_FOLDS}-fold | Job ${SLURM_JOB_ID:-N/A} | $(hostname) | $(date) ==="
module load cuda/12.2 2>/dev/null || true
cd "$PROJECT_DIR" || { echo "ERROR: project dir not found: $PROJECT_DIR"; exit 1; }
mkdir -p logs
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv is required on the compute node"; exit 1; }
UV_RUN=(uv run --frozen --no-sync --package autobench)
set -a; source benchmarks/.env; set +a

DATA_ROOT=$("${UV_RUN[@]}" python -c "from autobench.config import load_dataset_config as L; print(L('${DATASET}').data_root)") \
    || { echo "ERROR: cannot load dataset config '${DATASET}' (check name + AUTOBENCH_*_ROOT in benchmarks/.env)"; exit 1; }
FEATURES_BASE=$("${UV_RUN[@]}" python -c "from autobench.config import load_dataset_config as L; print(L('${DATASET}').features_base_dir)")
PATHOLOGY_ROOT="$(dirname "$(dirname "$DATA_ROOT")")"
BENCHMARK_DIR="$PATHOLOGY_ROOT/autoMIL/phase2/${DATASET}/benchmark_${N_FOLDS}fold"
TITAN_DIR="$FEATURES_BASE/features_titan"
[ ! -e "$TITAN_DIR" ] && { echo "ERROR: TITAN features missing at $TITAN_DIR. Run: sbatch submit_titan_extract.sh ${DATASET}"; exit 1; }
echo "Benchmark dir: $BENCHMARK_DIR | TITAN features: $(ls "$TITAN_DIR"/*.h5 2>/dev/null | wc -l | tr -d ' ')"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true

"${UV_RUN[@]}" python benchmarks/scripts/run_benchmark.py --dataset "$DATASET" --benchmark_dir "$BENCHMARK_DIR" \
    --frameworks titan --n_folds "$N_FOLDS" --gpu 0 --no_wandb
EXIT=$?

if [ $EXIT -eq 0 ]; then
    echo "TITAN arm completed."
elif [ $EXIT -eq 143 ] || [ $EXIT -eq 137 ]; then
    echo "Time limit — resubmitting ${DATASET} ${N_FOLDS}-fold..."
    sbatch --parsable "$SELF" "$DATASET" "$N_FOLDS" || echo "ERROR: resubmit failed. Run: sbatch $SELF $DATASET $N_FOLDS"
else
    echo "TITAN arm exited $EXIT — check logs."
fi
echo "End: $(date)"
exit $EXIT
