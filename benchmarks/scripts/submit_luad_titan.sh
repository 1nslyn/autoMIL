#!/bin/bash
# SLURM: TCGA-LUAD TITAN arm, 1x H100. Fold count is an OPTIONAL argument that
# defaults to 5. TITAN is a slide-level foundation model (768-d) -> linear probe.
#
# Requires the shared TITAN features (extract ONCE via submit_luad_titan_extract.sh,
# which is fold-independent). Writes results/titan into benchmark_<n>fold, reusing
# that run's <n>-fold splits (auto-prepped if absent).
#
# Usage:
#   sbatch benchmarks/scripts/submit_luad_titan.sh        # 5-fold (standard)
#   sbatch benchmarks/scripts/submit_luad_titan.sh 10     # 10-fold (comparison)
# Chain after extraction:  sbatch --dependency=afterok:<extract_jobid> ...

#SBATCH --job-name=luad_titan
#SBATCH --account=rrg-jma
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=96G
#SBATCH --output=/scratch/yinshuol/autoMIL/logs/titan_%x_%j.out
#SBATCH --error=/scratch/yinshuol/autoMIL/logs/titan_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leo.yin@mail.utoronto.ca

set -uo pipefail
N_FOLDS="${1:-5}"
[[ "$N_FOLDS" =~ ^[0-9]+$ ]] && [ "$N_FOLDS" -ge 2 ] || { echo "ERROR: n_folds must be an integer >= 2 (got '$N_FOLDS')"; exit 1; }

DATASET="tcga_luad"
TASKS="egfr kras"
PROJECT_DIR="/home/yinshuol/scratch/autoMIL/autoMIL"
SELF="$PROJECT_DIR/benchmarks/scripts/submit_luad_titan.sh"

echo "=== LUAD TITAN arm — ${N_FOLDS}-fold | Job ${SLURM_JOB_ID:-N/A} | $(hostname) | $(date) ==="
module load cuda/12.2 2>/dev/null || true
cd "$PROJECT_DIR" || { echo "ERROR: project dir not found"; exit 1; }
source .venv/bin/activate
set -a; source benchmarks/.env; set +a
[ -z "${AUTOBENCH_TCGA_LUAD_ROOT:-}" ] && { echo "ERROR: AUTOBENCH_TCGA_LUAD_ROOT unset"; exit 1; }
PATHOLOGY_ROOT="$(dirname "$(dirname "$AUTOBENCH_TCGA_LUAD_ROOT")")"
BENCHMARK_DIR="$PATHOLOGY_ROOT/autoMIL/phase2/tcga_luad/benchmark_${N_FOLDS}fold"
TITAN_DIR="${AUTOBENCH_TCGA_LUAD_ROOT}/trident_output/20x_224px_0px_overlap/features_titan"
[ ! -e "$TITAN_DIR" ] && { echo "ERROR: TITAN features missing at $TITAN_DIR. Run submit_luad_titan_extract.sh first."; exit 1; }
echo "Benchmark dir: $BENCHMARK_DIR | TITAN features: $(ls "$TITAN_DIR"/*.h5 2>/dev/null | wc -l | tr -d ' ')"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || true

python benchmarks/scripts/run_benchmark.py --dataset "$DATASET" --benchmark_dir "$BENCHMARK_DIR" \
    --frameworks titan --tasks $TASKS --n_folds $N_FOLDS --gpu 0 --no_wandb
EXIT=$?

if [ $EXIT -eq 0 ]; then
    echo "TITAN arm completed."
elif [ $EXIT -eq 143 ] || [ $EXIT -eq 137 ]; then
    echo "Time limit — resubmitting ${N_FOLDS}-fold..."
    sbatch --parsable "$SELF" "$N_FOLDS" || echo "ERROR: resubmit failed. Run: sbatch $SELF $N_FOLDS"
else
    echo "TITAN arm exited $EXIT — check logs."
fi
echo "End: $(date)"
exit $EXIT
