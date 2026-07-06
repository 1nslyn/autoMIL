#!/bin/bash
# SLURM job: LUAD phase-2 TITAN arm — 5-fold CV, single GPU (slide-level, cheap).
#
# TITAN is a SLIDE-level foundation model: one 768-d embedding per slide (its own
# aggregation) -> frozen linear probe. Native recipe = CONCH v1.5 tiles @ 20x/512px
# -> TITAN pool. This job trains the head only; it REQUIRES the TITAN slide features
# to already exist (run submit_titan_extract_luad.sh first, then create the
# features_titan symlink — see that script / the plan).
#
# Writes results/titan/... into the SAME phase-2 benchmark_dir as the 4-framework
# grid, reusing its 5-fold splits. Runs independently of / after Track A.
#
# Usage:  sbatch benchmarks/scripts/submit_titan_luad_5fold.sh

#SBATCH --job-name=luad5f_titan
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

# ==================== CONFIG ====================
DATASET="tcga_luad"
TASKS="egfr kras"
N_FOLDS=5
PROJECT_DIR="/home/yinshuol/scratch/autoMIL/autoMIL"
SELF="$PROJECT_DIR/benchmarks/scripts/submit_titan_luad_5fold.sh"

echo "================================================"
echo "AutoBench LUAD — TITAN arm (5-fold, 1x H100)"
echo "Job ID: ${SLURM_JOB_ID:-N/A} | Node: $(hostname) | Start: $(date)"
echo "================================================"

# ==================== ENVIRONMENT ====================
module load cuda/12.2 2>/dev/null || true
cd "$PROJECT_DIR" || { echo "ERROR: Project dir not found: $PROJECT_DIR"; exit 1; }
source .venv/bin/activate
set -a; source benchmarks/.env; set +a

if [ -z "${AUTOBENCH_TCGA_LUAD_ROOT:-}" ]; then
    echo "ERROR: AUTOBENCH_TCGA_LUAD_ROOT not set (check benchmarks/.env)"; exit 1
fi
PATHOLOGY_ROOT="$(dirname "$(dirname "$AUTOBENCH_TCGA_LUAD_ROOT")")"
BENCHMARK_DIR="$PATHOLOGY_ROOT/autoMIL/phase2/tcga_luad/benchmark"
echo "Benchmark dir (phase-2): $BENCHMARK_DIR"

# Guard: TITAN slide features must be present under the default features_base_dir
# (20x_224px_0px_overlap/features_titan -> symlink to the 512px slide_features_titan).
TITAN_DIR="${AUTOBENCH_TCGA_LUAD_ROOT}/trident_output/20x_224px_0px_overlap/features_titan"
if [ ! -e "$TITAN_DIR" ]; then
    echo "ERROR: TITAN features not found at $TITAN_DIR"
    echo "       Run submit_titan_extract_luad.sh + create the features_titan symlink first."
    exit 1
fi
N_TITAN=$(ls "$TITAN_DIR"/*.h5 2>/dev/null | wc -l | tr -d ' ')
echo "TITAN slide-feature files: $N_TITAN"

echo "Python: $(which python)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || true

# ==================== TITAN TRAINING (1 GPU; slide-level linear probe) ====================
# --frameworks titan pins encoder="titan"/model="titan" internally; auto-preps the
# shared 5-fold splits (idempotent — reuses Track A's) and validates features_titan.
echo ""
echo "================ TITAN arm training ================"
python benchmarks/scripts/run_benchmark.py \
    --dataset "$DATASET" --benchmark_dir "$BENCHMARK_DIR" \
    --frameworks titan --tasks $TASKS --n_folds $N_FOLDS --gpu 0 --no_wandb
EXIT_CODE=$?

# ==================== AUTO-CONTINUATION ====================
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "TITAN arm completed successfully!"
elif [ $EXIT_CODE -eq 143 ] || [ $EXIT_CODE -eq 137 ]; then
    echo "Time limit (exit $EXIT_CODE) — auto-resubmitting (idempotent)..."
    NEW_JOB_ID=$(sbatch --parsable "$SELF")
    if [ $? -eq 0 ]; then echo "New job: $NEW_JOB_ID"; else echo "ERROR: resubmit failed. Run: sbatch $SELF"; fi
else
    echo "TITAN arm exited $EXIT_CODE — non-recoverable. Check logs."
fi
echo "End: $(date)"
exit $EXIT_CODE
