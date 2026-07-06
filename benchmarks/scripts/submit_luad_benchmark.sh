#!/bin/bash
# SLURM: TCGA-LUAD benchmark grid, 4x H100. Cross-validation fold count is an
# OPTIONAL argument that defaults to 5 (the lab standard) — group members just
# run it with no argument.
#
# Grid: CLAM{clam_sb,clam_mb,mil} + nnMIL{trans_mil,simple_mil} + DTFD{dtfd_mil}
#       + ABMIL{abmil}  x 3 encoders x 2 tasks = 42 experiments x <n_folds> folds.
#       (TITAN is a separate arm — see submit_luad_titan.sh.)
# Writes to <Pathology>/autoMIL/phase2/tcga_luad/benchmark_<n>fold (isolated per
# fold count). Idempotent; auto-resubmits on the 24h limit, preserving <n_folds>.
#
# Usage:
#   sbatch benchmarks/scripts/submit_luad_benchmark.sh        # 5-fold (standard)
#   sbatch benchmarks/scripts/submit_luad_benchmark.sh 10     # 10-fold (comparison)

#SBATCH --job-name=luad_bench
#SBATCH --account=rrg-jma
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=0
#SBATCH --output=/scratch/yinshuol/autoMIL/logs/bench_%x_%j.out
#SBATCH --error=/scratch/yinshuol/autoMIL/logs/bench_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leo.yin@mail.utoronto.ca

set -uo pipefail
N_FOLDS="${1:-5}"
[[ "$N_FOLDS" =~ ^[0-9]+$ ]] && [ "$N_FOLDS" -ge 2 ] || { echo "ERROR: n_folds must be an integer >= 2 (got '$N_FOLDS')"; exit 1; }

DATASET="tcga_luad"
FRAMEWORKS="clam nnmil dtfd abmil"
ENCODERS="hoptimus1 uni_v2 virchow2"
TASKS="egfr kras"
CLAM_MODELS="clam_sb clam_mb mil"
NNMIL_MODELS="trans_mil simple_mil"
DTFD_MODELS="dtfd_mil"
ABMIL_MODELS="abmil"
PROJECT_DIR="/home/yinshuol/scratch/autoMIL/autoMIL"
SELF="$PROJECT_DIR/benchmarks/scripts/submit_luad_benchmark.sh"

echo "================================================"
echo "AutoBench LUAD grid — ${N_FOLDS}-fold, 4x H100"
echo "Job ${SLURM_JOB_ID:-N/A} | $(hostname) | $(date)"
echo "================================================"

module load cuda/12.2 2>/dev/null || true
cd "$PROJECT_DIR" || { echo "ERROR: project dir not found: $PROJECT_DIR"; exit 1; }
source .venv/bin/activate
set -a; source benchmarks/.env; set +a
[ -z "${AUTOBENCH_TCGA_LUAD_ROOT:-}" ] && { echo "ERROR: AUTOBENCH_TCGA_LUAD_ROOT unset (check benchmarks/.env)"; exit 1; }
PATHOLOGY_ROOT="$(dirname "$(dirname "$AUTOBENCH_TCGA_LUAD_ROOT")")"
BENCHMARK_DIR="$PATHOLOGY_ROOT/autoMIL/phase2/tcga_luad/benchmark_${N_FOLDS}fold"
echo "Benchmark dir: $BENCHMARK_DIR"
mkdir -p "$BENCHMARK_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || true

# --- validate grid == 42 experiments (maps via the real _FRAMEWORK_MAP) ---
echo "===== validate config ====="
python - <<PYEOF || { echo "ERROR: config validation failed"; exit 1; }
from autobench.config import load_dataset_config
from autobench.pipeline.config import build_registries, generate_all_experiments, BenchmarkConfig, Framework
_MAP={"clam":Framework.CLAM,"nnmil":Framework.NNMIL,"dtfd":Framework.DTFD,"titan":Framework.TITAN,"abmil":Framework.ABMIL}
ds=load_dataset_config("${DATASET}"); reg=build_registries(ds)
cfg=BenchmarkConfig.from_dataset_config(ds, frameworks=[_MAP[f] for f in "${FRAMEWORKS}".split()],
    encoder_keys="${ENCODERS}".split(), tasks="${TASKS}".split(), model_types="${CLAM_MODELS}".split(),
    nnmil_model_types="${NNMIL_MODELS}".split(), dtfd_model_types="${DTFD_MODELS}".split(),
    abmil_model_types="${ABMIL_MODELS}".split(), n_folds=${N_FOLDS})
exps=generate_all_experiments(cfg,reg)
print(f"  experiments={len(exps)} fold-trainings={len(exps)*${N_FOLDS}} (expect 42 experiments)")
assert len(exps)==42, f"expected 42 experiments, got {len(exps)}"
print("  OK")
PYEOF

# --- prep: n-fold splits + tile H5->PT ---
echo "===== prep (${N_FOLDS}-fold splits) ====="
python benchmarks/scripts/run_benchmark.py --dataset "$DATASET" --benchmark_dir "$BENCHMARK_DIR" \
    --prep_only --encoders $ENCODERS --tasks $TASKS --n_folds $N_FOLDS || { echo "ERROR: prep failed"; exit 1; }

# --- train (4x H100) ---
echo "===== train (4x H100) ====="
python benchmarks/scripts/run_benchmark.py --dataset "$DATASET" --benchmark_dir "$BENCHMARK_DIR" \
    --all_gpus --frameworks $FRAMEWORKS --models $CLAM_MODELS --nnmil_models $NNMIL_MODELS \
    --dtfd_models $DTFD_MODELS --abmil_models $ABMIL_MODELS --encoders $ENCODERS --tasks $TASKS \
    --n_folds $N_FOLDS --no_wandb
EXIT=$?

# --- auto-resubmit on timeout (preserves N_FOLDS) ---
if [ $EXIT -eq 0 ]; then
    echo "Benchmark completed."
elif [ $EXIT -eq 143 ] || [ $EXIT -eq 137 ]; then
    echo "Time limit — resubmitting ${N_FOLDS}-fold (idempotent resume)..."
    sbatch --parsable "$SELF" "$N_FOLDS" || echo "ERROR: resubmit failed. Run: sbatch $SELF $N_FOLDS"
else
    echo "Benchmark exited $EXIT — non-recoverable. Check logs."
fi
echo "End: $(date)"
exit $EXIT
