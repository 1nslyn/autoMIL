#!/bin/bash
# SLURM: MIL benchmark grid for ANY preprint dataset, 4x H100.
#
# Runs the tile-encoder grid — CLAM{clam_sb,clam_mb,mil} + nnMIL{trans_mil,
# simple_mil} + DTFD{dtfd_mil} + ABMIL{abmil} x 3 encoders x the dataset's
# configured task(s) — reading tasks/rosters/encoders straight from the dataset
# YAML. Writes to <Pathology>/autoMIL/phase2/<dataset>/benchmark_<n>fold.
# Idempotent (finished experiments are skipped); auto-resubmits before the 24h
# wall (SIGUSR1). TITAN is a separate arm — see submit_titan_extract.sh + submit_titan.sh.
#
# Usage:
#   sbatch benchmarks/scripts/submit_benchmark.sh <dataset>          # 5-fold (standard)
#   sbatch benchmarks/scripts/submit_benchmark.sh <dataset> 10       # 10-fold (comparison)
#   e.g.  sbatch benchmarks/scripts/submit_benchmark.sh tcga_lgg
# (legacy env form still works: DATASET=tcga_lgg sbatch benchmarks/scripts/submit_benchmark.sh)
# Run it from the repo root so SLURM_SUBMIT_DIR points at your checkout.

#SBATCH --job-name=mil_bench
#SBATCH --account=rrg-jma
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=0
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/bench_%x_%j.out
#SBATCH --error=logs/bench_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL

set -uo pipefail
DATASET="${1:-${DATASET:-}}"
N_FOLDS="${2:-${N_FOLDS:-5}}"
FRAMEWORKS="${FRAMEWORKS:-clam nnmil dtfd abmil}"
[ -n "$DATASET" ] || { echo "usage: sbatch submit_benchmark.sh <dataset> [n_folds]"; exit 1; }
[[ "$N_FOLDS" =~ ^[0-9]+$ ]] && [ "$N_FOLDS" -ge 2 ] || { echo "ERROR: n_folds must be an integer >= 2 (got '$N_FOLDS')"; exit 1; }

# Repo root = where you ran sbatch (portable across users); fall back to Leo's.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-/home/yinshuol/scratch/autoMIL/autoMIL}"
SELF="$PROJECT_DIR/benchmarks/scripts/submit_benchmark.sh"

# Resubmit-before-timeout: SLURM sends SIGUSR1 300s before the wall (see
# --signal above); we resubmit ourselves (idempotent resume) BEFORE the hard kill.
_resubmitted=0
_resubmit_before_wall() {
    if [ "$_resubmitted" -eq 0 ]; then
        _resubmitted=1
        echo "[signal] Wall limit approaching — resubmitting ${DATASET} ${N_FOLDS}-fold to resume..."
        sbatch --parsable "$SELF" "$DATASET" "$N_FOLDS" && echo "  resubmitted" || echo "  ERROR: resubmit failed"
    fi
}
trap _resubmit_before_wall USR1

echo "================================================"
echo "AutoBench ${DATASET} grid — ${N_FOLDS}-fold, 4x H100"
echo "Job ${SLURM_JOB_ID:-N/A} | $(hostname) | $(date)"
echo "================================================"

module load cuda/12.2 2>/dev/null || true
cd "$PROJECT_DIR" || { echo "ERROR: project dir not found: $PROJECT_DIR"; exit 1; }
mkdir -p logs
source .venv/bin/activate
set -a; source benchmarks/.env; set +a

# Resolve the dataset's data_root (env vars now sourced) -> phase-2 benchmark dir.
DATA_ROOT=$(python -c "from autobench.config import load_dataset_config as L; print(L('${DATASET}').data_root)") \
    || { echo "ERROR: cannot load dataset config '${DATASET}' (check the name + its AUTOBENCH_*_ROOT in benchmarks/.env)"; exit 1; }
[ -n "$DATA_ROOT" ] || { echo "ERROR: empty data_root for ${DATASET}"; exit 1; }
PATHOLOGY_ROOT="$(dirname "$(dirname "$DATA_ROOT")")"
BENCHMARK_DIR="$PATHOLOGY_ROOT/autoMIL/phase2/${DATASET}/benchmark_${N_FOLDS}fold"
echo "Data root:     $DATA_ROOT"
echo "Benchmark dir: $BENCHMARK_DIR"
mkdir -p "$BENCHMARK_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || true

# --- validate config + show the grid the 4 frameworks will generate ---
echo "===== validate config ====="
python - <<PYEOF || { echo "ERROR: config validation failed"; exit 1; }
from autobench.config import load_dataset_config
from autobench.pipeline.config import build_registries, generate_all_experiments, BenchmarkConfig, Framework
_MAP={"clam":Framework.CLAM,"nnmil":Framework.NNMIL,"dtfd":Framework.DTFD,"titan":Framework.TITAN,"abmil":Framework.ABMIL}
ds=load_dataset_config("${DATASET}"); reg=build_registries(ds)
cfg=BenchmarkConfig.from_dataset_config(ds, frameworks=[_MAP[f] for f in "${FRAMEWORKS}".split()], n_folds=${N_FOLDS})
exps=generate_all_experiments(cfg,reg)
print(f"  dataset={ds.name}  tasks={list(ds.tasks.keys())}  encoders={list(ds.encoder_dims.keys())}")
print(f"  nnmil={ds.nnmil_models} dtfd={ds.dtfd_models} abmil={ds.abmil_models} clam={ds.clam_models}")
print(f"  experiments={len(exps)}  fold-trainings={len(exps)*${N_FOLDS}}")
assert len(exps) > 0, "grid is empty — check the dataset YAML's tasks/encoders"
print("  OK")
PYEOF

# --- prep: n-fold splits + tile H5->PT (uses the config's encoders + tasks) ---
echo "===== prep (${N_FOLDS}-fold splits) ====="
python benchmarks/scripts/run_benchmark.py --dataset "$DATASET" --benchmark_dir "$BENCHMARK_DIR" \
    --prep_only --n_folds "$N_FOLDS" || { echo "ERROR: prep failed"; exit 1; }

# --- train (4x H100) — background + wait so the USR1 trap can fire ---
echo "===== train (4x H100) ====="
python benchmarks/scripts/run_benchmark.py --dataset "$DATASET" --benchmark_dir "$BENCHMARK_DIR" \
    --all_gpus --frameworks $FRAMEWORKS --n_folds "$N_FOLDS" --no_wandb &
TRAIN_PID=$!
while true; do
    wait "$TRAIN_PID"; EXIT=$?
    kill -0 "$TRAIN_PID" 2>/dev/null || break
done

if [ $EXIT -eq 0 ]; then
    echo "Benchmark completed."
elif [ $EXIT -eq 143 ] || [ $EXIT -eq 137 ]; then
    echo "Time limit (exit $EXIT) — ensuring a resume job is queued (fallback)..."
    _resubmit_before_wall
else
    echo "Benchmark exited $EXIT — non-recoverable. Check logs."
fi
echo "End: $(date)"
exit $EXIT
