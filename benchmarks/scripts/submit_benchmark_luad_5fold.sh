#!/bin/bash
# SLURM job: MULTI-GPU LUAD phase-2 benchmark — 5-fold CV, full tile-MIL grid.
#
# Grid: frameworks {clam, nnmil, dtfd, abmil}
#       CLAM {clam_sb, clam_mb, mil} | nnMIL {trans_mil, simple_mil}
#       DTFD {dtfd_mil} | ABMIL {abmil}
#       x encoders {hoptimus1, uni_v2, virchow2} x tasks {egfr, kras}
#       = 7 model-arms x 3 enc x 2 task = 42 experiments x 5 folds = 210 fold-trainings.
#   NB: TITAN is a SEPARATE arm (submit_titan_luad_5fold.sh) — native 512px/conch recipe.
#
# Phase-2 seam: writes to <Pathology>/autoMIL/phase2/tcga_luad/benchmark, isolated
# from phase-1's 10-fold benchmark/. Shared read-only H5 tile features under
# features_base_dir (20x_224px_0px_overlap) are reused (nnMIL/DTFD/ABMIL read H5
# directly; CLAM converts to .pt inside benchmark_dir).
#
# Idempotent: completed experiments/folds are skipped. The 24h time-limit
# auto-resubmit below uses $SELF so the full hardcoded config is preserved
# (unlike submit_benchmark.sh, whose resubmit drops env overrides).
#
# Usage:  sbatch benchmarks/scripts/submit_benchmark_luad_5fold.sh

#SBATCH --job-name=luad5f_bench
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

# ==================== CONFIG (hardcoded — preserved across resubmit) ====================
DATASET="tcga_luad"
FRAMEWORKS="clam nnmil dtfd abmil"
ENCODERS="hoptimus1 uni_v2 virchow2"
TASKS="egfr kras"
CLAM_MODELS="clam_sb clam_mb mil"
NNMIL_MODELS="trans_mil simple_mil"
DTFD_MODELS="dtfd_mil"
ABMIL_MODELS="abmil"
# Phase-2 uses 5-fold patient-stratified CV (phase-1 was 10-fold). Explicit here
# so a resubmit re-runs the SAME grid; the CLI default is also 5 post-2026-07.
N_FOLDS=5
PROJECT_DIR="/home/yinshuol/scratch/autoMIL/autoMIL"
SELF="$PROJECT_DIR/benchmarks/scripts/submit_benchmark_luad_5fold.sh"

# ==================== JOB INFO ====================
echo "================================================"
echo "AutoBench LUAD — MULTI-GPU phase-2 (5-fold)"
echo "Job ID:   ${SLURM_JOB_ID:-N/A} | Node: $(hostname)"
echo "Frameworks: $FRAMEWORKS | Folds: $N_FOLDS"
echo "  CLAM: $CLAM_MODELS | nnMIL: $NNMIL_MODELS | DTFD: $DTFD_MODELS | ABMIL: $ABMIL_MODELS"
echo "Encoders: $ENCODERS | Tasks: $TASKS"
echo "Start:    $(date)"
echo "================================================"

# ==================== ENVIRONMENT ====================
module load cuda/12.2 2>/dev/null || true
cd "$PROJECT_DIR" || { echo "ERROR: Project dir not found: $PROJECT_DIR"; exit 1; }
source .venv/bin/activate
set -a; source benchmarks/.env; set +a

if [ -z "${AUTOBENCH_TCGA_LUAD_ROOT:-}" ]; then
    echo "ERROR: AUTOBENCH_TCGA_LUAD_ROOT not set (check benchmarks/.env)"; exit 1
fi
# Phase-2 benchmark dir = <Pathology>/autoMIL/phase2/tcga_luad/benchmark, where
# <Pathology> is two levels up from the LUAD root (.../Pathology/TCGA/TCGA-LUAD).
PATHOLOGY_ROOT="$(dirname "$(dirname "$AUTOBENCH_TCGA_LUAD_ROOT")")"
BENCHMARK_DIR="$PATHOLOGY_ROOT/autoMIL/phase2/tcga_luad/benchmark"
echo "Benchmark dir (phase-2): $BENCHMARK_DIR"
mkdir -p "$BENCHMARK_DIR"

echo "Python:   $(which python)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || true

# ==================== VALIDATION (real _FRAMEWORK_MAP; assert 42 experiments) ====================
echo ""
echo "================ Validating config + experiment count ================"
python - <<PYEOF || { echo "ERROR: config validation failed"; exit 1; }
from autobench.config import load_dataset_config
from autobench.pipeline.config import (
    build_registries, generate_all_experiments, BenchmarkConfig, Framework,
)
_MAP = {"clam": Framework.CLAM, "nnmil": Framework.NNMIL,
        "dtfd": Framework.DTFD, "titan": Framework.TITAN, "abmil": Framework.ABMIL}
ds = load_dataset_config("${DATASET}")
reg = build_registries(ds)
fw = [_MAP[f] for f in "${FRAMEWORKS}".split()]
cfg = BenchmarkConfig.from_dataset_config(
    ds, frameworks=fw,
    encoder_keys="${ENCODERS}".split(), tasks="${TASKS}".split(),
    model_types="${CLAM_MODELS}".split(),
    nnmil_model_types="${NNMIL_MODELS}".split(),
    dtfd_model_types="${DTFD_MODELS}".split(),
    abmil_model_types="${ABMIL_MODELS}".split(),
    n_folds=${N_FOLDS},
)
exps = generate_all_experiments(cfg, reg)
print(f"  experiments={len(exps)}  fold-trainings={len(exps) * ${N_FOLDS}}  (expect 42 / 210)")
assert len(exps) == 42, f"expected 42 experiments, got {len(exps)}"
print("  OK")
PYEOF

# ==================== DATA PREP (5-fold splits + tile H5->PT) ====================
echo ""
echo "================ Phase 1: Data Preparation ================"
python benchmarks/scripts/run_benchmark.py \
    --dataset "$DATASET" --benchmark_dir "$BENCHMARK_DIR" --prep_only \
    --encoders $ENCODERS --tasks $TASKS --n_folds $N_FOLDS
PREP_EXIT=$?
if [ $PREP_EXIT -ne 0 ]; then echo "ERROR: prep failed (exit $PREP_EXIT)"; exit $PREP_EXIT; fi

# ==================== BENCHMARK TRAINING (4x H100) ====================
echo ""
echo "================ Phase 2: Benchmark Training (4x H100) ================"
CMD=(python benchmarks/scripts/run_benchmark.py
    --dataset "$DATASET" --benchmark_dir "$BENCHMARK_DIR" --all_gpus
    --frameworks $FRAMEWORKS
    --models $CLAM_MODELS --nnmil_models $NNMIL_MODELS
    --dtfd_models $DTFD_MODELS --abmil_models $ABMIL_MODELS
    --encoders $ENCODERS --tasks $TASKS --n_folds $N_FOLDS --no_wandb
)
echo "Command: ${CMD[*]}"
echo ""
"${CMD[@]}"
EXIT_CODE=$?

# ==================== AUTO-CONTINUATION ($SELF preserves config) ====================
echo ""
echo "================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Benchmark completed successfully!"
elif [ $EXIT_CODE -eq 143 ] || [ $EXIT_CODE -eq 137 ]; then
    echo "Time limit reached (exit $EXIT_CODE) — auto-resubmitting (idempotent resume)..."
    NEW_JOB_ID=$(sbatch --parsable "$SELF")
    if [ $? -eq 0 ]; then echo "New job: $NEW_JOB_ID"; else echo "ERROR: resubmit failed. Run: sbatch $SELF"; fi
else
    echo "Benchmark exited $EXIT_CODE — non-recoverable. Check logs."
fi
echo "End: $(date)"
echo "================================================"
exit $EXIT_CODE
