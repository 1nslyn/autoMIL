#!/bin/bash
# SLURM job: TCGA-LUAD GOLDMARK-parity VERIFICATION (branch feat/goldmark-parity).
#
# Purpose: prove whether our matched-task AUCs read lower than GOLDMARK because
# of a *protocol* difference (we report disjoint held-out TEST AUC; GOLDMARK's
# internal-CV number selects the best epoch on, AND reports, the SAME held-out
# fold — val_split_value==test_split_value=='test') rather than a pipeline
# deficiency. We run CLAM on LUAD in BOTH modes, fold-matched (N=5), and look at
# the delta:
#   - default : our conservative 3-way ~70/10/20, report held-out TEST AUC.
#   - parity  : GOLDMARK 2-way 70/30, holdout fold = val(selection)==test(report).
# If parity rises ~+0.02..+0.06 toward GOLDMARK's frozen-encoder numbers while
# default sits lower, the gap is protocol, not pipeline. Virchow2 is the clean
# control (exact encoder match on both sides).
#
# Runs the BRANCH code (feat/goldmark-parity) via PYTHONPATH, using the main
# checkout's .venv for dependencies. The live LGG/COAD jobs (main checkout) are
# untouched. ALL output to /scratch — NOTHING written to the shared dir.
# Features are REUSED from the shared LUAD CLAM PT store via symlink, so the
# forced H5->PT convert no-ops (no re-extraction, no shared writes).
#
# Grid: framework {clam} x models {clam_sb, clam_mb} x encoders
#       {hoptimus1, uni_v2, virchow2} x tasks {egfr, kras} x 5 folds x 2 modes.
#
# Idempotent: completed folds skip, so the 12h auto-resubmit resumes.
#
# Usage:  sbatch benchmarks/scripts/submit_goldmark_parity_luad.sh

#SBATCH --job-name=gmparity_luad
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
DATASET="tcga_luad"
FRAMEWORKS="clam"
CLAM_MODELS="clam_sb clam_mb"
ENCODERS="hoptimus1 uni_v2 virchow2"
TASKS="egfr kras"
# Fold-matched to GOLDMARK's 5 stratified splits (and cheaper than our 10-fold).
N_FOLDS=5
HOLDOUT_FRAC=0.30

# Branch worktree (the code under test) + main checkout (.venv + .env + features).
WORKTREE="/scratch/yinshuol/autoMIL/wt-goldmark-parity"
MAIN_CHECKOUT="/scratch/yinshuol/autoMIL/autoMIL"
SELF="$WORKTREE/benchmarks/scripts/submit_goldmark_parity_luad.sh"

# Shared LUAD assets (read-only to us) — reused, never written.
LUAD_SHARED="/home/yinshuol/projects/rrg-jma/shared/Pathology/TCGA/TCGA-LUAD"
MAPPING_CSV="$LUAD_SHARED/normalized_manifest.csv"
H5_BASE="$LUAD_SHARED/trident_output/20x_224px_0px_overlap"   # features_<enc>/*.h5
PT_SHARED="$LUAD_SHARED/benchmark/features"                   # <enc>/pt_files/*.pt

# Scratch output root (NOT the shared dir).
OUTROOT="/scratch/yinshuol/autoMIL/goldmark_parity_luad"

# ==================== JOB INFO ====================
echo "================================================"
echo "AutoBench LUAD — GOLDMARK-parity verification (branch feat/goldmark-parity)"
echo "================================================"
echo "Job ID:        ${SLURM_JOB_ID:-N/A}"
echo "Dataset:       $DATASET   Tasks: $TASKS"
echo "Framework:     $FRAMEWORKS ($CLAM_MODELS)   Encoders: $ENCODERS   Folds: $N_FOLDS"
echo "Modes:         default (3-way test)  +  parity (2-way ${HOLDOUT_FRAC} val==test)"
echo "Worktree:      $WORKTREE"
echo "Out root:      $OUTROOT  (scratch only)"
echo "Node:          $(hostname)   Start: $(date)"
echo "================================================"

# ==================== ENVIRONMENT ====================
module load cuda/12.2 2>/dev/null || true
cd "$WORKTREE" || { echo "ERROR: worktree not found"; exit 1; }
source "$MAIN_CHECKOUT/.venv/bin/activate"
# Run the BRANCH code, not the editable-installed main checkout.
export PYTHONPATH="$WORKTREE/benchmarks/src:$WORKTREE/src${PYTHONPATH:+:$PYTHONPATH}"
# .env (HF/WANDB tokens + AUTOBENCH_TCGA_LUAD_ROOT) lives only in the main checkout.
set -a; source "$MAIN_CHECKOUT/benchmarks/.env"; set +a
echo "Python:     $(which python)"
echo "autobench:  $(python -c 'import autobench, os; print(os.path.dirname(autobench.__file__))')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || true

# Sanity: the branch code must expose --goldmark_parity, else we'd silently
# run main's code and produce a meaningless (non-parity) result.
if ! python benchmarks/scripts/run_benchmark.py --help 2>/dev/null | grep -q -- "--goldmark_parity"; then
    echo "ERROR: branch code not on PYTHONPATH (--goldmark_parity missing). Aborting."
    exit 1
fi

# ==================== FEATURE REUSE (symlink; no re-extraction) ==============
# Symlink each mode's benchmark_dir/features/<enc> -> shared CLAM PT store, so
# convert_h5_to_pt finds every {sid}.pt already present and writes nothing.
link_features () {
    local bench_dir="$1"
    mkdir -p "$bench_dir/features"
    for e in $ENCODERS; do
        ln -sfn "$PT_SHARED/$e" "$bench_dir/features/$e"
    done
}

# ==================== RUN ONE MODE ====================
run_mode () {
    local mode="$1"; shift
    local extra_flags=("$@")
    local bench_dir="$OUTROOT/$mode/benchmark"
    echo ""
    echo "################ MODE: $mode  (flags: ${extra_flags[*]:-none}) ################"
    echo "benchmark_dir: $bench_dir"
    mkdir -p "$bench_dir"
    link_features "$bench_dir"

    echo "---- prep ($mode) ----"
    python benchmarks/scripts/run_benchmark.py \
        --dataset "$DATASET" \
        --benchmark_dir "$bench_dir" \
        --mapping_csv "$MAPPING_CSV" \
        --features_base_dir "$H5_BASE" \
        --prep_only \
        --encoders $ENCODERS \
        --tasks $TASKS \
        --n_folds $N_FOLDS \
        "${extra_flags[@]}"
    local prep_exit=$?
    if [ $prep_exit -ne 0 ]; then echo "ERROR: prep failed ($mode, exit $prep_exit)"; return $prep_exit; fi

    echo "---- benchmark ($mode) ----"
    python benchmarks/scripts/run_benchmark.py \
        --dataset "$DATASET" \
        --benchmark_dir "$bench_dir" \
        --mapping_csv "$MAPPING_CSV" \
        --features_base_dir "$H5_BASE" \
        --gpu 0 \
        --frameworks $FRAMEWORKS \
        --models $CLAM_MODELS \
        --encoders $ENCODERS \
        --tasks $TASKS \
        --n_folds $N_FOLDS \
        --no_wandb \
        "${extra_flags[@]}"
    return $?
}

# Parity first (the new measurement), then fold-matched default for the delta.
run_mode parity --goldmark_parity --holdout_frac $HOLDOUT_FRAC
PARITY_EXIT=$?
run_mode default
DEFAULT_EXIT=$?
EXIT_CODE=$(( PARITY_EXIT != 0 ? PARITY_EXIT : DEFAULT_EXIT ))

# ==================== AUTO-CONTINUATION ====================
echo ""
echo "================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "GOLDMARK-parity verification completed (parity + default)."
elif [ $EXIT_CODE -eq 143 ] || [ $EXIT_CODE -eq 137 ]; then
    echo "Time limit reached (exit $EXIT_CODE) — auto-resubmitting (idempotent resume)..."
    NEW_JOB_ID=$(sbatch --parsable "$SELF")
    [ $? -eq 0 ] && echo "New job submitted: $NEW_JOB_ID" || echo "ERROR: resubmit failed. Run: sbatch $SELF"
else
    echo "Exited with code $EXIT_CODE — non-recoverable. Check logs."
fi
echo "End time: $(date)"
echo "================================================"
exit $EXIT_CODE
