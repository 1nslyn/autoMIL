#!/bin/bash
# SLURM job: GOLDMARK EXACT apples-to-apples (branch feat/goldmark-parity).
#
# Goal (Leo): prove our model (encoder + MIL) beats GOLDMARK under *identical*
# training logic. We run TWO arms, both on GOLDMARK's exact comparison SPLIT
# (5x StratifiedShuffleSplit, holdout_frac=0.33, val==test), so both are
# directly comparable to GOLDMARK's published mean-per-split AUROC:
#   - goldmark : --goldmark_recipe  (AdamW lr1e-4 wd1e-4, ReduceLROnPlateau,
#                CE, 120 epochs, NO early stop, best-val-AUC @ cadence
#                {2,5,10,20,50,80,120}; nnMIL keeps all patches, selects on AUC).
#                clam_sb runs instance-loss-OFF as the GMA proxy.
#   - our      : our native recipe (Adam lr2e-4, val-loss early stop) on the
#                same split — our model's best case.
# Models: clam_sb (GMA-proxy) + clam_mb + nnMIL simple_mil. Encoders:
# hoptimus1, uni_v2, virchow2. Compare to /scratch/.../goldmark-portal/
# goldmark_authoritative.csv (mean_per_split column).
#
# Runs BRANCH code via PYTHONPATH, main checkout's .venv + .env. ALL output to
# /scratch — NOTHING to the shared dir. Features REUSED from the shared PT/H5
# stores (symlink for CLAM PT; nnMIL reads shared H5 directly). Idempotent:
# completed folds skip, so the 12h auto-resubmit resumes.
#
# Usage:
#   sbatch --export=ALL,COHORT=luad benchmarks/scripts/submit_goldmark_exact.sh
#   sbatch --export=ALL,COHORT=lgg  benchmarks/scripts/submit_goldmark_exact.sh
#   sbatch --export=ALL,COHORT=coad benchmarks/scripts/submit_goldmark_exact.sh
#   # SMOKE (1 task/1 enc/1 fold, goldmark arm only — verify before full matrix):
#   sbatch --export=ALL,COHORT=luad,SMOKE=1 benchmarks/scripts/submit_goldmark_exact.sh
#
# Tunable via --export env: COHORT, MODES, NFOLDS, ENCODERS, TASKS,
#   CLAM_MODELS, NNMIL_MODELS, SMOKE.

#SBATCH --job-name=gmexact
#SBATCH --account=rrg-jma
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=96G
#SBATCH --exclude=fc10512
#SBATCH --signal=B:TERM@120
#SBATCH --output=/scratch/yinshuol/autoMIL/logs/bench_%x_%j.out
#SBATCH --error=/scratch/yinshuol/autoMIL/logs/bench_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leo.yin@mail.utoronto.ca

set -uo pipefail

# ==================== CONFIG ====================
COHORT="${COHORT:-luad}"
SMOKE="${SMOKE:-0}"
HOLDOUT_FRAC=0.33                      # GOLDMARK code default (test_frac=0.33)

case "$COHORT" in
    luad) DATASET="tcga_luad"; DEF_TASKS="egfr kras"; CO_UP="LUAD" ;;
    lgg)  DATASET="tcga_lgg";  DEF_TASKS="idh1";      CO_UP="LGG"  ;;
    coad) DATASET="tcga_coad"; DEF_TASKS="braf";      CO_UP="COAD" ;;
    *) echo "ERROR: unknown COHORT='$COHORT' (luad|lgg|coad)"; exit 1 ;;
esac

FRAMEWORKS="clam nnmil"
CLAM_MODELS="${CLAM_MODELS:-clam_sb clam_mb}"
NNMIL_MODELS="${NNMIL_MODELS:-simple_mil}"
ENCODERS="${ENCODERS:-hoptimus1 uni_v2 virchow2}"
TASKS="${TASKS:-$DEF_TASKS}"
N_FOLDS="${NFOLDS:-5}"
MODES="${MODES:-goldmark our}"

if [ "$SMOKE" = "1" ]; then
    # Minimal end-to-end verification of the new goldmark_recipe path: 1 task,
    # 1 encoder, 1 fold, goldmark arm only — exercises clam_sb (inst-off branch),
    # clam_mb (inst-on branch) and nnMIL recipe on real features.
    TASKS="$(echo $TASKS | awk '{print $1}')"
    ENCODERS="hoptimus1"
    N_FOLDS=1
    MODES="goldmark"
fi

# Branch worktree (code under test) + main checkout (.venv + .env + features).
WORKTREE="/scratch/yinshuol/autoMIL/wt-goldmark-parity"
MAIN_CHECKOUT="/scratch/yinshuol/autoMIL/autoMIL"
SELF="$WORKTREE/benchmarks/scripts/submit_goldmark_exact.sh"

# Shared cohort assets (read-only to us) — reused, never written.
SHARED="/home/yinshuol/projects/rrg-jma/shared/Pathology/TCGA/TCGA-${CO_UP}"
MAPPING_CSV="$SHARED/normalized_manifest.csv"
H5_BASE="$SHARED/trident_output/20x_224px_0px_overlap"   # features_<enc>/*.h5
PT_SHARED="$SHARED/benchmark/features"                   # <enc>/pt_files/*.pt

OUTROOT="/scratch/yinshuol/autoMIL/goldmark_exact/$COHORT"

# ==================== WALL-TIME AUTO-RESUBMIT ====================
# A clean 12h TIMEOUT sends SIGTERM to THIS batch shell ~120s before the wall
# (via #SBATCH --signal=B:TERM@120). Without a trap, bash dies before the
# end-of-script resubmit runs, so the matrix stops mid-run and never continues
# (observed 2026-06-18: every job TIMEOUT at 12h with NO resume). The trap
# resubmits the same scoped job (idempotent — completed folds skip) before the
# hard kill, so the matrix self-continues across walls until it is complete.
resubmit_self() {
    sbatch --parsable --export=ALL,COHORT=$COHORT,SMOKE=$SMOKE,MODES="$MODES",NFOLDS=$N_FOLDS,ENCODERS="$ENCODERS",TASKS="$TASKS" "$SELF"
}
WALL_RESUBMITTED=0
on_sigterm() {
    if [ "$WALL_RESUBMITTED" -eq 0 ]; then
        WALL_RESUBMITTED=1
        echo ""
        echo ">>> SIGTERM (wall approaching) — auto-resubmitting for idempotent resume..."
        if NEW=$(resubmit_self); then echo ">>> Resubmitted as: $NEW"; else echo ">>> ERROR: wall resubmit failed (run: sbatch $SELF)"; fi
    fi
    exit 143
}
trap on_sigterm SIGTERM

# ==================== JOB INFO ====================
echo "================================================"
echo "GOLDMARK EXACT apples-to-apples — $DATASET (branch feat/goldmark-parity)"
echo "================================================"
echo "Job ID:     ${SLURM_JOB_ID:-N/A}   SMOKE=$SMOKE"
echo "Tasks:      $TASKS   Encoders: $ENCODERS   Folds: $N_FOLDS"
echo "Frameworks: $FRAMEWORKS   CLAM: $CLAM_MODELS   nnMIL: $NNMIL_MODELS"
echo "Modes:      $MODES   (split: parity holdout=$HOLDOUT_FRAC val==test)"
echo "Out root:   $OUTROOT  (scratch only)"
echo "Node:       $(hostname)   Start: $(date)"
echo "================================================"

# ==================== ENVIRONMENT ====================
module load cuda/12.2 2>/dev/null || true
cd "$WORKTREE" || { echo "ERROR: worktree not found"; exit 1; }
source "$MAIN_CHECKOUT/.venv/bin/activate"
export PYTHONPATH="$WORKTREE/benchmarks/src:$WORKTREE/src${PYTHONPATH:+:$PYTHONPATH}"
set -a; source "$MAIN_CHECKOUT/benchmarks/.env"; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "Python:     $(which python)"
echo "autobench:  $(python -c 'import autobench, os; print(os.path.dirname(autobench.__file__))')"

# Sanity: branch code must expose BOTH flags, else we'd silently run main's code.
HELP="$(python benchmarks/scripts/run_benchmark.py --help 2>/dev/null)"
for flag in -- "--goldmark_parity" "--goldmark_recipe"; do :; done
if ! echo "$HELP" | grep -q -- "--goldmark_recipe"; then
    echo "ERROR: branch code lacks --goldmark_recipe (wrong PYTHONPATH). Aborting."; exit 1
fi
if ! echo "$HELP" | grep -q -- "--goldmark_parity"; then
    echo "ERROR: branch code lacks --goldmark_parity. Aborting."; exit 1
fi

# ==================== GPU PREFLIGHT (capped requeue; never infinite loop) ======
gpu_mem_status () {
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
        --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
}
MIN_FREE_MIB=16000
MAX_PREFLIGHT_TRIES=4
TRY="${GMEXACT_PREFLIGHT_TRY:-1}"
echo "GPU at start (preflight try $TRY/$MAX_PREFLIGHT_TRIES): $(gpu_mem_status)"
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "${FREE_MIB:-}" ] && [ "${FREE_MIB:-0}" -lt "$MIN_FREE_MIB" ] 2>/dev/null; then
    if [ "$TRY" -ge "$MAX_PREFLIGHT_TRIES" ]; then
        echo "ERROR: only ${FREE_MIB} MiB free after $TRY tries — giving up (NOT looping)."; exit 1
    fi
    echo "WARN: only ${FREE_MIB} MiB free (< ${MIN_FREE_MIB}) — requeue (attempt $((TRY+1))/$MAX_PREFLIGHT_TRIES)..."
    NEW_JOB_ID=$(sbatch --parsable --export=ALL,GMEXACT_PREFLIGHT_TRY=$((TRY+1)),COHORT=$COHORT,SMOKE=$SMOKE,MODES="$MODES",NFOLDS=$N_FOLDS,ENCODERS="$ENCODERS",TASKS="$TASKS" "$SELF")
    [ $? -eq 0 ] && echo "Requeued as: $NEW_JOB_ID" || echo "ERROR: requeue failed. Run: sbatch $SELF"
    exit 0
fi
echo "GPU preflight OK: ${FREE_MIB:-unknown} MiB free."

# ==================== FEATURE REUSE (symlink CLAM PT; no re-extraction) ========
link_features () {
    local bench_dir="$1"
    mkdir -p "$bench_dir/features"
    for e in $ENCODERS; do
        [ -d "$PT_SHARED/$e" ] && ln -sfn "$PT_SHARED/$e" "$bench_dir/features/$e"
    done
}

# ==================== RUN ONE MODE ====================
run_mode () {
    local mode="$1"
    local extra_flags=()
    case "$mode" in
        goldmark) extra_flags=(--goldmark_parity --holdout_frac "$HOLDOUT_FRAC" --goldmark_recipe) ;;
        our)      extra_flags=(--goldmark_parity --holdout_frac "$HOLDOUT_FRAC") ;;
        *) echo "ERROR: unknown mode '$mode'"; return 2 ;;
    esac
    local bench_dir="$OUTROOT/$mode/benchmark"
    echo ""
    echo "################ MODE: $mode  (flags: ${extra_flags[*]}) ################"
    echo "benchmark_dir: $bench_dir"
    mkdir -p "$bench_dir"
    link_features "$bench_dir"

    echo "---- prep ($mode): splits + dataset_csv + CLAM PT ----"
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

    echo "---- benchmark ($mode): clam {$CLAM_MODELS} + nnmil {$NNMIL_MODELS} ----  GPU: $(gpu_mem_status)"
    # Run in the BACKGROUND and `wait`: bash defers a trap until the current
    # FOREGROUND command returns, so a foreground python here would swallow the
    # wall SIGTERM until the (hours-long) run ends — the trap would never fire
    # before SIGKILL. `wait` is interruptible, so on_sigterm runs immediately.
    python benchmarks/scripts/run_benchmark.py \
        --dataset "$DATASET" \
        --benchmark_dir "$bench_dir" \
        --mapping_csv "$MAPPING_CSV" \
        --features_base_dir "$H5_BASE" \
        --all_gpus \
        --frameworks $FRAMEWORKS \
        --models $CLAM_MODELS \
        --nnmil_models $NNMIL_MODELS \
        --encoders $ENCODERS \
        --tasks $TASKS \
        --n_folds $N_FOLDS \
        --no_wandb \
        "${extra_flags[@]}" &
    BENCH_PID=$!
    wait "$BENCH_PID"
    return $?
}

EXIT_CODE=0
for m in $MODES; do
    run_mode "$m"
    rc=$?
    [ $rc -ne 0 ] && EXIT_CODE=$rc
done

# ==================== AUTO-CONTINUATION ====================
echo ""
echo "================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "GOLDMARK-exact run completed ($COHORT, modes: $MODES)."
elif [ $EXIT_CODE -eq 143 ] || [ $EXIT_CODE -eq 137 ]; then
    if [ "$WALL_RESUBMITTED" -eq 0 ]; then
        echo "Time limit reached (exit $EXIT_CODE) — auto-resubmitting (idempotent resume)..."
        NEW_JOB_ID=$(resubmit_self)
        [ -n "$NEW_JOB_ID" ] && echo "New job submitted: $NEW_JOB_ID" || echo "ERROR: resubmit failed."
    else
        echo "Wall resubmit already issued by SIGTERM trap; not resubmitting again."
    fi
else
    echo "Exited with code $EXIT_CODE — non-recoverable. Check logs."
fi
echo "End time: $(date)"
echo "================================================"
exit $EXIT_CODE
