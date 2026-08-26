#!/bin/bash
# SLURM: preprint campaign v3 native baselines — ONE full node, 4x H100.
#
# Runs `campaign_stage.py run-baseline` for every not-yet-registered cell of
# the active roster (tcga_luad, tcga_hnsc, cptac_pdac — reduced 2026-08-23),
# packing 4 workers onto the node's 4 GPUs. Idempotent: registered cells are
# skipped, an interrupted cell retrains on the next pass. Auto-resubmits
# before the 24h wall (SIGUSR1) while unregistered work remains.
# The 10 Gate-1 regime cells (LUAD uni_v2 x 4 tile arms + TITAN, x kras/os)
# are ordered first so every arm/task regime is exercised earliest.
#
# Usage (from the repo root):
#   sbatch benchmarks/scripts/slurm/submit_baseline_campaign.sh

#SBATCH --job-name=bl_campaign
#SBATCH --account=rrg-jma
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=0
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/bl_campaign_%j.out
#SBATCH --error=logs/bl_campaign_%j.err
#SBATCH --mail-type=FAIL

set -uo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-/home/yinshuol/scratch/autoMIL/autoMIL}"
RUNTIME="$PROJECT_DIR/benchmarks/campaigns/preprint_130/runtime"
MANIFEST="$PROJECT_DIR/benchmarks/campaigns/preprint_130/manifest.json"
SELF="$PROJECT_DIR/benchmarks/scripts/slurm/submit_baseline_campaign.sh"
# The active roster's identity lives in ONE committed artifact —
# active_roster.json (cohorts + cell census) — which this launcher executes
# but never decides. Roster changes are edits to that file, reviewable in
# git; the scan below verifies the declaration against the frozen manifest
# and refuses any mismatch. The manifest stays a byte-identical 130-cell
# superset permanently (exporter ports and manifest_sha256 bindings are
# row-indexed and cannot move), so this file is the permanent census
# authority the framework and this launcher validate against — not an
# interim stand-in for a manifest that will one day shrink to match it.
ROSTER="$PROJECT_DIR/benchmarks/campaigns/preprint_130/active_roster.json"
# Registered cell archives are mirrored into project storage after each cell
# finishes: training MUST write into the cell root (the attestation and
# sealed-evidence chain verifies those exact paths, and the runtime lives in
# purge-eligible scratch), so the durable, browsable copy is the mirror.
# The mirror mapping, the sealed/public split, and the hash-verified
# EXPORT_OK marker all live in ONE place — campaign_export.py — and the
# destination root comes from AUTOBENCH_EXPORT_ROOT in benchmarks/.env.
N_GPUS=4

cd "$PROJECT_DIR" || { echo "ERROR: project dir not found: $PROJECT_DIR"; exit 1; }
[ -d "$RUNTIME" ] || { echo "ERROR: runtime not materialized: $RUNTIME"; exit 1; }
[ -f benchmarks/.env ] || { echo "ERROR: benchmarks/.env missing"; exit 1; }
[ -f "$ROSTER" ] || { echo "ERROR: active roster missing: $ROSTER"; exit 1; }
mkdir -p logs/baseline_cells
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv is required on the compute node"; exit 1; }
module load cuda/12.2 2>/dev/null || true
set -a; source benchmarks/.env; set +a
[ -n "${AUTOBENCH_EXPORT_ROOT:-}" ] || { echo "ERROR: AUTOBENCH_EXPORT_ROOT missing from benchmarks/.env"; exit 1; }
[ -d "$AUTOBENCH_EXPORT_ROOT" ] || { echo "ERROR: export root not a directory: $AUTOBENCH_EXPORT_ROOT"; exit 1; }

# Pending = roster cells whose stage state has no registered baseline
# ("baseline" stays null until registration). Gate-1 regime cells first.
# The roster comes from the frozen manifest, not a filesystem glob — every
# manifest cell of the active cohorts must have a materialized state root
# whose recorded cell_id matches, or the scan fails loudly. A quietly
# narrowed queue would end the job with "nothing to do" while cells were
# never run.
list_pending() {
    uv run --frozen --no-sync --package autobench python - \
        "$MANIFEST" "$RUNTIME" "$ROSTER" "${1:-pending}" <<'PYEOF'
import json, sys
from pathlib import Path
manifest, runtime, roster_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
mode = sys.argv[4]
if mode not in ("pending", "registered"):
    sys.exit(f"unknown scan mode {mode!r}")
try:
    roster = json.loads(roster_path.read_text())
    cohorts, expected = list(roster["cohorts"]), int(roster["cells"])
except (OSError, ValueError, KeyError) as exc:
    sys.exit(f"cannot read active roster {roster_path}: {exc!r}")
if not cohorts:
    sys.exit(f"{roster_path}: empty cohort list")
records = json.loads(manifest.read_text())["cells"]
unknown = sorted(set(cohorts) - {c["dataset"] for c in records})
if unknown:
    sys.exit(f"roster cohorts absent from the manifest: {unknown}")
cells = sorted(c["cell_id"] for c in records if c["dataset"] in cohorts)
if len(cells) != expected:
    sys.exit(f"roster census mismatch: {len(cells)} manifest cells for "
             f"{sorted(cohorts)}, declared cells={expected}")
print(f"roster: {sorted(cohorts)} = {len(cells)} cells ({roster_path.name})",
      file=sys.stderr)
canary_first, rest, registered = [], [], []
for cell in cells:
    state_path = runtime / cell / "campaign_state.json"
    if not state_path.is_file():
        sys.exit(f"{cell}: no materialized state root under {runtime}")
    state = json.loads(state_path.read_text())
    if state.get("cell_id") != cell:
        sys.exit(f"{cell}: state carries cell_id {state.get('cell_id')!r}")
    baseline = state.get("baseline")
    if baseline is not None:
        # Same shape contract as run_native_baseline: a present-but-non-dict
        # baseline is an invalid registration, never "already done".
        if not isinstance(baseline, dict):
            sys.exit(f"{cell}: registered baseline state is invalid "
                     f"({type(baseline).__name__})")
        registered.append(cell)
        continue
    encoder = cell.split("__")[2]
    (canary_first if cell.startswith("tcga_luad__") and encoder in ("uni_v2", "titan")
     else rest).append(cell)
print("\n".join(registered if mode == "registered" else canary_first + rest))
PYEOF
}
PENDING=$(list_pending) || { echo "ERROR: pending-cell scan failed"; exit 1; }
echo "$PENDING" | grep -c . | xargs echo "pending cells:"

QUEUE_FILE=$(mktemp)
FAIL_FILE=$(mktemp)
echo "$PENDING" > "$QUEUE_FILE"
trap 'rm -f "$QUEUE_FILE" "$FAIL_FILE"' EXIT

# Resubmit-before-wall: SLURM sends USR1 300s before the wall (see --signal).
# Resubmit UNCONDITIONALLY — nothing else runs in the signal path. The trap
# only fires when work filled the full 24h, so work almost surely remains; in
# the rare drained case the continuation job scans, finds nothing, and exits
# cleanly. Running a scan here instead would let one filesystem stall at the
# wall eat the resubmit and strand the campaign. sbatch is retried because a
# transient scheduler error must not strand it either.
_resubmitted=0
_resubmit_before_wall() {
    if [ "$_resubmitted" -eq 0 ]; then
        _resubmitted=1
        echo "[signal] Wall approaching — resubmitting to resume..."
        local attempt
        for attempt in 1 2 3; do
            if sbatch --parsable "$SELF"; then
                echo "  resubmitted (attempt $attempt)"
                return
            fi
            echo "  sbatch failed (attempt $attempt)"
            sleep 20
        done
        echo "  ERROR: resubmit failed after 3 attempts — resume manually"
    fi
}
trap _resubmit_before_wall USR1

pop_cell() {
    (
        flock 9
        head -n 1 "$QUEUE_FILE"
        sed -i '1d' "$QUEUE_FILE"
    ) 9>>"$QUEUE_FILE.lock"
}

# Mirror one registered cell into project storage via campaign_export.py
# (hash-verified, sealed/public split, EXPORT_OK marker, per-cell lock).
# Export failure is loud (FAIL_FILE -> nonzero job exit -> FAIL mail) but
# does not undo the local registration; export is idempotent, so the next
# pass repairs it.
export_cell() {
    local cell="$1"
    if uv run --frozen --no-sync --package autobench \
        python benchmarks/scripts/campaign_export.py --cell "$cell"; then
        return 0
    fi
    echo "$cell export-failed" >> "$FAIL_FILE"
    return 1
}

# Catch-up: mirror every already-registered roster cell (covers cells
# finished by earlier job generations that ran without the export step) and
# seed the campaign identity artifacts. Failure is recorded in FAIL_FILE so
# the job cannot end claiming "mirrored" after a pass that never ran.
export_registered() {
    if ! uv run --frozen --no-sync --package autobench \
        python benchmarks/scripts/campaign_export.py --all-registered; then
        echo "WARNING: catch-up export reported failures"
        echo "catch-up-export export-failed" >> "$FAIL_FILE"
        return 1
    fi
}

worker() {
    local gpu="$1" cell rc
    while :; do
        cell=$(pop_cell)
        [ -n "$cell" ] || break
        echo "[gpu$gpu] $(date +%H:%M:%S) start $cell"
        uv run --frozen --no-sync --package autobench \
            python benchmarks/scripts/campaign_stage.py run-baseline \
            --cell-root "$RUNTIME/$cell" --gpu "$gpu" \
            > "logs/baseline_cells/${cell}.log" 2>&1
        rc=$?
        echo "[gpu$gpu] $(date +%H:%M:%S) done  $cell rc=$rc"
        if [ "$rc" -eq 0 ]; then
            export_cell "$cell" && echo "[gpu$gpu] exported $cell"
        else
            echo "$cell rc=$rc" >> "$FAIL_FILE"
        fi
    done
}

echo "================================================"
echo "preprint campaign v3 baselines — roster: $ROSTER"
echo "Job ${SLURM_JOB_ID:-N/A} | $(hostname) | $(date)"
echo "================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || true

# Catch-up mirror first, so cells registered by earlier job generations
# (before the export step existed) reach version3 even when nothing is
# pending any more.
export_registered
if [ -z "$PENDING" ]; then
    if [ -s "$FAIL_FILE" ]; then
        echo "Nothing pending, but exports failed:"; sed 's/^/  /' "$FAIL_FILE"
        exit 1
    fi
    echo "All roster baselines are registered and mirrored — nothing to do."
    exit 0
fi

for gpu in $(seq 0 $((N_GPUS - 1))); do
    worker "$gpu" &
done
while [ -n "$(jobs -pr)" ]; do
    wait -n || true
done

echo "---"
if [ -s "$FAIL_FILE" ]; then
    echo "FAILED cells (see logs/baseline_cells/<cell>.log):"
    sed 's/^/  /' "$FAIL_FILE"
    exit 1
fi
echo "All attempted cells registered cleanly. $(date)"
