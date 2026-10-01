#!/bin/bash
# Chain driver for the preprint discovery campaign on a workstation without
# SLURM: run the cells of one runtime set one after another on this host's
# GPUs.
#
# Each cell runs the same job a fir node runs (slurm/submit_discovery_campaign.sh),
# with the allocation passed in from here: a run id, the GPU indexes, and a
# wall end this driver sets itself (--wall-hours). The driver stands in for
# SLURM's chaining: after a clean cell it checks the weekly usage window,
# scans the set and starts the next cell. It stops at the first failed cell
# (the job has written logs/discovery_cells/<set>/FAILED.tsv), when the
# weekly window reaches 85%, or when no cell is left.
#
# Usage, from anywhere, inside a tmux server of its own so it survives SSH
# drops:
#   tmux -L disc_chain new -s chain
#   <checkout>/benchmarks/scripts/run_discovery_chain.sh --runtime runtime-aihub --gpus 0,1,2
#
# Options:
#   --runtime NAME   the cell-root set under the campaign dir (required); its
#                    GPU type is the one reproduction_policy.json declares for it
#   --gpus LIST      the GPU indexes the chain may use, comma separated (required)
#   --wall-hours H   each cell's wall in whole hours, at least 12 (default 48);
#                    the session ends early enough to finish inside it
#   --dry-run        print the scan and the pick order; run nothing
#
# One chain per set: the driver holds logs/discovery_cells/<set>/.chain.lock,
# and the running job inherits it. Each job runs in a session of its own, so
# stopping the driver never interrupts a cell: after Ctrl-C the driver waits
# for the running cell to end and then exits without starting another.

set -uo pipefail
SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export DISC_PROJECT_DIR
DISC_PROJECT_DIR=$(cd "$SELF_DIR/../.." && pwd)
JOB_SCRIPT="$SELF_DIR/slurm/submit_discovery_campaign.sh"

RUNTIME_NAME=""; GPUS=""; WALL_HOURS=48; DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --runtime) RUNTIME_NAME="${2:-}"; shift ;;
        --gpus) GPUS="${2:-}"; shift ;;
        --wall-hours) WALL_HOURS="${2:-}"; shift ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help) sed -n '2,/^$/p' "$0"; exit 0 ;;
        *) echo "unknown option: $1"; exit 2 ;;
    esac
    shift
done
[ -n "$RUNTIME_NAME" ] || { echo "--runtime is required"; exit 2; }
[[ "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "--gpus must list GPU indexes, comma separated: '$GPUS'"; exit 2; }
[ "$(tr ',' '\n' <<< "$GPUS" | sort -u | wc -l)" -eq "$(tr ',' '\n' <<< "$GPUS" | wc -l)" ] \
    || { echo "--gpus lists an index twice: '$GPUS'"; exit 2; }
[[ "$WALL_HOURS" =~ ^[0-9]+$ ]] && [ "$WALL_HOURS" -ge 12 ] \
    || { echo "--wall-hours must be a whole number of hours, at least 12: '$WALL_HOURS'"; exit 2; }

# shellcheck source=slurm/discovery_lib.sh
source "$SELF_DIR/slurm/discovery_lib.sh"
disc_paths || exit 1
disc_env || exit 1
# The cell's tmux panes and the agent's shell run bash, as on fir, whatever
# the account's login shell is.
export SHELL=/bin/bash
export AUTOMIL_TMUX_SOCKET="disc_chain_$$"
if ! disc_static_preflight; then
    [ "$DRY_RUN" = 1 ] && echo "(dry run: preflight would refuse a real run)" || exit 1
fi

if [ "$DRY_RUN" = 0 ]; then
    exec 9>>"$LOG_DIR/.chain.lock"
    flock -n 9 || { echo "ERROR: another chain of $RUNTIME_NAME holds $LOG_DIR/.chain.lock"; exit 1; }
fi
# The job runs in a session of its own and never sees a Ctrl-C, so bash would
# treat the signal as handled and go on to the next cell. The trap runs once
# the running job has ended and stops the chain there.
trap 'echo "$(date "+%m-%d %H:%M") stop requested: the chain ends after this cell"; exit 130' INT

while :; do
    if [ "$DRY_RUN" = 0 ]; then
        disc_usage_probe refuse "$LOG_DIR/usage_probe_${USER}_$(date +%Y%m%d%H%M%S).txt" || exit 1
    fi
    SCAN=$(disc_scan) || { echo "ERROR: cell scan failed"; exit 1; }
    disc_scan_report "$SCAN"
    ROWS=$(disc_candidates "$SCAN") || { echo "ERROR: cannot order the candidates"; exit 1; }
    if [ "$DRY_RUN" = 1 ]; then
        echo "pick order (dry run, nothing started):"
        echo "${ROWS:-  (none)}"
        exit 0
    fi
    [ -n "$ROWS" ] || { echo "$(date '+%m-%d %H:%M') chain complete: no finishable or pending cell left"; exit 0; }
    ENTRY=$(head -n 1 <<< "$ROWS"); MODE="${ENTRY%%:*}"; CELL="${ENTRY#*:}"
    RUN_ID="ws$(date +%Y%m%d%H%M%S)-$$"
    OUT="$PROJECT_DIR/logs/disc_cell_${RUN_ID}.out"
    echo "$(date '+%m-%d %H:%M') run $RUN_ID: $CELL ($MODE) on GPUs $GPUS, wall ${WALL_HOURS}h -> $OUT"
    DISC_RUN_ID="$RUN_ID" DISC_GPUS="$GPUS" DISC_WALL_END=$(( $(date +%s) + 10#$WALL_HOURS * 3600 )) \
        DISC_CELL="$CELL" DISC_MODE="$MODE" DISC_RUNTIME="$RUNTIME_NAME" \
        setsid -w bash "$JOB_SCRIPT" > "$OUT" 2>&1 < /dev/null
    RC=$?
    if [ "$RC" != 0 ]; then
        echo "$(date '+%m-%d %H:%M') run $RUN_ID failed (exit $RC): chain stopped. See $OUT and $LOG_DIR/FAILED.tsv"
        exit "$RC"
    fi
done
