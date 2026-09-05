#!/bin/bash
# Member entry point for the preprint discovery campaign: submit ONE cell as
# ONE SLURM job, shaped for that cell.
#
# Run from any login node as yourself, from anywhere:
#   /project/.../work/autoMIL/benchmarks/scripts/slurm/submit_discovery_cell.sh
#
# What it does, in order:
#   1. static preflight (pinned claude on PATH, your Claude login present,
#      no user memory/plugins, clean instruction surface on the shared path,
#      every sibling session record readable) — refused BEFORE anything is
#      claimed, so a refusal never burns a queue slot;
#   2. scan the roster (campaign_scan.py) and take the first cell that can
#      be driven: finish-only recoveries first, then pending cells in roster
#      order;
#   3. fit the job to the cell (campaign_shape.py: 1, 2 or 4 GPUs; 12 h or
#      24 h wall; 12 cores + 128 GB per GPU; cheapest fitting shape by
#      default, --prefer fast for the shortest wall) — a cell that fits no
#      shape is reported, never submitted;
#   4. sbatch the job, then claim the cell with the NEW job id (O_EXCL). If a
#      concurrent submitter won the claim first, the fresh job is cancelled
#      (it has no queue age to lose) and the next cell is tried.
# The job itself (submit_discovery_campaign.sh) verifies at start that the
# claim carries its own id and, at its clean end, calls this script with
# --chain to submit the next cell as the same user.
#
# Options:
#   --dry-run       classify + shape every cell; submit nothing
#   --cell ID       submit exactly this cell (must be pending/finishable)
#   --account NAME  SLURM account (default: def-jma-ab)
#   --max-gpus N    never request more than N GPUs for this submission
#   --prefer MODE   cheap (default: fewest GPU-hours) | fast (shortest wall)
#   --runtime NAME  cell-root directory under the campaign dir (default:
#                   runtime, the final grid); a rehearsal set such as
#                   runtime-rehearsal has its own roster NAME.roster.json
#   --no-chain      the job stops after its own cell instead of submitting the
#                   next one (rehearsals, member tests); decided per submission
#                   from this option only, never from the caller's environment
#   --chain         quiet mode used by a finishing job; exit 0 when nothing
#                   is left to submit

set -uo pipefail
SELF_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export DISC_PROJECT_DIR
DISC_PROJECT_DIR=$(cd "$SELF_DIR/../../.." && pwd)
# shellcheck source=discovery_lib.sh
source "$SELF_DIR/discovery_lib.sh"
JOB_SCRIPT="$SELF_DIR/submit_discovery_campaign.sh"

DRY_RUN=0; ONLY_CELL=""; ACCOUNT="$DISC_ACCOUNT_DEFAULT"; MAX_GPUS=4; CHAIN=0; NO_CHAIN=0; RUNTIME_NAME="runtime"
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --cell) ONLY_CELL="$2"; shift ;;
        --account) ACCOUNT="$2"; shift ;;
        --max-gpus) MAX_GPUS="$2"; shift ;;
        --prefer) export DISC_PREFER="$2"; shift ;;
        --no-chain) NO_CHAIN=1 ;;
        --runtime) RUNTIME_NAME="$2"; shift ;;
        --chain) CHAIN=1 ;;
        -h|--help) sed -n '2,/^$/p' "$0"; exit 0 ;;
        *) echo "unknown option: $1"; exit 2 ;;
    esac
    shift
done

case "$MAX_GPUS" in 1|2|4) ;; *) echo "--max-gpus must be 1, 2 or 4"; exit 2 ;; esac
disc_paths || exit 1
disc_env
export AUTOMIL_TMUX_SOCKET="disc_submit_$$"
if ! disc_static_preflight; then
    [ "$DRY_RUN" = 1 ] && echo "(dry run: preflight would refuse a real submission)" || exit 1
fi
if [ "$DRY_RUN" = 0 ]; then
    mkdir -p "$LOG_DIR"
    disc_usage_probe refuse "$LOG_DIR/usage_probe_${USER}_$(date +%Y%m%d%H%M%S).txt" || exit 1
fi

SCAN=$(disc_scan) || { echo "ERROR: cell scan failed"; exit 1; }
summary=$(echo "$SCAN" | pyrun -c \
    "import json,sys;d=json.load(sys.stdin);print(', '.join(f'{k}={len(d[k])}' for k in ('pending','finishable','claimed','done','stranded','blocked')), '| squeue_ok=%s'%d['squeue_ok'])")
[ "$CHAIN" = 1 ] || echo "scan ($RUNTIME_NAME, roster $(basename "$ROSTER")): $summary"
echo "$SCAN" | pyrun -c "import json,sys;d=json.load(sys.stdin);[print('  note:',c,'-',n) for c,n in sorted(d['notes'].items())]"

# A heredoc replaces stdin, so JSON produced upstream travels in a variable.
candidates() {
    SCAN_JSON="$SCAN" pyrun - "$ONLY_CELL" <<'PYEOF'
import json, os, sys
d = json.loads(os.environ["SCAN_JSON"]); only = sys.argv[1]
rows = [("finish", c) for c in d["finishable"]] + [("full", c) for c in d["pending"]]
if only:
    rows = [r for r in rows if r[1] == only] or sys.exit(f"{only} is not finishable or pending")
print("\n".join(f"{m}:{c}" for m, c in rows))
PYEOF
}

# One predictor call per cell: "gpus wall cpus mem whole_node predicted e5".
# A finish-only recovery takes the predictor's finish lane (one GPU, short
# wall: promotion of ten candidates fits it for every roster cell).
shape_for() {
    local mode="$1" cell="$2" args shape_json
    if [ "$mode" = "finish" ]; then args="--finish"; else args="--runtime $RUNTIME --prefer $DISC_PREFER --cells $cell --json"; fi
    shape_json=$(pyrun benchmarks/scripts/campaign_shape.py $args) || return 1
    SHAPE_JSON="$shape_json" pyrun - "$cell" "$mode" <<'PYEOF'
import json, os, sys
payload = json.loads(os.environ["SHAPE_JSON"]); cell, mode = sys.argv[1:3]
shape = payload if mode == "finish" else payload[cell]
if "unshaped" in shape:
    sys.exit(f"{cell}: {shape['unshaped']}")
print(shape["gpus"], shape["wall_hours"], shape["cpus"], shape["mem_gb"],
      int(bool(shape["whole_node"])), shape["predicted_hours"], shape.get("baseline_elapsed_seconds", 0))
PYEOF
}

submit_one() {
    local mode="$1" cell="$2" shape gpus wall cpus mem whole pred e5 jobid mem_flag name
    shape=$(shape_for "$mode" "$cell") || { echo "  $cell: no shape fits (see campaign_shape.py) — skipped"; return 1; }
    read -r gpus wall cpus mem whole pred e5 <<< "$shape"
    if [ "$gpus" -gt "$MAX_GPUS" ]; then
        echo "  $cell: needs $gpus GPUs > --max-gpus $MAX_GPUS — skipped"; return 1
    fi
    # A whole-node shape takes the node's memory like the baseline launcher did.
    if [ "$whole" = 1 ]; then mem_flag="--mem=0"; else mem_flag="--mem=${mem}G"; fi
    name="disc_$(echo "$cell" | awk -F__ '{print $1"__"$2"__"$3"__"$4}')"
    printf '  %-58s mode=%-6s gpus=%s wall=%sh cpus=%s mem=%s predicted=%sh\n' \
        "$cell" "$mode" "$gpus" "$wall" "$cpus" "${mem_flag#--mem=}" "$pred"
    [ "$DRY_RUN" = 1 ] && return 1
    # The decision is bound to this one sbatch call, after every sourced file
    # (benchmarks/.env via disc_env) has had its say, so the job sees one
    # value however sbatch merges --export=ALL with explicit assignments.
    jobid=$(DISC_NO_CHAIN="$NO_CHAIN" DISC_RUNTIME="$RUNTIME_NAME" sbatch --parsable --account="$ACCOUNT" --time="${wall}:00:00" \
        --nodes=1 --ntasks-per-node=1 --cpus-per-task="$cpus" "$mem_flag" \
        --gpus-per-node="h100:$gpus" --job-name="$name" \
        --output="logs/disc_cell_%j.out" --error="logs/disc_cell_%j.err" \
        --export="ALL,DISC_PROJECT_DIR=$PROJECT_DIR,DISC_CELL=$cell,DISC_MODE=$mode,DISC_ACCOUNT=$ACCOUNT,DISC_PREFER=$DISC_PREFER,DISC_NO_CHAIN=$NO_CHAIN,DISC_RUNTIME=$RUNTIME_NAME" \
        "$JOB_SCRIPT") || { echo "  sbatch failed for $cell"; return 1; }
    jobid="${jobid%%;*}"
    if ! take_claim "$cell" "$jobid"; then
        echo "  $cell was claimed by another submitter first — cancelling fresh job $jobid"
        scancel "$jobid" 2>/dev/null || true
        return 1
    fi
    mkdir -p "$RUNTIME/$cell/operator"
    pyrun - "$RUNTIME/$cell/operator/plan.json" "$cell" "$mode" "$jobid" "$gpus" "$wall" "$cpus" "$mem" "$whole" "$pred" "$e5" "$ACCOUNT" "$RUNTIME_NAME" <<'PYEOF'
import importlib.util, json, os, sys, datetime as dt
path, cell, mode, jobid, gpus, wall, cpus, mem, whole, pred, e5, account, runtime_name = sys.argv[1:]
spec = importlib.util.spec_from_file_location("shape", "benchmarks/scripts/campaign_shape.py")
shape = importlib.util.module_from_spec(spec); sys.modules["shape"] = shape; spec.loader.exec_module(shape)
payload = {
    "cell_id": cell, "mode": mode, "job_id": jobid, "account": account, "runtime": runtime_name,
    "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    "submitted_by": os.environ.get("USER"),
    "shape": {"gpus": int(gpus), "wall_hours": int(wall), "cpus": int(cpus), "mem_gb": int(mem),
              "whole_node": whole == "1"},
    "predicted_hours": float(pred),
    "baseline_elapsed_seconds": float(e5),
    "predictor": {"cap_per_gpu": shape.CAP_PER_GPU, "efficiency": shape.EFFICIENCY,
                  "fit_fraction": shape.FIT_FRACTION, "overhead_h": shape.OVERHEAD_H,
                  "prefer": os.environ.get("DISC_PREFER", "cheap")},
}
with open(path, "w") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True); fh.write("\n")
PYEOF
    echo "  submitted job $jobid for $cell ($mode, ${gpus}xH100, ${wall}h, set $RUNTIME_NAME, chain $([ "$NO_CHAIN" = 1 ] && echo off || echo on)) — claim taken"
    return 0
}

ROWS=$(candidates) || { echo "$ROWS"; exit 1; }
if [ -z "$ROWS" ]; then
    [ "$CHAIN" = 1 ] || echo "Nothing to submit: no finishable or pending cells."
    exit 0
fi
[ "$DRY_RUN" = 1 ] && echo "candidates (dry run, nothing submitted):"
while IFS= read -r entry; do
    [ -n "$entry" ] || continue
    if submit_one "${entry%%:*}" "${entry#*:}"; then
        exit 0
    fi
done <<< "$ROWS"
[ "$DRY_RUN" = 1 ] && exit 0
echo "No cell could be submitted (all skipped or claimed)."
exit 1
