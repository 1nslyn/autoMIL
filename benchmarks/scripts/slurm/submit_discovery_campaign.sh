#!/bin/bash
# SLURM: preprint campaign DISCOVERY — ONE cell per job, on the GPUs this job
# was shaped for (1, 2 or 4 x H100; 12 h or 24 h wall). Submit through
# submit_discovery_cell.sh, which fits the shape to the cell and claims the
# cell with this job's id; a bare `sbatch` of this file (defaults below:
# 1 GPU, 12 h) is also valid and then picks a cell that fits its own wall.
#
# The whole cell runs on this node: reproduction gate -> up (orchestrator
# daemon on this job's GPUs) -> launch (pinned claude, interactive in a
# job-private tmx server) -> bind -> release line -> watch (with the
# active-time nudge) -> usage capture -> /exit -> finish (freeze ->
# promotion on the same GPUs -> winner -> finalize) -> chain the next cell.
#
# THE ONE RULE THAT MATTERS: a wall-kill mid-session strands the cell
# PERMANENTLY (one session per cell, no relaunch; freeze demands exactly 30
# charged attempts). So the job never starts a cell whose predicted duration
# exceeds its remaining wall, and USR1 only reports. Claims are once-only
# tombstones (see discovery_lib.sh); this job refuses to run a cell whose
# claim it does not hold.
#
# Runs as the submitting member inside the shared project-space tree: umask
# 007, uv never syncs, git trusts the shared checkout, tmx is job-private,
# and every file the cell leaves behind is normalized group read/write.

#SBATCH --job-name=disc_cell
#SBATCH --account=def-jma-ab
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=128G
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/disc_cell_%j.out
#SBATCH --error=logs/disc_cell_%j.err
#SBATCH --mail-type=FAIL

set -uo pipefail
[ -n "${SLURM_JOB_ID:-}" ] || { echo "ERROR: this script runs only as a SLURM job (submit through submit_discovery_cell.sh)"; exit 1; }
# A spooled batch script no longer knows where it came from: the tree is
# named explicitly by the wrapper, or by the submit directory. Never a
# user-path fallback.
export DISC_PROJECT_DIR="${DISC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
[ -n "$DISC_PROJECT_DIR" ] || { echo "ERROR: DISC_PROJECT_DIR is unset (submit through submit_discovery_cell.sh)"; exit 1; }
[ -f "$DISC_PROJECT_DIR/benchmarks/scripts/slurm/discovery_lib.sh" ] \
    || { echo "ERROR: $DISC_PROJECT_DIR is not the campaign checkout"; exit 1; }
# shellcheck source=discovery_lib.sh
# The chain decision travels in from the submit wrapper; it is read here,
# before any sourced file (benchmarks/.env via disc_env) can redefine it.
NO_CHAIN="${DISC_NO_CHAIN:-0}"
source "$DISC_PROJECT_DIR/benchmarks/scripts/slurm/discovery_lib.sh"
disc_paths || exit 1
disc_env
module load cuda/12.2 2>/dev/null || true
disc_static_preflight || exit 1
export AUTOMIL_TMUX_SOCKET="disc_${SLURM_JOB_ID:-manual}"
FAILED_TSV="$PROJECT_DIR/logs/discovery_cells/FAILED.tsv"

N_GPUS="${SLURM_GPUS_ON_NODE:-1}"
GPU_LIST=$(seq -s, 0 $((N_GPUS - 1)))
SEEN_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
[ "$SEEN_GPUS" = "$N_GPUS" ] || { echo "ERROR: SLURM granted $N_GPUS GPUs but nvidia-smi shows $SEEN_GPUS"; exit 1; }

record_failure() {  # cell reason
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "${SLURM_JOB_ID:-manual}" "$USER" "$1" "$2" >> "$FAILED_TSV"
    echo "FAIL $1: $2"
}

_usr1_report() {
    echo "[signal] CRITICAL: wall reached while $CELL may be mid-session — no relaunch exists; OPERATOR NEEDED"
}
trap _usr1_report USR1

# ---------------------------------------------------------------- cell pick
CELL="${DISC_CELL:-}"; MODE="${DISC_MODE:-full}"
if [ -n "$CELL" ]; then
    # The wrapper claims the cell with this job's id right AFTER sbatch
    # returns; a job that lands on a free node can start inside that window,
    # so give the claim up to two minutes to appear before refusing.
    for _ in $(seq 1 24); do
        holder=$(claim_holder "$CELL")
        [ "$holder" = "${SLURM_JOB_ID:-manual}" ] && break
        sleep 5
    done
    holder=$(claim_holder "$CELL")   # a claim written during the last sleep is still valid
    if [ "$holder" != "${SLURM_JOB_ID:-manual}" ]; then
        echo "ERROR: claim for $CELL is held by '${holder:-nobody}', not this job — refusing"; exit 4
    fi
else
    # Direct sbatch: take the first cell that fits THIS job's wall and GPUs.
    SCAN=$(disc_scan) || { echo "ERROR: cell scan failed"; exit 1; }
    HOURS=$(remaining_hours)
    PICK=$(SCAN_JSON="$SCAN" pyrun - "$RUNTIME" "$N_GPUS" "$HOURS" <<'PYEOF'
import importlib.util, json, os, sys
from pathlib import Path
d = json.loads(os.environ["SCAN_JSON"]); runtime, gpus, hours = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
spec = importlib.util.spec_from_file_location("shape", "benchmarks/scripts/campaign_shape.py")
shape = importlib.util.module_from_spec(spec); sys.modules["shape"] = shape; spec.loader.exec_module(shape)
for cell in d["finishable"]:
    print(f"finish:{cell}"); sys.exit(0)
for cell in d["pending"]:
    state, reason = shape._read_campaign_state(runtime, cell)
    e5, reason = shape._baseline_elapsed_seconds(state) if reason is None else (None, reason)
    if e5 is None:
        continue
    if shape.predict_hours(e5, gpus) <= shape.FIT_FRACTION * hours:
        print(f"full:{cell}"); sys.exit(0)
PYEOF
    )
    [ -n "$PICK" ] || { echo "No cell fits this job (${N_GPUS} GPU, ${HOURS}h left)."; exit 0; }
    MODE="${PICK%%:*}"; CELL="${PICK#*:}"
    take_claim "$CELL" || { echo "$CELL was claimed first — exiting"; exit 0; }
fi
LOG="$PROJECT_DIR/logs/discovery_cells/${CELL}.log"
OPDIR="$RUNTIME/$CELL/operator"; mkdir -p "$OPDIR"
trap 'normalize_cell_modes "$CELL"' EXIT
echo "================================================"
echo "preprint DISCOVERY cell $CELL (mode=$MODE) | job ${SLURM_JOB_ID:-manual} | $(hostname) | $USER | $(date)"
echo "GPUs $GPU_LIST | wall left $(remaining_hours)h | tmux socket $AUTOMIL_TMUX_SOCKET | chain $([ "$NO_CHAIN" = 1 ] && echo off || echo on)"
echo "================================================"

# ------------------------------------------------------ usage-window probe
# The wrapper already refused above the weekly threshold before claiming;
# here the value is only recorded (the queue wait may have moved it).
usage_probe() { disc_usage_probe log "$OPDIR/usage_probe.txt"; }

# The /usage panel (seat windows + this session's tokens and cost) is a
# local UI command: no model turn, no journal event. Captured after bind
# and before /exit for the seat-portion measurement.
capture_status() {  # name outfile
    wait_idle "$1" >> "$LOG" 2>&1 || echo "[$1] capturing /usage from a busy runtime" >> "$LOG"
    type_line "$1" "/usage"
    sleep 10
    tmx capture-pane -p -t "=$1:agent" -S -80 > "$2" 2>/dev/null
    tmx send-keys -t "=$1:agent" Escape 2>/dev/null || true
    sleep 2
}

# Token/cost counters straight from the session's own exporter, in the exact
# field set finalize-agent-session validates. Status is "estimated": the
# scrape precedes the final turn.
scrape_usage() {  # cell outfile
    local port
    port=$(pyrun - "$RUNTIME/$1/automil/config.yaml" <<'PYEOF'
import sys, yaml
port = (yaml.safe_load(open(sys.argv[1])).get("activity") or {}).get("exporter_port")
sys.exit("cell config declares no activity.exporter_port") if port is None else print(port)
PYEOF
    ) || return 1
    local metrics attempt
    for attempt in 1 2 3; do   # the exporter answers slowly at times (daemon logs "timed out")
        metrics=$(curl -s -m 20 "http://127.0.0.1:$port/metrics") && [ -n "$metrics" ] && break
        echo "exporter scrape attempt $attempt on port $port failed" >&2; metrics=""; sleep 10
    done
    [ -n "$metrics" ] || return 1
    METRICS="$metrics" pyrun - "$2" <<'PYEOF'
import json, os, re, sys, datetime as dt
out = sys.argv[1]; tokens = {}; cost = 0.0
for line in os.environ["METRICS"].splitlines():
    m = re.match(r'claude_code_token_usage_total\{([^}]*)\}\s+([0-9.eE+-]+)', line)
    if m:
        kind = re.search(r'type="([^"]+)"', m.group(1)).group(1)
        tokens[kind] = tokens.get(kind, 0.0) + float(m.group(2)); continue
    m = re.match(r'claude_code_cost_usage_total\{[^}]*\}\s+([0-9.eE+-]+)', line)
    if m:
        cost += float(m.group(1))
if "input" not in tokens or "output" not in tokens:
    sys.exit("exporter scrape carried no token counters")
payload = {
    "status": "estimated",
    "input_tokens": int(tokens["input"]), "output_tokens": int(tokens["output"]),
    "cached_input_tokens": int(tokens.get("cacheRead", 0) + tokens.get("cacheCreation", 0)),
    "cost_usd": round(cost, 6),
    "basis": "launcher scrape of the session's OTEL prometheus exporter before /exit at "
             + dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
             + " (cached = cacheRead + cacheCreation; cost = runtime list-price estimate)",
}
with open(out, "w") as fh:
    json.dump(payload, fh, indent=2); fh.write("\n")
PYEOF
}

# Poll until discovery is complete (30/30 charged by the orchestrator's own
# count, queue and running empty, twice in a row), the agent window dies, or
# the deadline passes. The nudge
# fires only when the runtime's active time has been FLAT for
# DISC_NUDGE_IDLE_S while the queue is drained and attempts remain — an
# empty queue alone is normal while the agent diagnoses and plans.
watch_discovery() {
    local cell="$1" name="$2" deadline="$3" stable=0 verdict=""
    while :; do
        if [ "$(date +%s)" -ge "$deadline" ]; then echo "deadline"; return 1; fi
        verdict=$(pyrun - "$RUNTIME/$cell" "$OPDIR/watch_state.json" \
            "$DISC_NUDGE_IDLE_S" "$DISC_NUDGE_MIN_GAP_S" "$DISC_NUDGE_MAX" <<'PYEOF'
import importlib.util, json, sys, time
from pathlib import Path
from autobench.campaign import DISCOVERY_ATTEMPTS
spec = importlib.util.spec_from_file_location("campaign_scan", "benchmarks/scripts/campaign_scan.py")
scan = importlib.util.module_from_spec(spec); sys.modules["campaign_scan"] = scan; spec.loader.exec_module(scan)
root, state_path = Path(sys.argv[1]), Path(sys.argv[2])
idle_s, gap_s, max_nudges = (int(x) for x in sys.argv[3:6])
# The ledger's attempts_charged is written by freeze-discovery only; the
# orchestrator's budget-cell census and its queue/running dirs are the live
# view, read the way the cell scan reads them (campaign_scan).
charged = scan.attempts_charged(root)
queued, running = scan.pending_work(root)
if charged == DISCOVERY_ATTEMPTS and not queued and not running:
    print("complete"); sys.exit(0)
try:
    samples = json.loads((root / "automil" / ".activity.samples.json").read_text())["sessions"]
    active = sum(float(s.get("active_seconds") or 0) for s in samples.values())
except (OSError, ValueError, KeyError):
    active = None
now = time.time()
try:
    prior = json.loads(state_path.read_text())
except (OSError, ValueError):
    prior = {"active": None, "changed_at": now, "nudges": 0, "last_nudge": 0}
moved = active is not None and active != prior.get("active")
st = {**prior, "active": active, "changed_at": now} if moved else prior
flat = active is not None and now - st["changed_at"] >= idle_s
action = "working"
if flat and not queued and not running and st["nudges"] < max_nudges and now - st["last_nudge"] >= gap_s:
    st = {**st, "nudges": st["nudges"] + 1, "last_nudge": now}
    action = "nudge"
state_path.write_text(json.dumps(st))
print(f"{action} charged={charged} q={queued} r={running} active={active} flat_s={int(now - st['changed_at'])}")
PYEOF
        ) || verdict="status-error"
        if ! tmx list-windows -t "=$name" 2>/dev/null | grep -q "agent"; then
            echo "agent-window-dead"; return 1
        fi
        case "$verdict" in
            complete) stable=$((stable + 1)); [ "$stable" -ge 2 ] && { echo "complete"; return 0; } ;;
            nudge*)
                stable=0
                type_line "$name" "$DISC_NUDGE_LINE"
                printf '{"event":"operator_nudge","at":"%s","detail":"%s"}\n' "$(date -Is)" "$verdict" >> "$RUNTIME/$cell/operator_events.jsonl"
                echo "[$cell] nudge sent ($verdict)" >> "$LOG" ;;
            *) stable=0 ;;
        esac
        sleep "$DISC_WATCH_INTERVAL_S"
    done
}

# /exit into the agent window, then wait for SessionEnd evidence (journal
# session_end, read by campaign_scan.py). One resend, then give up loudly.
end_session() {
    local cell="$1" name="$2" attempt
    for attempt in 1 2; do
        wait_idle "$name" >> "$LOG" 2>&1 || echo "[$cell] sending /exit to a busy runtime" >> "$LOG"
        type_line "$name" "/exit" 2>/dev/null
        for _ in $(seq 1 30); do
            sleep 30
            pyrun benchmarks/scripts/campaign_scan.py --runtime "$RUNTIME" --session-ended "$cell" && return 0
        done
        echo "[$cell] /exit attempt $attempt produced no session_end in 15m"
    done
    return 1
}

# The finish ladder on this job's GPUs, with the session's scraped usage
# when a run (this one or the one that ended the session) left it behind.
finish_cell() {  # cell
    local usage_flag=()
    [ -s "$OPDIR/usage.json" ] && usage_flag=(--usage-json "$OPDIR/usage.json")
    operate finish "$RUNTIME/$1" --gpu "$GPU_LIST" ${usage_flag[@]+"${usage_flag[@]}"} >> "$LOG" 2>&1
}

run_cell() {
    local cell="$1" mode="$2" name release_line force_flag="" outcome deadline
    if [ "$mode" = "finish" ]; then
        if finish_cell "$cell"; then
            echo "$(date +%m-%d\ %H:%M) DONE $cell (finish-only)"; return 0
        fi
        record_failure "$cell" "finish-failed (see $LOG)"; return 1
    fi
    usage_probe || return 1

    # 1. Reproduction gate (gate mode) on the first GPU. Measurement-mode
    #    blocks from the epsilon derivation are superseded exactly once.
    # --force supersedes, auditably, a measurement-mode block or a verdict
    # recorded at a commit other than this tree's HEAD (the tree moved; the
    # launch preflight anchors on the verdict commit, so a stale pass would
    # refuse the launch and a plain re-run is refused without --force).
    if pyrun - "$RUNTIME/$cell/campaign_state.json" "$(git rev-parse HEAD)" <<'PYEOF' 2>/dev/null
import json, sys
b = (json.load(open(sys.argv[1])).get("baseline_reproduction") or {})
stale = b.get("verdict") is not None and b.get("commit") != sys.argv[2]
sys.exit(0 if b.get("mode") == "measurement" or stale else 1)
PYEOF
    then
        force_flag="--force"
    fi
    if ! stage run-baseline-reproduction --cell-root "$RUNTIME/$cell" --gpu 0 $force_flag >> "$LOG" 2>&1; then
        if grep -q "reproduction FAILED" "$LOG"; then
            record_failure "$cell" "gate-FAILED (drift recorded; discovery blocked until --force after review)"
        else
            record_failure "$cell" "reproduction-error (see $LOG)"
        fi
        return 1
    fi

    # 2-4. Daemon on this job's GPUs + session + bind, via the canonical operator.
    local name_and_release
    name_and_release=$(pyrun -c "
import importlib.util
spec=importlib.util.spec_from_file_location('op','benchmarks/scripts/campaign_operate.py')
import sys; m=importlib.util.module_from_spec(spec); sys.modules['op']=m; spec.loader.exec_module(m)
from pathlib import Path
print(m._session_name(Path('$RUNTIME/$cell')))
print(m.RELEASE_LINE)")
    name=$(echo "$name_and_release" | sed -n 1p)
    release_line=$(echo "$name_and_release" | sed -n 2p)
    for step in "up $RUNTIME/$cell --gpu $GPU_LIST" "launch $RUNTIME/$cell" \
                "bind $RUNTIME/$cell --timeout-s 900"; do
        if [ "${step%% *}" = "launch" ]; then
            wait_daemon "$RUNTIME/$cell" >> "$LOG" 2>&1 || { record_failure "$cell" "orchestrator-not-up (see $LOG)"; return 1; }
            clear_plugins
            trust_paths "$RUNTIME/$cell" >> "$LOG" 2>&1
        fi
        if [ "${step%% *}" = "bind" ]; then
            wait_exporter "$RUNTIME/$cell" >> "$LOG" 2>&1 || { record_failure "$cell" "exporter-not-serving (see $LOG)"; tmx kill-session -t "=$name" 2>/dev/null; return 1; }
            # If a first-run prompt still shows, record the pane; the folder
            # trust prompt defaults to "Yes" and is answered with Enter.
            local waited=0 pane
            while [ "$waited" -lt 60 ]; do
                pane=$(tmx capture-pane -p -t "=$name:agent" 2>/dev/null)
                if echo "$pane" | grep -q "trust this folder"; then
                    echo "[$cell] answering the folder-trust prompt" >> "$LOG"; tmx send-keys -t "=$name:agent" Enter; sleep 5
                elif echo "$pane" | grep -q "Yes, I accept"; then   # the dialog, not the status bar's "bypass permissions on"
                    echo "[$cell] bypass-permissions prompt still shown after pre-seeding; pane follows" >> "$LOG"; echo "$pane" >> "$LOG"; break
                else
                    break
                fi
                waited=$((waited + 5))
            done
            wake_runtime "$name"; sleep 20   # first active-time sample lands within ~20 s
        fi
        if ! operate $step >> "$LOG" 2>&1; then
            record_failure "$cell" "operate-${step%% *}-failed (see $LOG)"
            if [ "${step%% *}" != "up" ]; then
                # A runtime may already be booting: kill it so it can never
                # journal a late session_open after this job gave up.
                tmx kill-session -t "=$name" 2>/dev/null || true
            fi
            return 1
        fi
    done
    capture_status "$name" "$OPDIR/usage_before.txt"
    sleep 5
    type_line "$name" "$release_line"
    echo "$(date +%m-%d\ %H:%M) session bound + released for $cell (tmux $name on $AUTOMIL_TMUX_SOCKET)"

    # 5. Watch until 30/30 and drained; the deadline keeps the finish
    #    reserve inside this job's wall.
    deadline=$(( $(date +%s) + ($(remaining_hours) - DISC_FINISH_RESERVE_H) * 3600 ))
    outcome=$(watch_discovery "$cell" "$name" "$deadline")
    if [ "$outcome" = "deadline" ]; then
        scrape_usage "$cell" "$OPDIR/usage.json" 2>> "$LOG" || echo "[$cell] exporter scrape failed at the deadline" >> "$LOG"
        capture_status "$name" "$OPDIR/usage_after.txt"
        end_session "$cell" "$name" || true
        record_failure "$cell" "wall-deadline (session ended for the ledger; the scan says finishable or stranded) — OPERATOR NEEDED"
        return 1
    fi
    if [ "$outcome" != "complete" ]; then
        record_failure "$cell" "session-died-mid-discovery ($outcome) — OPERATOR NEEDED, cell may be stranded"
        return 1
    fi

    # 6. Usage capture, session end, then the finish ladder on the same GPUs.
    scrape_usage "$cell" "$OPDIR/usage.json" 2>> "$LOG" || echo "[$cell] exporter scrape failed; finish will record usage as unavailable" >> "$LOG"
    capture_status "$name" "$OPDIR/usage_after.txt"
    if ! end_session "$cell" "$name"; then
        record_failure "$cell" "no-session-end-after-exit — OPERATOR NEEDED"; return 1
    fi
    if ! finish_cell "$cell"; then
        record_failure "$cell" "finish-failed (see $LOG)"; return 1
    fi
    tmx kill-session -t "=$name" 2>/dev/null || true
    echo "$(date +%m-%d\ %H:%M) DONE $cell (winner finalized)"
    return 0
}

run_cell "$CELL" "$MODE"; RC=$?
normalize_cell_modes "$CELL"
tmx kill-server 2>/dev/null || true
echo "---"
# Chain ONLY after a clean cell. A failed cell must stop the chain: a
# systematic failure would otherwise burn one queue slot after another
# (overnight 2026-09-03: ~70 short jobs alternating between two cells).
if [ "$RC" = 0 ] && [ "$NO_CHAIN" != 1 ]; then
    echo "chaining: submitting the next cell as $USER"
    "$PROJECT_DIR/benchmarks/scripts/slurm/submit_discovery_cell.sh" --chain --account "${DISC_ACCOUNT:-$DISC_ACCOUNT_DEFAULT}" \
        || echo "  chain: nothing submitted (see above)"
fi
[ "$RC" = 0 ] && echo "Cell finished cleanly. $(date)"
exit "$RC"
