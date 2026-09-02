#!/bin/bash
# SLURM: preprint campaign DISCOVERY — ONE cell per job, on the GPUs this job
# was shaped for (1, 2 or 4 x H100; 12 h or 24 h wall). Submit through
# submit_discovery_cell.sh, which fits the shape to the cell and claims the
# cell with this job's id; a bare `sbatch` of this file (defaults below:
# 1 GPU, 12 h) is also valid and then picks a cell that fits its own wall.
#
# The whole cell runs on this node: reproduction gate -> up (orchestrator
# daemon on this job's GPUs) -> launch (pinned claude, interactive in a
# job-private tmux server) -> bind -> release line -> watch (with the
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
# 007, uv never syncs, git trusts the shared checkout, tmux is job-private,
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
# A spooled batch script no longer knows where it came from: the tree is
# named explicitly by the wrapper, or by the submit directory. Never a
# user-path fallback.
export DISC_PROJECT_DIR="${DISC_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
[ -n "$DISC_PROJECT_DIR" ] || { echo "ERROR: DISC_PROJECT_DIR is unset (submit through submit_discovery_cell.sh)"; exit 1; }
[ -f "$DISC_PROJECT_DIR/benchmarks/scripts/slurm/discovery_lib.sh" ] \
    || { echo "ERROR: $DISC_PROJECT_DIR is not the campaign checkout"; exit 1; }
# shellcheck source=discovery_lib.sh
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
    holder=$(claim_holder "$CELL")
    if [ "$holder" != "${SLURM_JOB_ID:-manual}" ]; then
        echo "ERROR: claim for $CELL is held by '${holder:-nobody}', not this job — refusing"; exit 4
    fi
else
    # Direct sbatch: take the first cell that fits THIS job's wall and GPUs.
    SCAN=$(disc_scan) || { echo "ERROR: cell scan failed"; exit 1; }
    HOURS=$(remaining_hours)
    PICK=$(echo "$SCAN" | pyrun - "$RUNTIME" "$N_GPUS" "$HOURS" <<'PYEOF'
import importlib.util, json, sys
from pathlib import Path
d = json.load(sys.stdin); runtime, gpus, hours = Path(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
spec = importlib.util.spec_from_file_location("shape", "benchmarks/scripts/campaign_shape.py")
shape = importlib.util.module_from_spec(spec); spec.loader.exec_module(shape)
for cell in d["finishable"]:
    print(f"finish:{cell}"); sys.exit(0)
for cell in d["pending"]:
    try:
        e5 = json.loads((runtime / cell / "campaign_state.json").read_text())["baseline"]["resources"]["elapsed_seconds"]["total"]
    except (OSError, ValueError, KeyError, TypeError):
        continue
    if shape.predict_hours(float(e5), gpus) <= shape.FIT_FRACTION * hours:
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
echo "GPUs $GPU_LIST | wall left $(remaining_hours)h | tmux socket $AUTOMIL_TMUX_SOCKET"
echo "================================================"

# ------------------------------------------------------ usage-window probe
# A throwaway runtime in $HOME reads /status (the plan's remaining
# allocation) BEFORE the one-shot session is opened. Killed afterwards;
# unparsable output only warns.
usage_probe() {
    local out="$OPDIR/usage_probe.txt" pct
    tmux -L "$AUTOMIL_TMUX_SOCKET" new-session -d -s usage_probe -c "$HOME" \
        "claude --setting-sources project --strict-mcp-config" 2>/dev/null || return 0
    sleep 15
    tmux -L "$AUTOMIL_TMUX_SOCKET" send-keys -t "=usage_probe" -l "/status"
    tmux -L "$AUTOMIL_TMUX_SOCKET" send-keys -t "=usage_probe" Enter
    sleep 10
    tmux -L "$AUTOMIL_TMUX_SOCKET" capture-pane -p -t "=usage_probe" -S -80 > "$out" 2>/dev/null
    tmux -L "$AUTOMIL_TMUX_SOCKET" kill-session -t "=usage_probe" 2>/dev/null || true
    pct=$(grep -iE 'week' "$out" | grep -oE '[0-9]{1,3}%' | head -1 | tr -d '%')
    if [ -z "$pct" ]; then
        echo "WARNING: could not parse the weekly usage window from /status (see $out) — proceeding"
        return 0
    fi
    echo "usage window: weekly ${pct}% used"
    if [ "$pct" -ge "$DISC_WEEKLY_USAGE_MAX_PCT" ]; then
        record_failure "$CELL" "weekly-usage-${pct}pct (claim left for a later job)"
        return 1
    fi
}

capture_status() {  # name outfile
    tmux send-keys -t "=$1:agent" -l "/status"; tmux send-keys -t "=$1:agent" Enter
    sleep 8
    tmux capture-pane -p -t "=$1:agent" -S -80 > "$2" 2>/dev/null
    tmux send-keys -t "=$1:agent" Escape 2>/dev/null || true
}

# Token/cost counters straight from the session's own exporter, in the exact
# field set finalize-agent-session validates. Status is "estimated": the
# scrape precedes the final turn.
scrape_usage() {  # cell outfile
    local port
    port=$(pyrun -c "import yaml;print((yaml.safe_load(open('$RUNTIME/$1/automil/config.yaml')).get('activity') or {}).get('exporter_port', 9464))")
    curl -s "http://127.0.0.1:$port/metrics" | pyrun - "$2" <<'PYEOF'
import json, re, sys, datetime as dt
out = sys.argv[1]; tokens = {}; cost = 0.0
for line in sys.stdin:
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

# Poll until discovery is complete (30/30 charged, queue and running empty,
# twice in a row), the agent window dies, or the deadline passes. The nudge
# fires only when the runtime's active time has been FLAT for
# DISC_NUDGE_IDLE_S while the queue is drained and attempts remain — an
# empty queue alone is normal while the agent diagnoses and plans.
watch_discovery() {
    local cell="$1" name="$2" deadline="$3" stable=0 verdict=""
    while :; do
        if [ "$(date +%s)" -ge "$deadline" ]; then echo "deadline"; return 1; fi
        verdict=$(pyrun - "$RUNTIME/$cell" "$OPDIR/watch_state.json" \
            "$DISC_NUDGE_IDLE_S" "$DISC_NUDGE_MIN_GAP_S" "$DISC_NUDGE_MAX" <<'PYEOF'
import json, subprocess, sys, time
from pathlib import Path
root, state_path = Path(sys.argv[1]), Path(sys.argv[2])
idle_s, gap_s, max_nudges = (int(x) for x in sys.argv[3:6])
status = json.loads(subprocess.run(
    [sys.executable, "benchmarks/scripts/campaign_stage.py", "status", "--cell-root", str(root)],
    capture_output=True, text=True, check=True).stdout)
adir = root / "automil"
queued = list(adir.glob("orchestrator/queue/*.json"))
running = list(adir.glob("orchestrator/running/**/*.json"))
charged = (status.get("discovery") or {}).get("attempts_charged") or 0
if charged == 30 and not queued and not running:
    print("complete"); sys.exit(0)
try:
    samples = json.loads((adir / ".activity.samples.json").read_text())["sessions"]
    active = sum(float(s.get("active_seconds") or 0) for s in samples.values())
except (OSError, ValueError, KeyError):
    active = None
now = time.time()
try:
    st = json.loads(state_path.read_text())
except (OSError, ValueError):
    st = {"active": None, "changed_at": now, "nudges": 0, "last_nudge": 0}
if active is not None and active != st.get("active"):
    st.update(active=active, changed_at=now)
flat = active is not None and now - st["changed_at"] >= idle_s
action = "working"
if flat and not queued and not running and st["nudges"] < max_nudges and now - st["last_nudge"] >= gap_s:
    st.update(nudges=st["nudges"] + 1, last_nudge=now)
    action = "nudge"
state_path.write_text(json.dumps(st))
print(f"{action} charged={charged} q={len(queued)} r={len(running)} active={active} flat_s={int(now - st['changed_at'])}")
PYEOF
        ) || verdict="status-error"
        if ! tmux list-windows -t "=$name" 2>/dev/null | grep -q "agent"; then
            echo "agent-window-dead"; return 1
        fi
        case "$verdict" in
            complete) stable=$((stable + 1)); [ "$stable" -ge 2 ] && { echo "complete"; return 0; } ;;
            nudge*)
                stable=0
                tmux send-keys -t "=$name:agent" -l "$DISC_NUDGE_LINE"; tmux send-keys -t "=$name:agent" Enter
                printf '{"event":"operator_nudge","at":"%s","detail":"%s"}\n' "$(date -Is)" "$verdict" >> "$RUNTIME/$cell/operator_events.jsonl"
                echo "[$cell] nudge sent ($verdict)" >> "$LOG" ;;
            *) stable=0 ;;
        esac
        sleep "$DISC_WATCH_INTERVAL_S"
    done
}

# /exit into the agent window, then wait for SessionEnd evidence (journal
# session_end). One resend, then give up loudly.
end_session() {
    local cell="$1" name="$2" attempt evidence
    for attempt in 1 2; do
        tmux send-keys -t "=$name:agent" -l "/exit" 2>/dev/null
        tmux send-keys -t "=$name:agent" Enter 2>/dev/null
        for _ in $(seq 1 30); do
            sleep 30
            evidence=$(pyrun - "$RUNTIME/$cell" <<'PYEOF'
import json, sys
from pathlib import Path
journal = Path(sys.argv[1]) / "automil" / ".activity.jsonl"
if journal.is_file():
    for line in journal.read_text().splitlines():
        try:
            if json.loads(line).get("event") == "session_end":
                print("ended"); break
        except ValueError:
            continue
PYEOF
            )
            [ "$evidence" = "ended" ] && return 0
        done
        echo "[$cell] /exit attempt $attempt produced no session_end in 15m"
    done
    return 1
}

run_cell() {
    local cell="$1" mode="$2" name release_line force_flag="" outcome deadline
    if [ "$mode" = "finish" ]; then
        if operate finish "$RUNTIME/$cell" --gpu "$GPU_LIST" >> "$LOG" 2>&1; then
            echo "$(date +%m-%d\ %H:%M) DONE $cell (finish-only)"; return 0
        fi
        record_failure "$cell" "finish-failed (see $LOG)"; return 1
    fi
    usage_probe || return 1

    # 1. Reproduction gate (gate mode) on the first GPU. Measurement-mode
    #    blocks from the epsilon derivation are superseded exactly once.
    if pyrun -c "
import json,sys
b=(json.load(open('$RUNTIME/$cell/campaign_state.json')).get('baseline_reproduction') or {})
sys.exit(0 if b.get('mode')=='measurement' else 1)" 2>/dev/null; then
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
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from pathlib import Path
print(m._session_name(Path('$RUNTIME/$cell')))
print(m.RELEASE_LINE)")
    name=$(echo "$name_and_release" | sed -n 1p)
    release_line=$(echo "$name_and_release" | sed -n 2p)
    for step in "up $RUNTIME/$cell --gpu $GPU_LIST" "launch $RUNTIME/$cell" \
                "bind $RUNTIME/$cell --timeout-s 900"; do
        if ! operate $step >> "$LOG" 2>&1; then
            record_failure "$cell" "operate-${step%% *}-failed (see $LOG)"
            if [ "${step%% *}" != "up" ]; then
                # A runtime may already be booting: kill it so it can never
                # journal a late session_open after this job gave up.
                tmux kill-session -t "=$name" 2>/dev/null || true
            fi
            return 1
        fi
    done
    capture_status "$name" "$OPDIR/usage_before.txt"
    sleep 5
    tmux send-keys -t "=$name:agent" -l "$release_line"; tmux send-keys -t "=$name:agent" Enter
    echo "$(date +%m-%d\ %H:%M) session bound + released for $cell (tmux $name on $AUTOMIL_TMUX_SOCKET)"

    # 5. Watch until 30/30 and drained; the deadline keeps the finish
    #    reserve inside this job's wall.
    deadline=$(( $(date +%s) + ($(remaining_hours) - DISC_FINISH_RESERVE_H) * 3600 ))
    outcome=$(watch_discovery "$cell" "$name" "$deadline")
    if [ "$outcome" = "deadline" ]; then
        scrape_usage "$cell" "$OPDIR/usage.json" 2>/dev/null || true
        end_session "$cell" "$name" || true
        record_failure "$cell" "wall-deadline-STRANDED (session ended for the ledger; <30 attempts) — OPERATOR NEEDED"
        return 1
    fi
    if [ "$outcome" != "complete" ]; then
        record_failure "$cell" "session-died-mid-discovery ($outcome) — OPERATOR NEEDED, cell may be stranded"
        return 1
    fi

    # 6. Usage capture, session end, then the finish ladder on the same GPUs.
    scrape_usage "$cell" "$OPDIR/usage.json" || echo "[$cell] exporter scrape failed; finish will record usage as unavailable" >> "$LOG"
    capture_status "$name" "$OPDIR/usage_after.txt"
    if ! end_session "$cell" "$name"; then
        record_failure "$cell" "no-session-end-after-exit — OPERATOR NEEDED"; return 1
    fi
    local usage_flag=()
    [ -s "$OPDIR/usage.json" ] && usage_flag=(--usage-json "$OPDIR/usage.json")
    if ! operate finish "$RUNTIME/$cell" --gpu "$GPU_LIST" "${usage_flag[@]}" >> "$LOG" 2>&1; then
        record_failure "$cell" "finish-failed (see $LOG)"; return 1
    fi
    tmux kill-session -t "=$name" 2>/dev/null || true
    echo "$(date +%m-%d\ %H:%M) DONE $cell (winner finalized)"
    return 0
}

run_cell "$CELL" "$MODE"; RC=$?
normalize_cell_modes "$CELL"
tmux -L "$AUTOMIL_TMUX_SOCKET" kill-server 2>/dev/null || true
echo "---"
if [ "${DISC_NO_CHAIN:-0}" != 1 ]; then
    echo "chaining: submitting the next cell as $USER"
    "$PROJECT_DIR/benchmarks/scripts/slurm/submit_discovery_cell.sh" --chain --account "${DISC_ACCOUNT:-$DISC_ACCOUNT_DEFAULT}" \
        || echo "  chain: nothing submitted (see above)"
fi
[ "$RC" = 0 ] && echo "Cell finished cleanly. $(date)"
exit "$RC"
