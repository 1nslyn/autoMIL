#!/bin/bash
# SLURM: preprint campaign DISCOVERY — ONE full node, 4 agentic cells in parallel.
#
# Each GPU hosts one cell at a time, driven end-to-end by campaign_operate.py
# inside that cell's tmux session: reproduction gate -> up (daemon) ->
# launch (pinned claude, interactive in tmux) -> bind -> completion watch ->
# automated /exit -> finish (freeze -> promotion on the same GPU -> winner).
#
# THE ONE RULE THAT MATTERS: a wall-kill mid-session strands the cell
# PERMANENTLY (one session per cell, no relaunch; freeze demands exactly 30
# charged attempts). So a worker only STARTS a cell while
# remaining-wall > CELL_WALL_BUDGET_H, and the job never resubmits from the
# USR1 signal — it exits cleanly once no worker can safely start another
# cell, then chains the next job itself. USR1 firing at all means the
# budget was mis-sized; it logs the cells at risk and nothing else.
#
# Cells are claimed via an atomic marker carrying this job id, so several
# chains can run concurrently without double-driving a cell (stale claims
# from dead jobs are reaped at scan time). Cells whose reproduction gate
# fails, or that carry session evidence without a finished ladder
# (stranded), are reported and skipped — never retried silently.
#
# Usage (from the repo root):
#   sbatch benchmarks/scripts/slurm/submit_discovery_campaign.sh

#SBATCH --job-name=disc_campaign
#SBATCH --account=def-jma-ab
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=0
#SBATCH --signal=B:USR1@300
#SBATCH --output=logs/disc_campaign_%j.out
#SBATCH --error=logs/disc_campaign_%j.err
#SBATCH --mail-type=FAIL

set -uo pipefail
PROJECT_DIR="${SLURM_SUBMIT_DIR:-/home/yinshuol/scratch/autoMIL/autoMIL}"
RUNTIME="$PROJECT_DIR/benchmarks/campaigns/preprint_130/runtime"
ROSTER="$PROJECT_DIR/benchmarks/campaigns/preprint_130/active_roster.json"
SELF="$PROJECT_DIR/benchmarks/scripts/slurm/submit_discovery_campaign.sh"
N_GPUS=4
# Conservative worst case for one cell start-to-finish: reproduction ~2h +
# 30 discovery attempts (3-fold, nominal ~1.2h, 10h attempt timeout tail) +
# promotion (10 candidates x 2 folds) ~10h + slack.
CELL_WALL_BUDGET_H=66
WATCH_INTERVAL_S=120

cd "$PROJECT_DIR" || { echo "ERROR: project dir not found: $PROJECT_DIR"; exit 1; }
[ -d "$RUNTIME" ] || { echo "ERROR: runtime not materialized: $RUNTIME"; exit 1; }
[ -f benchmarks/.env ] || { echo "ERROR: benchmarks/.env missing"; exit 1; }
[ -f "$ROSTER" ] || { echo "ERROR: active roster missing: $ROSTER"; exit 1; }
mkdir -p logs/discovery_cells
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv is required"; exit 1; }
command -v tmux >/dev/null 2>&1 || { echo "ERROR: tmux is required"; exit 1; }
module load cuda/12.2 2>/dev/null || true
set -a; source benchmarks/.env; set +a
export DISABLE_AUTOUPDATER=1
# The frozen toolset forbids these; a stray module could have set them.
unset ANTHROPIC_MODEL ANTHROPIC_SMALL_FAST_MODEL ANTHROPIC_BASE_URL \
      CLAUDE_CODE_USE_BEDROCK CLAUDE_CODE_USE_VERTEX CLAUDE_CODE_EFFORT_LEVEL \
      2>/dev/null || true

# The pinned runtime must resolve on this node before any cell is touched.
PROTOCOL_VERSION_PIN=$(uv run --frozen --no-sync --package autobench python -c \
    "import json;print(json.load(open('$RUNTIME/agent_protocol.json'))['runtime_version'])") \
    || { echo "ERROR: cannot read protocol runtime_version"; exit 1; }
OBSERVED_CLAUDE=$(claude --version 2>/dev/null | awk '{print $1}')
if [ "$OBSERVED_CLAUDE" != "$PROTOCOL_VERSION_PIN" ]; then
    echo "ERROR: claude on PATH is ${OBSERVED_CLAUDE:-absent}, protocol pins $PROTOCOL_VERSION_PIN"
    exit 1
fi
# The launch preflight refuses a non-empty ~/.claude/plugins.
rm -rf "$HOME/.claude/plugins" 2>/dev/null || true

# Scan: classify every roster cell. Only clean, unclaimed, gate-eligible
# cells become pending; everything else is reported by class. Claims from
# jobs no longer in squeue are reaped here.
scan_cells() {
    uv run --frozen --no-sync --package autobench python - \
        "$RUNTIME" "$ROSTER" "${SLURM_JOB_ID:-manual}" <<'PYEOF'
import json, shutil, subprocess, sys
from pathlib import Path
runtime, roster_path, job_id = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
roster = json.loads(roster_path.read_text())
cells = sorted(
    entry.name for entry in runtime.iterdir()
    if entry.is_dir() and entry.name.split("__")[0] in set(roster["cohorts"])
)
if len(cells) != int(roster["cells"]):
    sys.exit(f"runtime holds {len(cells)} roster cells, roster declares {roster['cells']}")
try:
    alive = set(subprocess.run(
        ["squeue", "-h", "-o", "%i"], capture_output=True, text=True, timeout=30,
    ).stdout.split())
except Exception:
    alive = None  # cannot verify: treat every claim as live (fail safe)
pending, done, stranded, blocked, claimed, finishable = [], [], [], [], [], []
for cell in cells:
    root = runtime / cell
    state = json.loads((root / "campaign_state.json").read_text())
    phase = state.get("phase")
    session = root / "agent_session.json"
    session_status = None
    if session.is_file():
        try:
            session_status = json.loads(session.read_text()).get("status")
        except (OSError, ValueError):
            session_status = "unreadable"
    if phase == "certified" or (
        phase == "winner-frozen" and session_status == "finalized"
    ):
        done.append(cell)
        continue
    claim = root / ".discovery_claim"
    if claim.is_file():
        holder = claim.read_text().strip()
        if alive is not None and holder not in alive and holder != job_id:
            # Stale claim from a dead job: classify as pending and leave the
            # file alone — take_claim() is the ONE place that may replace a
            # claim (compare-and-swap under flock). Unlinking here could
            # delete a claim another chain just legitimately stole.
            pass
        else:
            claimed.append(cell)
            continue
    journal = root / "automil" / ".activity.jsonl"
    journal_text = journal.read_text() if journal.is_file() else ""
    has_evidence = session.is_file() or bool(journal_text.strip())
    if has_evidence:
        # Session evidence with an unfinished ladder. If the session ENDED
        # cleanly after a full 30-attempt budget, the finish ladder is
        # idempotent and safely resumable — queue a finish-only pass.
        # Anything else needs a human (live elsewhere, or stranded).
        session_ended = any(
            json.loads(line).get("event") == "session_end"
            for line in journal_text.splitlines()
            if line.strip().startswith("{")
        )
        charged = (state.get("discovery") or {}).get("attempts_charged")
        if session_ended and charged == 30:
            finishable.append(cell)
        else:
            stranded.append(cell)
        continue
    reproduction = state.get("baseline_reproduction") or {}
    if reproduction.get("mode") == "gate" and reproduction.get("verdict") == "fail":
        blocked.append(cell)
        continue
    if state.get("baseline") is None:
        sys.exit(f"{cell}: no registered baseline — discovery cannot start")
    pending.append(cell)
print(json.dumps({
    "pending": pending, "done": done, "stranded": stranded,
    "blocked": blocked, "claimed": claimed, "finishable": finishable,
}))
PYEOF
}

SCAN=$(scan_cells) || { echo "ERROR: cell scan failed"; exit 1; }
# Queue entries are "<mode>:<cell>" — finish-only recoveries first (cheap,
# frees complete cells), then full discovery runs.
PENDING=$(echo "$SCAN" | uv run --frozen --no-sync python -c "
import json, sys
scan = json.load(sys.stdin)
rows = [f'finish:{c}' for c in scan['finishable']]
rows += [f'full:{c}' for c in scan['pending']]
print('\n'.join(rows))")
echo "scan: $(echo "$SCAN" | uv run --frozen --no-sync python -c \
    "import json,sys;d=json.load(sys.stdin);print(', '.join(f'{k}={len(v)}' for k,v in d.items()))")"
for class in stranded blocked; do
    echo "$SCAN" | uv run --frozen --no-sync python -c \
        "import json,sys;[print('  $class:', c) for c in json.load(sys.stdin)['$class']]"
done

QUEUE_FILE=$(mktemp)
FAIL_FILE=$(mktemp)
echo "$PENDING" > "$QUEUE_FILE"
trap 'rm -f "$QUEUE_FILE" "$FAIL_FILE"' EXIT

# USR1 here means CELL_WALL_BUDGET_H was mis-sized: sessions may be live and
# will be killed by the wall — record which, for the operator. NO resubmit
# from the signal path: chaining happens only at a clean end.
_usr1_report() {
    echo "[signal] CRITICAL: wall reached with cells possibly mid-session:"
    ls "$RUNTIME"/*/.discovery_claim 2>/dev/null | sed 's/^/  /'
    echo "  these cells may be PERMANENTLY stranded (no session relaunch exists)"
}
trap _usr1_report USR1

remaining_hours() {
    local end
    end=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %e 2>/dev/null | head -1)
    if [ -z "$end" ] || [ "$end" = "N/A" ]; then echo 0; return; fi
    echo $(( ($(date -d "$end" +%s) - $(date +%s)) / 3600 ))
}

pop_cell() {
    (
        flock 9
        head -n 1 "$QUEUE_FILE"
        sed -i '1d' "$QUEUE_FILE"
    ) 9>>"$QUEUE_FILE.lock"
}

# Atomically take a cell for this job, or refuse. O_EXCL (noclobber) is the
# single enforcement point that makes concurrent chains safe: every queue is
# built from a scan-time snapshot, so two chains WILL both queue the same
# cell — only one may ever drive it. A dead holder's claim is stolen once
# (its cell was already re-scanned or the deep session guards will refuse);
# a live holder's cell is skipped silently, which is normal contention.
take_claim() {
    local cell="$1" claim="$RUNTIME/$1/.discovery_claim" me holder
    me="${SLURM_JOB_ID:-manual}"
    if ( set -C; echo "$me" > "$claim" ) 2>/dev/null; then
        return 0
    fi
    holder=$(cat "$claim" 2>/dev/null)
    [ "$holder" = "$me" ] && return 0
    # Steal only when a SUCCESSFUL full queue listing omits the holder — a
    # scheduler hiccup must never read as "holder dead" (a fail-open steal
    # from a live chain is exactly the race this function exists to close),
    # and probing the job id directly exits nonzero for purged jobs, which
    # would make genuinely-dead holders unreapable. The replacement itself
    # is a compare-and-swap under flock: re-read the holder under the lock
    # and replace only the exact dead holder we verified — a bare
    # rm-then-create lets several stealers each clobber the previous
    # winner's fresh claim in sequence.
    local alive_list
    if [ -n "$holder" ] && alive_list=$(squeue -h -o %i 2>/dev/null) \
        && ! echo "$alive_list" | grep -qx "$holder"; then
        (
            flock -n 9 || exit 1
            [ "$(cat "$claim" 2>/dev/null)" = "$holder" ] || exit 1
            rm -f "$claim"
            ( set -C; echo "$me" > "$claim" ) 2>/dev/null
        ) 9>>"$claim.lock" && return 0
    fi
    return 1
}

stage() {
    uv run --frozen --no-sync --package autobench \
        python benchmarks/scripts/campaign_stage.py "$@"
}

operate() {
    uv run --frozen --no-sync --package autobench \
        python benchmarks/scripts/campaign_operate.py "$@"
}

cell_status_json() {
    stage status --cell-root "$RUNTIME/$1" 2>/dev/null
}

# Poll until discovery is complete (30/30 charged, no queued or running
# specs, twice in a row), the agent window dies, or the per-cell deadline
# passes. Echoes the outcome.
watch_discovery() {
    local cell="$1" name="$2" deadline="$3" stable=0 verdict=""
    while :; do
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "deadline"
            return 1
        fi
        verdict=$(uv run --frozen --no-sync --package autobench python - \
            "$RUNTIME/$cell" <<'PYEOF'
import json, subprocess, sys
from pathlib import Path
root = Path(sys.argv[1])
status = json.loads(subprocess.run(
    [sys.executable, "benchmarks/scripts/campaign_stage.py",
     "status", "--cell-root", str(root)],
    capture_output=True, text=True, check=True,
).stdout)
adir = root / "automil"
queued = list(adir.glob("orchestrator/queue/*.json"))
running = list(adir.glob("orchestrator/running/**/*.json"))
charged = (status.get("discovery") or {}).get("attempts_charged")
complete = charged == 30 and not queued and not running
print("complete" if complete else f"working charged={charged} q={len(queued)} r={len(running)}")
PYEOF
        ) || verdict="status-error"
        if ! tmux list-windows -t "=$name" 2>/dev/null | grep -q "agent"; then
            echo "agent-window-dead"
            return 1
        fi
        if [ "$verdict" = "complete" ]; then
            stable=$((stable + 1))
            [ "$stable" -ge 2 ] && { echo "complete"; return 0; }
        else
            stable=0
        fi
        sleep "$WATCH_INTERVAL_S"
    done
}

# End the session the way the runbook's operator does: /exit into the agent
# window, then wait for SessionEnd evidence (exporter port free + journal
# session_end). One resend, then give up loudly.
end_session() {
    local cell="$1" name="$2" attempt evidence
    for attempt in 1 2; do
        tmux send-keys -t "=$name:agent" -l "/exit" 2>/dev/null
        tmux send-keys -t "=$name:agent" Enter 2>/dev/null
        for _ in $(seq 1 30); do
            sleep 30
            evidence=$(uv run --frozen --no-sync --package autobench python - \
                "$RUNTIME/$cell" <<'PYEOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
journal = root / "automil" / ".activity.jsonl"
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
    local gpu="$1" mode="$2" cell="$3" name log rc force_flag=""
    log="logs/discovery_cells/${cell}.log"
    if ! take_claim "$cell"; then
        echo "[gpu$gpu] $cell is claimed by another live chain — skipping"
        return 0
    fi
    echo "[gpu$gpu] $(date +%m-%d\ %H:%M) start $cell (mode=$mode)"

    if [ "$mode" = "finish" ]; then
        # Recovery lane: the session already ended after a full budget; the
        # finish ladder is idempotent and just needs a GPU for promotion.
        if operate finish "$RUNTIME/$cell" --gpu "$gpu" >> "$log" 2>&1; then
            rm -f "$RUNTIME/$cell/.discovery_claim"
            echo "[gpu$gpu] $(date +%m-%d\ %H:%M) DONE $cell (finish-only)"
            return 0
        fi
        echo "$cell finish-failed (see $log)" >> "$FAIL_FILE"
        rm -f "$RUNTIME/$cell/.discovery_claim"
        return 1
    fi

    # 1. Reproduction gate (gate mode). Measurement-mode blocks from the
    #    epsilon derivation are superseded exactly once, auditable.
    if uv run --frozen --no-sync --package autobench python -c "
import json,sys
b=(json.load(open('$RUNTIME/$cell/campaign_state.json')).get('baseline_reproduction') or {})
sys.exit(0 if b.get('mode')=='measurement' else 1)" 2>/dev/null; then
        force_flag="--force"
    fi
    if ! stage run-baseline-reproduction --cell-root "$RUNTIME/$cell" \
            --gpu "$gpu" $force_flag >> "$log" 2>&1; then
        if grep -q "reproduction FAILED" "$log"; then
            echo "$cell gate-FAILED (drift recorded; discovery blocked)" >> "$FAIL_FILE"
        else
            echo "$cell reproduction-error (see $log)" >> "$FAIL_FILE"
        fi
        rm -f "$RUNTIME/$cell/.discovery_claim"
        return 1
    fi

    # 2-4. Daemon + session + bind, via the canonical operator.
    rm -rf "$HOME/.claude/plugins" 2>/dev/null || true
    local name_and_release
    name_and_release=$(uv run --frozen --no-sync --package autobench python -c "
import importlib.util
spec=importlib.util.spec_from_file_location('op','benchmarks/scripts/campaign_operate.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from pathlib import Path
print(m._session_name(Path('$RUNTIME/$cell')))
print(m.RELEASE_LINE)")
    name=$(echo "$name_and_release" | sed -n 1p)
    local release_line
    release_line=$(echo "$name_and_release" | sed -n 2p)
    for step in "up $RUNTIME/$cell --gpu $gpu" "launch $RUNTIME/$cell" \
                "bind $RUNTIME/$cell --timeout-s 900"; do
        if ! operate $step >> "$log" 2>&1; then
            echo "$cell operate-${step%% *}-failed (see $log)" >> "$FAIL_FILE"
            rm -f "$RUNTIME/$cell/.discovery_claim"
            return 1
        fi
    done
    # The agent idles until the operator's release line arrives as its
    # first message (runbook: paste exactly that line). This launcher IS
    # the operator.
    sleep 5
    tmux send-keys -t "=$name:agent" -l "$release_line"
    tmux send-keys -t "=$name:agent" Enter
    echo "[gpu$gpu] $(date +%m-%d\ %H:%M) session bound + released for $cell (tmux $name)"

    # 5. Watch until the agent has spent its 30 attempts and gone idle. The
    #    deadline leaves room for /exit + finish inside the cell budget.
    local outcome deadline
    deadline=$(( $(date +%s) + (CELL_WALL_BUDGET_H - 14) * 3600 ))
    outcome=$(watch_discovery "$cell" "$name" "$deadline")
    if [ "$outcome" = "deadline" ]; then
        # Graceful strand: end the session so its final active-time sample
        # is captured; under-budget cells cannot freeze, so a human owns it.
        end_session "$cell" "$name" || true
        echo "$cell budget-deadline-STRANDED (session ended for the ledger; <30 attempts) — OPERATOR NEEDED" >> "$FAIL_FILE"
        return 1
    fi
    if [ "$outcome" != "complete" ]; then
        echo "$cell session-died-mid-discovery ($outcome) — OPERATOR NEEDED, cell may be stranded" >> "$FAIL_FILE"
        # Claim stays: nothing may touch this cell without a human.
        return 1
    fi

    # 6. Operator-equivalent session end, then the finish ladder (freeze ->
    #    promotion on this GPU -> winner -> finalize).
    if ! end_session "$cell" "$name"; then
        echo "$cell no-session-end-after-exit — OPERATOR NEEDED" >> "$FAIL_FILE"
        return 1
    fi
    if ! operate finish "$RUNTIME/$cell" --gpu "$gpu" >> "$log" 2>&1; then
        echo "$cell finish-failed (see $log)" >> "$FAIL_FILE"
        rm -f "$RUNTIME/$cell/.discovery_claim"
        return 1
    fi
    tmux kill-session -t "=$name" 2>/dev/null || true
    rm -f "$RUNTIME/$cell/.discovery_claim"
    echo "[gpu$gpu] $(date +%m-%d\ %H:%M) DONE $cell (winner finalized)"
    return 0
}

worker() {
    local gpu="$1" entry mode cell hours
    while :; do
        hours=$(remaining_hours)
        if [ "$hours" -le "$CELL_WALL_BUDGET_H" ]; then
            echo "[gpu$gpu] ${hours}h wall left < ${CELL_WALL_BUDGET_H}h budget — no new cells"
            break
        fi
        entry=$(pop_cell)
        [ -n "$entry" ] || break
        mode="${entry%%:*}"
        cell="${entry#*:}"
        run_cell "$gpu" "$mode" "$cell" || true
    done
}

echo "================================================"
echo "preprint campaign DISCOVERY — roster: $ROSTER"
echo "Job ${SLURM_JOB_ID:-N/A} | $(hostname) | $(date)"
echo "claude $OBSERVED_CLAUDE == protocol $PROTOCOL_VERSION_PIN | cell budget ${CELL_WALL_BUDGET_H}h"
echo "================================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>/dev/null || true

if [ ! -s "$QUEUE_FILE" ]; then
    echo "No pending discovery cells."
    [ -s "$FAIL_FILE" ] && { sed 's/^/  /' "$FAIL_FILE"; exit 1; }
    exit 0
fi

for gpu in $(seq 0 $((N_GPUS - 1))); do
    worker "$gpu" &
done
while [ -n "$(jobs -pr)" ]; do
    wait -n || true
done

echo "---"
LEFT=$(grep -c . "$QUEUE_FILE" 2>/dev/null || echo 0)
if [ -s "$FAIL_FILE" ]; then
    echo "Cells needing attention:"; sed 's/^/  /' "$FAIL_FILE"
fi
if [ "$LEFT" -gt 0 ]; then
    echo "$LEFT pending cells remain — chaining the next job."
    for attempt in 1 2 3; do
        sbatch --parsable "$SELF" && break
        echo "  sbatch failed (attempt $attempt)"; sleep 20
    done
fi
[ -s "$FAIL_FILE" ] && exit 1
echo "All attempted cells finished cleanly. $(date)"
