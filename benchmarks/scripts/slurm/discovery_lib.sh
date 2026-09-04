#!/bin/bash
# Shared library for the preprint discovery campaign launchers
# (submit_discovery_cell.sh = member entry point / chain step,
#  submit_discovery_campaign.sh = the per-cell SLURM job).
#
# One definition point for: campaign paths, the shared-tree environment
# (umask, uv/git/tmux isolation), the pinned-runtime and memory-surface
# preflight, the cell scan, the once-only claim protocol, the operator
# message lines, and the stage/operate wrappers. Nothing here submits or
# runs a cell.
#
# Source it with DISC_PROJECT_DIR already exported (the wrapper derives it
# from its own path; the job receives it through sbatch --export, because a
# spooled batch script no longer knows where it came from).

set -uo pipefail

DISC_ACCOUNT_DEFAULT="def-jma-ab"
DISC_CAMPAIGN_REL="benchmarks/campaigns/preprint_130"
# Operator lines. RELEASE_LINE is authoritative in campaign_operate.py and is
# imported from there at run time; the nudge is the launcher's own, sent only
# when the runtime's active time has been flat for NUDGE_IDLE_S while the
# queue is drained and attempts remain (see watch_discovery).
DISC_NUDGE_IDLE_S=1800
DISC_NUDGE_LINE="Operator note: no agent activity has been observed for $((DISC_NUDGE_IDLE_S / 60)) minutes while attempts remain. Continue per your policy."
DISC_NUDGE_MIN_GAP_S=3600
DISC_NUDGE_MAX=3
DISC_WEEKLY_USAGE_MAX_PCT=85
DISC_WATCH_INTERVAL_S=120
# Wall reserved after discovery completes for /exit + the finish ladder
# (freeze -> promotion on this job's GPUs -> winner -> finalize).
DISC_FINISH_RESERVE_H=4

disc_paths() {
    PROJECT_DIR="${DISC_PROJECT_DIR:?DISC_PROJECT_DIR must point at the campaign checkout}"
    CAMPAIGN_DIR="$PROJECT_DIR/$DISC_CAMPAIGN_REL"
    RUNTIME="$CAMPAIGN_DIR/runtime"
    ROSTER="$CAMPAIGN_DIR/active_roster.json"
    [ -d "$RUNTIME" ] || { echo "ERROR: runtime not materialized: $RUNTIME"; return 1; }
    [ -f "$ROSTER" ] || { echo "ERROR: active roster missing: $ROSTER"; return 1; }
    [ -f "$PROJECT_DIR/benchmarks/.env" ] || { echo "ERROR: $PROJECT_DIR/benchmarks/.env missing"; return 1; }
    cd "$PROJECT_DIR" || return 1
}

# The shared tree is group-owned: every file this process creates must be
# group read/write, uv must never sync the shared venv, git must accept a
# checkout owned by another member, and tmux must never reuse another job's
# server (which would leak that job's environment into new windows).
disc_env() {
    umask 007
    set -a; source "$PROJECT_DIR/benchmarks/.env"; set +a
    export DISABLE_AUTOUPDATER=1
    export UV_FROZEN=1 UV_NO_SYNC=1
    # The project-space tree keeps its Python environment in scratch (inode
    # quota); benchmarks/.env names it as UV_PROJECT_ENVIRONMENT, sourced above.
    [ -n "${UV_PROJECT_ENVIRONMENT:-}" ] && [ -x "$UV_PROJECT_ENVIRONMENT/bin/python" ] \
        || { echo "ERROR: UV_PROJECT_ENVIRONMENT (from benchmarks/.env) does not name a usable environment"; return 1; }
    export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$PROJECT_DIR"
    # The frozen toolset forbids these; a stray module could have set them.
    unset ANTHROPIC_MODEL ANTHROPIC_SMALL_FAST_MODEL ANTHROPIC_BASE_URL \
          CLAUDE_CODE_USE_BEDROCK CLAUDE_CODE_USE_VERTEX CLAUDE_CODE_EFFORT_LEVEL \
          2>/dev/null || true
    mkdir -p "$PROJECT_DIR/logs/discovery_cells"
}

# Every tmux call in the launchers goes through tmx so the job-private
# server (AUTOMIL_TMUX_SOCKET, honored by campaign_operate too) is the one
# enforcement point: a bare `tmux` would talk to the default server and
# never find the session campaign_operate created.
tmx() { tmux ${AUTOMIL_TMUX_SOCKET:+-L "$AUTOMIL_TMUX_SOCKET"} "$@"; }
pyrun() { uv run --frozen --no-sync --package autobench python "$@"; }
stage() { pyrun benchmarks/scripts/campaign_stage.py "$@"; }
operate() { pyrun benchmarks/scripts/campaign_operate.py "$@"; }

# The pinned runtime reports NO active-time metric while the agent is idle,
# and bind requires one recorded sample before it can complete. A typed and
# deleted character in the input line makes the runtime report activity
# (about one second) without submitting anything: no message reaches the
# model and nothing enters the transcript. Verified on 2.1.228 (fir login
# node): idle for minutes = no metric; "x" + Backspace = metric within 20 s.
wake_runtime() {  # tmux-session-name
    tmx send-keys -t "=$1:agent" -l "x"; sleep 1; tmx send-keys -t "=$1:agent" BSpace
}

# Poll the cell's own exporter until it serves (the runtime starts it a few
# seconds after launch); bind scrapes it through the daemon.
wait_exporter() {  # cell_root [timeout_s]
    local port waited=0 limit="${2:-180}"
    port=$(pyrun -c "import yaml,sys;print((yaml.safe_load(open('$1/automil/config.yaml')).get('activity') or {}).get('exporter_port', 9464))") || return 1
    while [ "$waited" -lt "$limit" ]; do
        curl -s -m 2 "http://127.0.0.1:$port/metrics" >/dev/null 2>&1 && { echo "exporter serving on $port after ${waited}s"; return 0; }
        sleep 5; waited=$((waited + 5))
    done
    echo "exporter on $port not serving within ${limit}s"; return 1
}

# `up` only sends the orchestrator start command into a tmux window; the
# daemon needs seconds to import, prune worktrees, recover orphans and write
# its pid file, and `launch` refuses until it is alive. Wait for it here.
wait_daemon() {  # cell_root [timeout_s]
    local root="$1" limit="${2:-300}" waited=0 pid=""
    while [ "$waited" -lt "$limit" ]; do
        pid=$(pyrun - "$root" <<'PYEOF'
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("op", "benchmarks/scripts/campaign_operate.py")
m = importlib.util.module_from_spec(spec); sys.modules["op"] = m; spec.loader.exec_module(m)
pid = m._daemon_alive(Path(sys.argv[1]) / "automil" / "orchestrator")
print(pid if pid else "")
PYEOF
        )
        [ -n "$pid" ] && { echo "orchestrator daemon alive (pid $pid) after ${waited}s"; return 0; }
        sleep 5; waited=$((waited + 5))
    done
    echo "orchestrator daemon did not come up within ${limit}s"; return 1
}
# Shape preference: "cheap" = fewest GPU-hours (smallest GPU count that fits
# either wall), "fast" = shortest wall first. Fair-share bills GPU-minutes,
# so cheap is the default; DISC_PREFER=fast trades GPU-hours for queue time.
DISC_PREFER="${DISC_PREFER:-cheap}"
shape_field() { pyrun benchmarks/scripts/campaign_shape.py --runtime "$RUNTIME" --prefer "$DISC_PREFER" --cell "$1" --field "$2"; }

disc_scan() {
    pyrun benchmarks/scripts/campaign_scan.py --runtime "$RUNTIME" --roster "$ROSTER" \
        --job-id "${SLURM_JOB_ID:-manual}" "$@"
}

# Preflight that needs no GPU and no session: pinned runtime on PATH, a login
# on this account, a clean instruction surface on every path the runtime
# reads memory from (its own home, and the shared path from the runtime root
# up to /), and every sibling session record readable (open-agent-session
# refuses otherwise, AFTER the session exists).
disc_static_preflight() {
    local pin observed
    command -v uv >/dev/null 2>&1 || { echo "ERROR: uv is required"; return 1; }
    command -v tmux >/dev/null 2>&1 || { echo "ERROR: tmux is required"; return 1; }
    command -v claude >/dev/null 2>&1 || { echo "ERROR: claude is not on PATH"; return 1; }
    pin=$(pyrun -c "import json;print(json.load(open('$RUNTIME/agent_protocol.json'))['runtime_version'])") \
        || { echo "ERROR: cannot read protocol runtime_version"; return 1; }
    observed=$(claude --version 2>/dev/null | awk '{print $1}')
    [ "$observed" = "$pin" ] || { echo "ERROR: claude on PATH is ${observed:-absent}, protocol pins $pin"; return 1; }
    [ -f "$HOME/.claude/.credentials.json" ] || {
        echo "ERROR: no Claude login on this account — run 'claude login' on a login node"; return 1; }
    [ ! -e "$HOME/.claude/CLAUDE.md" ] || { echo "ERROR: remove $HOME/.claude/CLAUDE.md (user memory must be absent)"; return 1; }
    # The runtime recreates ~/.claude/plugins (marketplace metadata) on every
    # start, so the job clears it immediately before launch (clear_plugins);
    # here it is only reported.
    if [ -d "$HOME/.claude/plugins" ] && [ -n "$(ls -A "$HOME/.claude/plugins" 2>/dev/null)" ]; then
        echo "note: $HOME/.claude/plugins is not empty; the job clears it right before launch"
    fi
    local tree_group
    tree_group=$(stat -c %G "$PROJECT_DIR" 2>/dev/null)
    if [ -n "$tree_group" ] && ! id -nG | tr ' ' '\n' | grep -qx "$tree_group"; then
        echo "ERROR: you are not in group $tree_group that owns $PROJECT_DIR"; return 1
    fi
    pyrun - "$RUNTIME" "$PROJECT_DIR" <<'PYEOF' || return 1
import sys
from pathlib import Path
runtime, repo = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
pinned = repo / "CLAUDE.md"
bad = []
node = runtime
while True:
    for candidate in (node / "CLAUDE.md", node / "CLAUDE.local.md", node / ".claude" / "CLAUDE.md"):
        if candidate.is_file() and candidate != pinned:
            bad.append(str(candidate))
    if node == node.parent:
        break
    node = node.parent
for sibling in runtime.iterdir():
    record = sibling / "agent_session.json"
    if record.is_file():
        try:
            record.read_bytes()
        except OSError as exc:
            bad.append(f"{record} unreadable ({exc.strerror}); ask its owner to run chmod g+r")
if bad:
    print("ERROR: instruction/session surface not clean:\n  " + "\n  ".join(bad))
    sys.exit(1)
PYEOF
}

# Once-only claim. O_EXCL (noclobber) is the single enforcement point that
# makes concurrent submitters safe. Claims are never released within a
# holder's lifetime; a dead holder (absent from a SUCCESSFUL cluster-wide
# squeue listing) is replaced by compare-and-swap under flock. $2 overrides
# the holder id (the wrapper claims on behalf of the job it just submitted).
take_claim() {
    local cell="$1" me="${2:-${SLURM_JOB_ID:-manual}}"
    local claim="$RUNTIME/$cell/.discovery_claim" holder alive_list
    if ( set -C; echo "$me" > "$claim" ) 2>/dev/null; then
        return 0
    fi
    holder=$(cat "$claim" 2>/dev/null)
    [ "$holder" = "$me" ] && return 0
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

claim_holder() { cat "$RUNTIME/$1/.discovery_claim" 2>/dev/null; }

remaining_hours() {
    local end
    end=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %e 2>/dev/null | head -1)
    if [ -z "$end" ] || [ "$end" = "N/A" ]; then echo 0; return; fi
    echo $(( ($(date -d "$end" +%s) - $(date +%s)) / 3600 ))
}

# Read the plan's remaining allocation from a throwaway runtime in $HOME
# (/status is local to the runtime; no model turn). $1 = refuse|log: the
# wrapper refuses a submission above DISC_WEEKLY_USAGE_MAX_PCT before any
# claim exists; the job only records the value it saw. Unparsable output
# warns. $2 = output file.
# Claude's first start in a directory asks "Is this a project you trust?" and
# the first --dangerously-skip-permissions run asks for an acknowledgement;
# both are per-user config flags in ~/.claude.json. The operator answers them
# here, before launch, exactly as the runbook operator does by hand — the
# frozen instruction surface is untouched (user settings are not loaded:
# --setting-sources project).
trust_paths() {  # path...
    pyrun - "$HOME/.claude.json" "$@" <<'PYEOF'
import json, os, sys
path, roots = sys.argv[1], sys.argv[2:]
try:
    cfg = json.load(open(path))
except (OSError, ValueError):
    cfg = {}
cfg["bypassPermissionsModeAccepted"] = True
projects = cfg.setdefault("projects", {})
for root in roots:
    entry = projects.setdefault(os.path.realpath(root), {})
    entry["hasTrustDialogAccepted"] = True
tmp = path + ".tmp"
with open(tmp, "w") as fh:
    json.dump(cfg, fh, indent=2)
os.replace(tmp, path)
PYEOF
}

# The /usage panel prints "Current week (all models)" and, on the next line,
# "<N>% used" (verified on 2.1.228).
usage_weekly_pct() {  # capture-file
    awk '/Current week/ {flag=1; next} flag && /% used/ {match($0, /[0-9]+%/); print substr($0, RSTART, RLENGTH-1); exit}' "$1"
}

# A throwaway runtime in a private probe directory reads /usage (the plan's
# remaining allocation). Pane targets use window.pane indexes: fir's tmux
# 3.3a rejects the bare "=session" form for pane commands.
disc_usage_probe() {
    local mode="$1" out="$2" pct probe_dir="$HOME/.automil_probe"
    mkdir -p "$probe_dir" && trust_paths "$probe_dir"
    tmx kill-session -t "usage_probe" 2>/dev/null || true
    if ! tmx new-session -d -s usage_probe -c "$probe_dir" -x 200 -y 50 2>/dev/null; then
        echo "WARNING: could not start the usage probe (tmux); weekly window unchecked"
        return 0
    fi
    tmx send-keys -t "usage_probe:0.0" -l "claude --setting-sources project --strict-mcp-config"
    tmx send-keys -t "usage_probe:0.0" Enter
    sleep 20
    tmx send-keys -t "usage_probe:0.0" -l "/usage"; tmx send-keys -t "usage_probe:0.0" Enter
    sleep 12
    tmx capture-pane -p -t "usage_probe:0.0" -S -80 > "$out" 2>/dev/null
    tmx kill-session -t "usage_probe" 2>/dev/null || true
    pct=$(usage_weekly_pct "$out")
    if [ -z "$pct" ]; then
        echo "WARNING: could not parse the weekly usage window from /status (see $out)"
        return 0
    fi
    echo "usage window: weekly ${pct}% used"
    if [ "$mode" = refuse ] && [ "$pct" -ge "$DISC_WEEKLY_USAGE_MAX_PCT" ]; then
        echo "ERROR: weekly usage ${pct}% >= ${DISC_WEEKLY_USAGE_MAX_PCT}% — not submitting a cell on this seat now"
        return 1
    fi
    return 0
}

# The launch preflight refuses a non-empty ~/.claude/plugins; the runtime
# recreates marketplace metadata there on each start, so clear it right
# before the formal launch (the toolset pins plugins absent).
clear_plugins() { rm -rf "$HOME/.claude/plugins"; }

# Every file the owner leaves behind becomes group read/write so a later
# member can finish or scan the cell. Owner-only by construction; errors on
# files we do not own are expected and ignored.
normalize_cell_modes() {
    chmod -R g+rwX "$RUNTIME/$1" 2>/dev/null || true
}
