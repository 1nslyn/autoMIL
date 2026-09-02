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
DISC_NUDGE_LINE="Operator note: no agent activity has been observed for 30 minutes while attempts remain. Continue per your policy."
DISC_NUDGE_IDLE_S=1800
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
    export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$PROJECT_DIR"
    # The frozen toolset forbids these; a stray module could have set them.
    unset ANTHROPIC_MODEL ANTHROPIC_SMALL_FAST_MODEL ANTHROPIC_BASE_URL \
          CLAUDE_CODE_USE_BEDROCK CLAUDE_CODE_USE_VERTEX CLAUDE_CODE_EFFORT_LEVEL \
          2>/dev/null || true
    mkdir -p "$PROJECT_DIR/logs/discovery_cells"
}

pyrun() { uv run --frozen --no-sync --package autobench python "$@"; }
stage() { pyrun benchmarks/scripts/campaign_stage.py "$@"; }
operate() { pyrun benchmarks/scripts/campaign_operate.py "$@"; }
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
    if [ -d "$HOME/.claude/plugins" ] && [ -n "$(ls -A "$HOME/.claude/plugins" 2>/dev/null)" ]; then
        echo "ERROR: $HOME/.claude/plugins is not empty — move it aside for the campaign"; return 1
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

# Every file the owner leaves behind becomes group read/write so a later
# member can finish or scan the cell. Owner-only by construction; errors on
# files we do not own are expected and ignored.
normalize_cell_modes() {
    chmod -R g+rwX "$RUNTIME/$1" 2>/dev/null || true
}
