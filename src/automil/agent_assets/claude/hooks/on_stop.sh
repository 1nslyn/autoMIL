#!/usr/bin/env bash
# Hook: trajectory recording + prevent agent from stopping while autoMIL loop is active.
# Claude Code delivers hook payload on stdin — HOOK_EVENT=$(cat) (NOT an env var).
# Exit 1 = prevent stop. Exit 0 = allow stop.
# Trajectory recording fires automil trajectory record (TRJ-04 / D-96).

# Read hook event from stdin (Claude Code hook delivery mechanism — D-95/D-96 CORRECTED)
HOOK_EVENT="$(cat)"

# Find project root by walking up
DIR="$PWD"
while [ "$DIR" != "/" ]; do
    if [ -f "$DIR/.automil_active" ]; then
        echo "autoMIL loop is active. Run 'automil stop-loop' to allow stopping."
        echo ""
        echo "Resume instructions:"
        echo "  1. Read config.yaml, graph.json, learnings.md, program.md"
        echo "  2. Run: uv run automil reconcile (or the consumer project's declared runner)"
        echo "  3. Continue the experiment loop"
        exit 1
    fi
    DIR="$(dirname "$DIR")"
done

# Trajectory recording — only fires if AUTOMIL_NODE_ID and AUTOMIL_RUNTIME are both set.
# These are set by the orchestrator before starting the agent session (not by Claude Code).
#
# WR-02 (Phase 3 review): Claude Code's Stop hook delivers a metadata payload
# (`{session_id, transcript_path, stop_hook_active}` per
# https://code.claude.com/docs/en/hooks) — NOT a `gen_ai.*` event. Forwarding
# the raw payload to `automil trajectory record` would fail schema validation
# every time and silently drop real stop events. Instead, wrap it in a
# `gen_ai.*` envelope with `event.name = stop_hook` so the recorder accepts
# it and the original payload is preserved under `gen_ai.event.payload`.
if [[ -n "${AUTOMIL_NODE_ID:-}" && -n "${AUTOMIL_RUNTIME:-}" && -n "$HOOK_EVENT" ]]; then
    # This shipped hook must run in consumer repos that may not use uv, so its
    # direct python3 call is a documented exception. python -c is the only
    # stdlib-portable way to embed a JSON payload as a value inside a JSON
    # envelope without quoting issues.
    ENVELOPE=$(AUTOMIL_HOOK_PAYLOAD="$HOOK_EVENT" \
               AUTOMIL_RUNTIME="$AUTOMIL_RUNTIME" \
               python3 -c '
import json, os, datetime
payload_raw = os.environ.get("AUTOMIL_HOOK_PAYLOAD", "")
runtime = os.environ.get("AUTOMIL_RUNTIME", "unknown")
try:
    payload = json.loads(payload_raw)
except Exception:
    payload = {"raw": payload_raw}
print(json.dumps({
    "gen_ai.provider.name": runtime,
    "gen_ai.event.name":    "stop_hook",
    "gen_ai.event.timestamp": datetime.datetime.utcnow().isoformat(timespec="microseconds") + "Z",
    "gen_ai.event.payload": payload,
}))
' 2>/dev/null)

    if [[ -n "$ENVELOPE" ]]; then
        # Runtime hooks must also work in non-uv consumer repos. The globally
        # installed console script is the documented exception to uv-first
        # source-repository commands.
        automil trajectory record "$ENVELOPE" \
            2>>"${AUTOMIL_DIR:-/tmp}/trajectory.err.log" || true
    fi
fi

exit 0
