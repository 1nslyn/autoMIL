#!/usr/bin/env bash
# autoMIL activity hook — registered (async) for PostToolUse / UserPromptSubmit /
# SessionStart by `automil init`.
#
# Stamps automil/.last_action_at on EVERY agent action so the orchestrator can
# bill only agent-active time: the agent emits a dense stream of tool calls while
# working (reads, edits, research, submits) and is quiescent — no tool calls —
# while waiting on experiments. The daemon pauses the budget clock when this
# marker goes stale (see automil.cells.cap.accrue_active).
#
# Must be fast and must NEVER block or fail the agent. Registered async, so its
# exit code is ignored; it always exits 0.

# Drain the hook payload from stdin (Claude Code delivers JSON there) and discard
# it — we only need the fact that a tool ran, not its contents.
cat >/dev/null 2>&1 || true

# Walk up from the agent's cwd to the automil/ overlay and stamp the marker.
dir="$PWD"
while [ "$dir" != "/" ]; do
    if [ -f "$dir/automil/config.yaml" ]; then
        date +%s.%N > "$dir/automil/.last_action_at" 2>/dev/null || true
        break
    fi
    dir="$(dirname "$dir")"
done

exit 0
