#!/usr/bin/env bash
# Hook: prevent agent from stopping while autoMIL loop is active.
# Exit 1 = prevent stop. Exit 0 = allow stop.

# Find project root by walking up
DIR="$PWD"
while [ "$DIR" != "/" ]; do
    if [ -f "$DIR/.automil_active" ]; then
        echo "autoMIL loop is active. Run 'automil stop-loop' to allow stopping."
        echo ""
        echo "Resume instructions:"
        echo "  1. Read config.yaml, graph.json, learnings.md, program.md"
        echo "  2. Run: uv run automil reconcile"
        echo "  3. Continue the experiment loop"
        exit 1
    fi
    DIR="$(dirname "$DIR")"
done

exit 0
