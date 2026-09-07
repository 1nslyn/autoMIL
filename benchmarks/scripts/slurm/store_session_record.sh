#!/bin/bash
# Store the runtime's own record of a cell's agent session in the cell root.
#
# Claude Code writes every session's transcript (each user, assistant and
# tool message) to ~/.claude/projects/<cwd-slug>/<session-id>.jsonl, with a
# sidecar directory <session-id>/ holding subagent transcripts and fetched
# tool results, in the HOME of the user who ran it, and prunes them after
# its cleanup period (30 days by default). The cell root is the campaign's
# record, so this copies both into <cell-root>/operator/session/ for every
# session the cell's activity journal bound. Idempotent; group-readable.
#
# Usage: store_session_record.sh <cell-root>
# The discovery job runs it on exit (any outcome); run it by hand for a
# cell whose job predates it.
set -u
ROOT="${1:?usage: store_session_record.sh <cell-root>}"
JOURNAL="$ROOT/automil/.activity.jsonl"
[ -s "$JOURNAL" ] || { echo "store_session_record: no activity journal in $ROOT"; exit 1; }
DEST="$ROOT/operator/session"
rc=0
found=0
# The journal is written with sorted keys by campaign_operate, one event per line.
for sid in $(grep '"event":"session_bind"' "$JOURNAL" | grep -oE '"session_id":"[0-9a-f-]+"' | cut -d'"' -f4 | awk '!seen[$0]++'); do
    found=1
    src=$(find "$HOME/.claude/projects" -maxdepth 2 -name "$sid.jsonl" 2>/dev/null | head -1)
    if [ -z "$src" ]; then
        echo "store_session_record: transcript $sid.jsonl not found under $HOME/.claude/projects"; rc=1; continue
    fi
    mkdir -p "$DEST" || { rc=1; continue; }
    cp -p "$src" "$DEST/" || { echo "store_session_record: copy of $sid.jsonl failed"; rc=1; continue; }
    if [ -d "${src%.jsonl}" ]; then
        cp -rp "${src%.jsonl}" "$DEST/" || { echo "store_session_record: copy of the $sid sidecar failed"; rc=1; }
    fi
    chmod -R g+rX "$DEST"
    echo "store_session_record: $sid -> $DEST ($(wc -l < "$src") transcript lines, $(find "${src%.jsonl}" -type f 2>/dev/null | wc -l) sidecar files)"
done
[ "$found" = 1 ] || { echo "store_session_record: no bound session in $JOURNAL"; exit 1; }
exit $rc
