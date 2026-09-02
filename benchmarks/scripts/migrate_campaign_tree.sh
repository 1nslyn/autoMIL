#!/bin/bash
# One-time, idempotent move of the preprint campaign execution tree into
# group-shared project space, so five members can drive cells from their own
# accounts. Run as the current owner of the scratch tree (Leo), while NO
# campaign job is running.
#
#   migrate_campaign_tree.sh --source ~/scratch/autoMIL --dest /project/6114359/shared/Pathology/autoMIL/work \
#                            --group rrg-jma --commit <sha> [--verify-only]
#
# Steps (each skipped when already done):
#   1. dest/autoMIL: git clone of the campaign repo at --commit, .venv via
#      `uv sync --frozen --all-packages`, benchmarks/.env rewritten so every
#      AUTOBENCH_*_ROOT points at dest/guard_roots/<cohort>;
#   2. the 78 cell roots + runtime/agent_protocol.json + logs/ rsync'd
#      (additive, never --delete) and verified by sha256 of every
#      campaign_state.json;
#   3. dest/guard_roots rebuilt by build_guard_root_shims.py --i-know-idle;
#   4. group + modes: chgrp -R, dirs 2770 (setgid so new files inherit the
#      group), files g+rw; `git worktree prune`;
#   5. verification: restart-safe materialize reports every root bound to
#      identical inputs, the scan sees 78 pending, the training tree is
#      clean, CLAUDE.md matches the protocol's pinned hash.
# The scratch tree is left in place; rename it yourself once the first cell
# has finished in the new tree.

set -euo pipefail
SOURCE=""; DEST=""; GROUP="rrg-jma"; COMMIT=""; VERIFY_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --source) SOURCE="$2"; shift ;;
        --dest) DEST="$2"; shift ;;
        --group) GROUP="$2"; shift ;;
        --commit) COMMIT="$2"; shift ;;
        --verify-only) VERIFY_ONLY=1 ;;
        *) echo "unknown option: $1"; exit 2 ;;
    esac
    shift
done
[ -n "$SOURCE" ] && [ -n "$DEST" ] || { echo "usage: --source <scratch/autoMIL> --dest <work> [--group G] --commit <sha>"; exit 2; }
SRC_REPO="$SOURCE/autoMIL"
DST_REPO="$DEST/autoMIL"
CAMPAIGN_REL="benchmarks/campaigns/preprint_130"
[ -d "$SRC_REPO/$CAMPAIGN_REL/runtime" ] || { echo "ERROR: no campaign runtime under $SRC_REPO"; exit 1; }
if squeue -h -u "$USER" -o %j 2>/dev/null | grep -qE '^(disc_|bl_campaign|repro_)'; then
    echo "ERROR: a campaign job is queued or running under $USER — migrate only while idle"; exit 1
fi
umask 007
step() { echo; echo "== $*"; }

if [ "$VERIFY_ONLY" = 0 ]; then
    step "1. checkout + venv + .env"
    [ -n "$COMMIT" ] || { echo "ERROR: --commit is required for the clone"; exit 1; }
    mkdir -p "$DEST"; chgrp "$GROUP" "$DEST"; chmod 2770 "$DEST"
    if [ ! -d "$DST_REPO/.git" ]; then
        git clone --quiet "$SRC_REPO" "$DST_REPO"
        git -C "$DST_REPO" remote set-url origin "$(git -C "$SRC_REPO" remote get-url origin)"
    fi
    git -C "$DST_REPO" fetch --quiet origin
    git -C "$DST_REPO" checkout --quiet "$COMMIT"
    (cd "$DST_REPO" && uv sync --frozen --all-packages)
    python3 - "$SRC_REPO/benchmarks/.env" "$DST_REPO/benchmarks/.env" "$DEST/guard_roots" <<'PYEOF'
import re, sys
from pathlib import Path
src, dst, roots = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
lines = []
for line in src.read_text().splitlines():
    m = re.match(r'^(AUTOBENCH_([A-Z0-9]+)_ROOT)=(.*)$', line)
    if m and m.group(2) != "EXPORT":
        line = f"{m.group(1)}={roots}/{m.group(2).lower()}"
    lines.append(line)
dst.write_text("\n".join(lines) + "\n")
print(f"wrote {dst}")
PYEOF

    step "2. cell roots + protocol + logs"
    mkdir -p "$DST_REPO/$CAMPAIGN_REL/runtime" "$DST_REPO/logs"
    rsync -a "$SRC_REPO/$CAMPAIGN_REL/runtime/" "$DST_REPO/$CAMPAIGN_REL/runtime/"
    rsync -a "$SRC_REPO/logs/" "$DST_REPO/logs/"
    (cd "$SRC_REPO/$CAMPAIGN_REL/runtime" && find . -name campaign_state.json -exec sha256sum {} + | sort) > /tmp/migrate_src.sha
    (cd "$DST_REPO/$CAMPAIGN_REL/runtime" && find . -name campaign_state.json -exec sha256sum {} + | sort) > /tmp/migrate_dst.sha
    diff /tmp/migrate_src.sha /tmp/migrate_dst.sha && echo "state files identical: $(wc -l < /tmp/migrate_dst.sha)"

    step "3. guard roots"
    (cd "$DST_REPO" && uv run --frozen --no-sync --package autobench python \
        benchmarks/scripts/build_guard_root_shims.py --guard-roots "$DEST/guard_roots" --i-know-idle)

    step "4. group + modes"
    chgrp -R "$GROUP" "$DEST"
    find "$DEST" -type d -exec chmod 2770 {} +
    find "$DEST" -type f -exec chmod g+rw {} +
    git -C "$DST_REPO" worktree prune
fi

step "5. verification"
cd "$DST_REPO"
uv run --frozen --no-sync --package autobench python benchmarks/scripts/campaign_manifest.py materialize \
    --agent-protocol "$CAMPAIGN_REL/agent_protocol.json" | tail -3
uv run --frozen --no-sync --package autobench python benchmarks/scripts/campaign_scan.py \
    --runtime "$CAMPAIGN_REL/runtime" --roster "$CAMPAIGN_REL/active_roster.json" \
    | python3 -c "import json,sys;d=json.load(sys.stdin);print('scan:',{k:len(d[k]) for k in ('pending','done','claimed','stranded','blocked','finishable')})"
git diff --quiet HEAD -- src benchmarks/src benchmarks/scripts && echo "training tree clean at $(git rev-parse --short HEAD)"
PIN=$(python3 -c "import json;print(json.load(open('$CAMPAIGN_REL/toolset.json'))['ancestor_memory']['CLAUDE.md'])")
[ "$(sha256sum CLAUDE.md | cut -c1-64)" = "$PIN" ] && echo "CLAUDE.md matches the pinned hash" || { echo "ERROR: CLAUDE.md drifted from the pinned hash"; exit 1; }
echo "done"
