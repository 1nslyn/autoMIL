#!/bin/bash
# One-time, idempotent move of the preprint campaign execution tree into
# group-shared project space, so five members can drive cells from their own
# accounts. Run as the current owner of the scratch tree (Leo), while NO
# campaign job is running.
#
#   migrate_campaign_tree.sh --source ~/scratch/autoMIL --dest /project/6114359/shared/Pathology/autoMIL/work \
#                            --group rrg-jma --commit <sha> --scratch /scratch/<owner>/autoMIL/work-scratch \
#                            [--verify-only]
#
# Project space on fir is INODE-limited (500K files per project, most already
# used), so the two file-heavy, rebuildable pieces live in --scratch (a
# group-traversable directory in the owner's scratch): the Python environment
# (UV_PROJECT_ENVIRONMENT, ~50K files) and the per-attempt git worktrees
# (.automil_worktrees, symlinked). Both are recorded in dest's benchmarks/.env
# so every launcher, hook and daemon resolves them the same way.
#
# Steps (each skipped when already done):
#   1. dest/autoMIL: git clone of the campaign repo at --commit, environment
#      via `uv sync --frozen --all-packages` into --scratch/venv, worktree
#      directory --scratch/worktrees symlinked as .automil_worktrees,
#      benchmarks/.env rewritten so every AUTOBENCH_*_ROOT points at
#      --scratch/guard_roots/<cohort> and UV_PROJECT_ENVIRONMENT at the venv;
#   2. the 78 cell roots + runtime/agent_protocol.json + logs/ rsync'd
#      (additive, never --delete) and verified by sha256 of every
#      campaign_state.json;
#   3. --scratch/guard_roots: the existing stand-in trees copied from
#      source/guard_roots (symlinks into version2 preserved) and re-verified
#      by build_guard_root_shims.py --i-know-idle (file-heavy, rebuildable:
#      scratch, not project space);
#   4. group + modes: chgrp -R, dirs 2770 (setgid so new files inherit the
#      group), files g+rw; `git worktree prune`;
#   5. verification: restart-safe materialize reports every root bound to
#      identical inputs, the scan sees 78 pending, the training tree is
#      clean, CLAUDE.md matches the protocol's pinned hash.
# The scratch tree is left in place; rename it yourself once the first cell
# has finished in the new tree.

set -euo pipefail
SOURCE=""; DEST=""; GROUP="rrg-jma"; COMMIT=""; VERIFY_ONLY=0; SCRATCH=""
while [ $# -gt 0 ]; do
    case "$1" in
        --source) SOURCE="$2"; shift ;;
        --dest) DEST="$2"; shift ;;
        --group) GROUP="$2"; shift ;;
        --commit) COMMIT="$2"; shift ;;
        --scratch) SCRATCH="$2"; shift ;;
        --verify-only) VERIFY_ONLY=1 ;;
        *) echo "unknown option: $1"; exit 2 ;;
    esac
    shift
done
[ -n "$SOURCE" ] && [ -n "$DEST" ] && [ -n "$SCRATCH" ] || { echo "usage: --source <scratch/autoMIL> --dest <work> --scratch <group-traversable scratch dir> [--group G] --commit <sha>"; exit 2; }
VENV="$SCRATCH/venv"; WORKTREES="$SCRATCH/worktrees"; GUARD_ROOTS="$SCRATCH/guard_roots"
export UV_PROJECT_ENVIRONMENT="$VENV"
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
    # Scratch pieces: group-traversable parents, group-writable worktree dir.
    mkdir -p "$SCRATCH" "$WORKTREES"; chgrp "$GROUP" "$SCRATCH" "$WORKTREES"
    chmod 2750 "$SCRATCH"; chmod 2770 "$WORKTREES"
    # The owner's scratch parents stay private; the group only gets traverse
    # (an ACL, so ownership and the listing bit are untouched).
    parent="$(dirname "$SCRATCH")"
    while [ "$parent" != "/" ] && [ "$parent" != "/scratch" ] && [ "$parent" != "/home" ]; do
        setfacl -m "g:$GROUP:x" "$parent"
        parent="$(dirname "$parent")"
    done
    [ -e "$DST_REPO/.automil_worktrees" ] || ln -s "$WORKTREES" "$DST_REPO/.automil_worktrees"
    rm -rf "$DST_REPO/.venv"   # never in project space (inodes)
    (cd "$DST_REPO" && uv sync --frozen --all-packages)
    chgrp -R "$GROUP" "$VENV"; chmod -R g+rX "$VENV"


    step "2. cell roots + protocol + logs"
    mkdir -p "$DST_REPO/$CAMPAIGN_REL/runtime" "$DST_REPO/logs"
    rsync -a "$SRC_REPO/$CAMPAIGN_REL/runtime/" "$DST_REPO/$CAMPAIGN_REL/runtime/"
    rsync -a "$SRC_REPO/logs/" "$DST_REPO/logs/"
    SRC_SHA=$(mktemp); DST_SHA=$(mktemp)
    (cd "$SRC_REPO/$CAMPAIGN_REL/runtime" && find . -name campaign_state.json -exec sha256sum {} + | sort) > "$SRC_SHA"
    (cd "$DST_REPO/$CAMPAIGN_REL/runtime" && find . -name campaign_state.json -exec sha256sum {} + | sort) > "$DST_SHA"
    diff "$SRC_SHA" "$DST_SHA" && echo "state files identical: $(wc -l < "$DST_SHA")"
    rm -f "$SRC_SHA" "$DST_SHA"

    step "3. guard roots (scratch)"
    [ -d "$SOURCE/guard_roots" ] || { echo "ERROR: no guard roots under $SOURCE"; exit 1; }
    mkdir -p "$GUARD_ROOTS"; rsync -a "$SOURCE/guard_roots/" "$GUARD_ROOTS/"
    (cd "$DST_REPO" && uv run --frozen --no-sync --package autobench python \
        benchmarks/scripts/build_guard_root_shims.py --guard-roots "$GUARD_ROOTS" --i-know-idle)
    chgrp -R "$GROUP" "$GUARD_ROOTS"; chmod -R g+rX "$GUARD_ROOTS"
    # benchmarks/.env for the new tree: cohort roots -> the copied stand-ins,
    # other paths canonicalized, the environment named.
    python3 - "$SRC_REPO/benchmarks/.env" "$DST_REPO/benchmarks/.env" "$GUARD_ROOTS" "$VENV" <<'PYEOF'
import os, re, sys
from pathlib import Path
src, dst, roots, venv = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
lines = []
for line in src.read_text().splitlines():
    if line.startswith("UV_PROJECT_ENVIRONMENT="):
        continue
    m = re.match(r'^(AUTOBENCH_([A-Z0-9_]+)_ROOT)=(.*)$', line)
    if m:
        cohort = m.group(2).lower()
        if (roots / cohort).is_dir():
            # A cohort with a stand-in tree points at the copied stand-in.
            line = f"{m.group(1)}={roots / cohort}"
        elif os.path.isabs(m.group(3)) and os.path.exists(m.group(3)):
            # Everything else keeps its target, but by its canonical path
            # (a ~/projects symlink only exists in the owner's home).
            line = f"{m.group(1)}={os.path.realpath(m.group(3))}"
    lines.append(line)
lines.append(f"UV_PROJECT_ENVIRONMENT={venv}")
dst.write_text("\n".join(lines) + "\n")
print(f"wrote {dst}")
PYEOF

    step "4. group + modes"
    chgrp -R "$GROUP" "$DEST"
    find "$DEST" -type d -exec chmod 2770 {} +
    find "$DEST" -type f -exec chmod g+rw {} +
    [ -x "$VENV/bin/python" ] || { echo "ERROR: $VENV has no python — uv sync failed"; exit 1; }
    git -C "$DST_REPO" worktree prune
fi

step "5. verification (each check fails the script)"
cd "$DST_REPO"
ROSTER_CELLS=$(python3 -c "import json;print(json.load(open('$CAMPAIGN_REL/active_roster.json'))['cells'])")
uv run --frozen --no-sync --package autobench python benchmarks/scripts/campaign_manifest.py materialize \
    --agent-protocol "$CAMPAIGN_REL/agent_protocol.json" > /dev/null \
    && echo "materialize: every existing root bound to identical inputs"
PENDING=$(uv run --frozen --no-sync --package autobench python benchmarks/scripts/campaign_scan.py \
    --runtime "$CAMPAIGN_REL/runtime" --roster "$CAMPAIGN_REL/active_roster.json" --class pending | grep -c .)
[ "$PENDING" = "$ROSTER_CELLS" ] || { echo "ERROR: scan sees $PENDING pending cells, roster declares $ROSTER_CELLS"; exit 1; }
echo "scan: $PENDING pending"
git diff --quiet HEAD -- src benchmarks/src benchmarks/scripts || { echo "ERROR: training tree is dirty"; exit 1; }
echo "training tree clean at $(git rev-parse --short HEAD)"
PIN=$(python3 -c "import json;print(json.load(open('$CAMPAIGN_REL/toolset.json'))['ancestor_memory']['CLAUDE.md'])")
[ "$(sha256sum CLAUDE.md | cut -c1-64)" = "$PIN" ] || { echo "ERROR: CLAUDE.md drifted from the pinned hash"; exit 1; }
echo "CLAUDE.md matches the pinned hash"
cat <<EOM

done. Before any member submits, run one reproduction gate per cohort so a
guard-root regression cannot block a cohort fleet-wide (sticky verdict); the
launcher's rule applies: --force only supersedes a measurement-mode block.
  sbatch --account=def-jma-ab --time=3:00:00 --gpus-per-node=h100:1 --cpus-per-task=12 --mem=64G \\
    --chdir="$DST_REPO" --wrap='set -a; source benchmarks/.env; set +a
      for c in tcga_luad__os__titan__titan tcga_hnsc__os__titan__titan cptac_pdac__os__titan__titan; do
        root=$CAMPAIGN_REL/runtime/\${c}__s42__preprint-v3
        force=\$(python3 -c "import json,sys;b=(json.load(open(\"\$root/campaign_state.json\")).get(\"baseline_reproduction\") or {});print(\"--force\" if b.get(\"mode\")==\"measurement\" else \"\")")
        uv run --frozen --no-sync --package autobench python benchmarks/scripts/campaign_stage.py run-baseline-reproduction --cell-root "\$root" --gpu 0 \$force
      done'
EOM
