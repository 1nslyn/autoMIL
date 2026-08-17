"""revert-baseline command: reset registry.protected paths to base_commit (CLI-02 / D-42).

Safety: MANDATORY pre-stash before any checkout. Leo's "never blind-checkout
after submit" memory: `git checkout -- <file>` silently destroys uncommitted
work. This command stashes first, prints the stash name, then checks out.
"""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

from automil.cli import main
from automil.cli._helpers import _find_automil_dir, _find_git_root
from automil.cli.lifecycle._shared import _load_registry_or_die

logger = logging.getLogger(__name__)


def _latest_executed_base_commit(adir: Path) -> Optional[str]:
    """Return the most-recent executed node's base_commit, or None if no graph.

    "Most recent" = lexicographic max on `created_at` ISO-8601 string (works
    because ISO-8601 sorts correctly).
    """
    graph_path = adir / "graph.json"
    if not graph_path.exists():
        return None
    try:
        graph = json.loads(graph_path.read_text())
    except json.JSONDecodeError:
        return None
    executed_with_base: list[tuple[str, str]] = []  # (created_at, base_commit)
    for node in graph.get("nodes", {}).values():
        if node.get("type") == "executed":
            base = node.get("base_commit")
            ts = node.get("created_at", "")
            if base:
                executed_with_base.append((ts, base))
    if not executed_with_base:
        return None
    executed_with_base.sort()
    return executed_with_base[-1][1]


def _has_uncommitted_changes(git_root: Path) -> bool:
    """`git status --porcelain` returns non-empty iff dirty (staged or unstaged)."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=git_root, capture_output=True, text=True, check=True,
    )
    return bool(result.stdout.strip())


@main.command("revert-baseline")
def revert_baseline():
    """Reset registry.protected paths to their base_commit state.

    Workflow: when the agent has accumulated edits to shared library files
    that should NOT have been touched (registry.protected violation),
    run `automil revert-baseline` to git-checkout those paths back to clean.
    STASHES uncommitted changes first (Leo's "never blind-checkout" memory) —
    the stash name is printed to stdout so you can recover via `git stash pop`.

    Idempotent: running on a clean tree is a no-op.

    Hard-fails if:
      - no graph.json (cannot determine base_commit)
      - no executed nodes (cannot determine base_commit)
      - registry.protected is empty (nothing to revert)
      - git stash or git checkout fails (stderr surfaced)
    """
    adir = _find_automil_dir()
    git_root = _find_git_root()
    cfg = _load_registry_or_die(adir)

    # Validate inputs.
    if not cfg.protected:
        raise click.ClickException(
            "registry.protected is empty in automil/config.yaml. "
            "Nothing to revert. Edit the config to list paths the agent "
            "must not touch (e.g., 'benchmarks/lib/CLAM/**')."
        )

    base_commit = _latest_executed_base_commit(adir)
    if base_commit is None:
        raise click.ClickException(
            "Cannot determine base_commit: no executed nodes in "
            "automil/graph.json with a recorded base_commit field. "
            "Run at least one experiment via `automil submit` first."
        )

    # Verify base_commit actually exists in this repo (cat-file -t checks object
    # existence; rev-parse --verify only validates the SHA format).
    rp = subprocess.run(
        ["git", "cat-file", "-t", base_commit],
        cwd=git_root, capture_output=True, text=True,
    )
    if rp.returncode != 0:
        raise click.ClickException(
            f"base_commit {base_commit!r} is not a valid git SHA (resolved "
            f"from the most recent executed node). Check your git history; "
            f"the commit may have been rewritten or pruned."
        )

    # Idempotence check: clean tree at protected paths -> no-op.
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *list(cfg.protected)],
        cwd=git_root, capture_output=True, text=True, check=True,
    )
    if not status.stdout.strip():
        click.echo("revert-baseline: protected paths already clean; nothing to do.")
        return

    # MANDATORY pre-stash (D-42 + Leo's never-blind-checkout memory).
    if _has_uncommitted_changes(git_root):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        stash_name = f"automil-revert-{ts}"
        stash_result = subprocess.run(
            ["git", "stash", "push", "--include-untracked", "-m", stash_name],
            cwd=git_root, capture_output=True, text=True,
        )
        if stash_result.returncode != 0:
            raise click.ClickException(
                f"`git stash push` failed: {stash_result.stderr.strip()}. "
                f"Resolve the conflict and retry, or revert manually with "
                f"`git checkout -p`."
            )
        click.echo(f"Stashed uncommitted changes as {stash_name!r}.")
        click.echo(f"  Recover via: git stash list  &&  git stash pop")
    else:
        click.echo("Working tree clean; no stash needed.")

    # A protected pattern may match nothing git knows about — e.g. a path
    # protected ahead of the branch that introduces it. `git checkout` is
    # all-or-nothing over its pathspecs: one unmatched pattern fails the
    # whole command AND leaves every other protected path dirty. Filter to
    # patterns that match at least one path known to base_commit or the
    # index (`ls-files --with-tree` shares checkout's pathspec semantics);
    # an unknown pattern has no baseline state to revert TO, and any
    # untracked file the agent created under it was already cleared by the
    # stash above.
    revertable = []
    skipped = []
    for pattern in cfg.protected:
        probe = subprocess.run(
            ["git", "ls-files", "--with-tree", base_commit, "--", pattern],
            cwd=git_root, capture_output=True, text=True,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            revertable.append(pattern)
        else:
            skipped.append(pattern)

    if revertable:
        co = subprocess.run(
            ["git", "checkout", base_commit, "--", *revertable],
            cwd=git_root, capture_output=True, text=True,
        )
        if co.returncode != 0:
            raise click.ClickException(
                f"`git checkout {base_commit[:8]}` for protected paths failed: "
                f"{co.stderr.strip()}. Inspect with `git status` and resolve manually."
            )

    click.echo(f"Reverted protected paths to base_commit {base_commit[:8]}.")
    click.echo(f"  Patterns: {revertable}")
    if skipped:
        click.echo(
            f"  Skipped (no baseline state at {base_commit[:8]}): {skipped}"
        )
