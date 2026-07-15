"""Git worktree overlay runner for experiment isolation."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class Runner:
    """Manages git worktree lifecycle for experiment execution."""

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root)
        self._worktree_base = self.project_root / ".automil_worktrees"
        self._worktree_base.mkdir(exist_ok=True)

    def worktree_path(self, node_id: str) -> Path:
        """Public accessor for a node's worktree path."""
        return self._worktree_base / node_id

    def create_worktree(self, base_commit: str, node_id: str) -> Path:
        """Create a detached worktree at the given commit.

        If a worktree directory already exists at the target path, it is
        wiped before the new ``git worktree add`` runs. This handles the
        common case where a previous launch was interrupted and left
        ``.automil_worktrees/<node_id>/`` orphaned. The wipe is logged at
        WARNING so the paper trail survives — if that orphan was holding
        unsaved state (extremely rare; framework-owned subtree), the
        operator can correlate against the log line.
        """
        wt_path = self._worktree_base / node_id
        if wt_path.exists():
            logger.warning(
                "Runner.create_worktree: %s already exists; wiping before recreate "
                "(likely an interrupted prior launch). Original contents lost.",
                wt_path,
            )
            shutil.rmtree(wt_path)

        subprocess.run(
            ["git", "worktree", "add", "--detach", str(wt_path), base_commit],
            cwd=self.project_root,
            capture_output=True,
            check=True,
        )
        return wt_path

    def apply_overlay(self, worktree_path: Path, overlay_dir: Path,
                      deletions: list[str] | None = None) -> None:
        """Copy modified files from overlay_dir on top of worktree.

        Also removes files listed in ``deletions`` from the worktree to
        support experiments that delete or rename files.

        Defensive boundary: even though ``automil submit`` validates paths
        upstream, the runner is the last line of defence before files
        land in the worktree. Reject ``..`` traversal, absolute paths,
        and symlinks pointing outside the worktree so a malicious or
        corrupt overlay (or deletions list) cannot land arbitrary files
        on disk.
        """
        wt_resolved = worktree_path.resolve()
        ov_resolved = overlay_dir.resolve()
        metadata_files = {Path("spec.json"), Path("run.log"), Path("result.json")}

        for src_file in overlay_dir.rglob("*"):
            if not src_file.is_file():
                continue
            # Reject symlinks in the overlay (they could resolve outside
            # overlay_dir and exfiltrate / overwrite host paths).
            if src_file.is_symlink():
                raise ValueError(
                    f"Overlay rejected: symlink in overlay at {src_file} "
                    "(symlinks are not permitted in overlays)"
                )
            rel = src_file.relative_to(overlay_dir)
            if rel in metadata_files:
                continue
            dst = (worktree_path / rel).resolve()
            try:
                dst.relative_to(wt_resolved)
            except ValueError:
                raise ValueError(
                    f"Overlay rejected: target {dst} escapes worktree "
                    f"root {wt_resolved}"
                )
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst)

        if deletions:
            for rel_path in deletions:
                rel = Path(rel_path)
                if rel.is_absolute() or any(p == ".." for p in rel.parts):
                    raise ValueError(
                        f"Overlay rejected: deletion path {rel_path!r} "
                        "must be relative and may not contain '..'"
                    )
                target = (worktree_path / rel).resolve()
                try:
                    target.relative_to(wt_resolved)
                except ValueError:
                    raise ValueError(
                        f"Overlay rejected: deletion target {target} "
                        f"escapes worktree root {wt_resolved}"
                    )
                if target.exists():
                    target.unlink()

    def collect_result(self, worktree_path: Path, archive_dir: Path) -> dict | None:
        """Persist the worktree result.json and return the parsed payload (or None).

        Val-firewall (Scope B): the raw result.json carries the sealed ``held_out``
        (test) block, so the durable copy is written into the off-limits
        ``archive/<node>/certify/`` subdir, never the agent-visible node-archive
        root. terminal_writer is the sole writer of the root ``result.json`` and
        strips test before writing it. The raw dict is still returned (held_out
        intact) so terminal_writer can route held_out into certify.json.
        """
        result_file = worktree_path / "result.json"
        if not result_file.exists():
            return None

        sealed_dir = archive_dir / "certify"
        sealed_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result_file, sealed_dir / "result.json")

        return json.loads(result_file.read_text())

    def cleanup_worktree(self, worktree_path: Path) -> None:
        """Remove a git worktree."""
        try:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree_path)],
                cwd=self.project_root,
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            if worktree_path.exists():
                shutil.rmtree(worktree_path)
            self.prune_stale_worktrees()

    def prune_stale_worktrees(self) -> None:
        """Remove references to deleted worktrees."""
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=self.project_root,
            capture_output=True,
        )
