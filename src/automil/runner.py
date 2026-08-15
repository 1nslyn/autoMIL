"""Git worktree overlay runner for experiment isolation."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _worktree_scope(project_root: Path, automil_dir: Path) -> str:
    """One filesystem-safe path component naming the automil project served.

    Worktrees live under ``.automil_worktrees/<scope>/<node_id>``. Scoping by
    the automil project directory makes concurrent orchestrators in one
    checkout disjoint even though every campaign cell numbers its nodes from
    ``node_0001`` (canary incident 2026-08-15: two concurrent promotion
    orchestrators wiped each other's live worktrees through the shared,
    node_id-keyed namespace — 18 of 20 jobs destroyed).
    """
    try:
        rel = automil_dir.resolve().relative_to(project_root.resolve())
    except ValueError:
        # An automil dir outside the checkout has no stable relative name;
        # fall back to a digest of its absolute path.
        return hashlib.sha256(str(automil_dir.resolve()).encode()).hexdigest()[:16]
    slug = "+".join(rel.parts) or "root"
    slug = re.sub(r"[^A-Za-z0-9+._-]", "_", slug)
    if len(slug) > 200:  # POSIX filename limit is 255 bytes; keep headroom
        slug = f"{slug[:184]}-{hashlib.sha256(slug.encode()).hexdigest()[:15]}"
    return slug


def _reject_nonfinite_constant(token: str):
    """``parse_constant`` hook: reject ``Infinity`` / ``-Infinity`` / ``NaN`` tokens.

    CR-1a (audit 2026-07-23): result.json is agent-writable and ``composite`` is
    trusted verbatim as the val-firewall selection signal. A non-finite composite
    would rig selection (``Infinity`` captures best_node and forces keep; ``NaN``
    poisons every ``>`` comparison and persists as an invalid-JSON token). Reject
    such tokens at the parse boundary — the semantic finite check in
    ``automil.schemas.validate_result`` is the second line of defense.
    """
    raise ValueError(f"non-finite JSON constant {token!r} is not permitted in result.json")


class Runner:
    """Manages git worktree lifecycle for experiment execution."""

    def __init__(self, project_root: str | Path, automil_dir: str | Path):
        self.project_root = Path(project_root)
        self._worktree_base = (
            self.project_root / ".automil_worktrees"
            / _worktree_scope(self.project_root, Path(automil_dir))
        )
        self._worktree_base.mkdir(parents=True, exist_ok=True)

    def worktree_path(self, node_id: str) -> Path:
        """Public accessor for a node's worktree path."""
        return self._worktree_base / node_id

    def create_worktree(self, base_commit: str, node_id: str) -> Path:
        """Create a detached worktree at the given commit.

        If a worktree directory already exists at the target path, it is
        removed (registration included) before the new ``git worktree add``
        runs. This handles the common case where a previous launch was
        interrupted and left the worktree orphaned. The removal is logged at
        WARNING so the paper trail survives — if that orphan was holding
        unsaved state (extremely rare; framework-owned subtree), the
        operator can correlate against the log line.

        Canary incident 2026-08-15: the previous implementation wiped the
        directory with a bare ``rmtree``, which leaves the git registration
        behind — the follow-up ``git worktree add`` then always fails with
        exit 128 ("missing but already registered worktree"), so the
        documented recovery never once succeeded. Removal now goes through
        ``cleanup_worktree`` (``git worktree remove --force`` with an
        rmtree+prune fallback), and a stale registration whose directory is
        already gone is cleared by the unconditional prune below.
        """
        wt_path = self._worktree_base / node_id
        if wt_path.exists():
            logger.warning(
                "Runner.create_worktree: %s already exists; removing before "
                "recreate (likely an interrupted prior launch). Original "
                "contents lost.",
                wt_path,
            )
            self.cleanup_worktree(wt_path)
        # A crash (or an out-of-band rmtree) can leave a registration whose
        # directory is already gone; `git worktree add` refuses over such a
        # stale registration, so prune unconditionally before adding.
        self.prune_stale_worktrees()

        subprocess.run(
            ["git", "worktree", "add", "--detach", str(wt_path), base_commit],
            cwd=self.project_root,
            capture_output=True,
            check=True,
        )
        return wt_path

    @staticmethod
    def _verify_overlay_manifest(overlay_dir: Path, manifest: dict[str, str]) -> None:
        """Check every file the manifest claims against its recorded digest.

        Raises:
            ValueError: a claimed file is missing, its digest does not match, or
                the recorded digest is malformed. All three are refusals rather
                than warnings: the manifest is the only record of what the agent
                actually submitted, so a mismatch means the archive no longer
                describes the experiment that was queued.
        """
        for rel, recorded in sorted(manifest.items()):
            if not isinstance(recorded, str) or not recorded.startswith("sha256:"):
                raise ValueError(
                    f"Overlay rejected: malformed digest for {rel!r} in the "
                    f"overlay manifest ({recorded!r}); expected 'sha256:<hex>'."
                )
            expected = recorded.split(":", 1)[1]
            src = overlay_dir / rel
            if not src.is_file():
                raise ValueError(
                    f"Overlay rejected: manifest claims {rel!r} but it is missing "
                    f"from {overlay_dir}."
                )
            actual = hashlib.sha256(src.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(
                    f"Overlay rejected: digest mismatch for {rel!r} — the archived "
                    f"file no longer matches what `automil submit` recorded "
                    f"(expected {expected[:12]}…, got {actual[:12]}…). The archive "
                    f"was modified after submit; refusing to run it."
                )

    def apply_overlay(self, worktree_path: Path, overlay_dir: Path,
                      deletions: list[str] | None = None,
                      *, manifest: dict[str, str] | None = None) -> None:
        """Copy modified files from overlay_dir on top of worktree.

        Also removes files listed in ``deletions`` from the worktree to
        support experiments that delete or rename files.

        Defensive boundary: even though ``automil submit`` validates paths
        upstream, the runner is the last line of defence before files
        land in the worktree. Reject ``..`` traversal, absolute paths,
        and symlinks pointing outside the worktree so a malicious or
        corrupt overlay (or deletions list) cannot land arbitrary files
        on disk.

        Args:
            manifest: HASH-0 — the ``{path: "sha256:..."}`` map recorded by
                ``automil submit``. Until now it was written into every queue
                spec and verified by nothing, so an archived overlay edited
                between submit and launch would run unnoticed. Verification
                happens BEFORE any copy, so a rejected overlay leaves the
                worktree untouched rather than half-applied. Only files the
                manifest actually claims are checked — the archive also holds
                run artifacts (``fold_*_result.json``, ``summary.json``) that
                were never part of the overlay. ``None`` skips verification, for
                legacy specs that carry no manifest.
        """
        wt_resolved = worktree_path.resolve()
        ov_resolved = overlay_dir.resolve()
        metadata_files = {Path("spec.json"), Path("run.log"), Path("result.json")}

        if manifest:
            self._verify_overlay_manifest(overlay_dir, manifest)

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
            # Val-firewall (Scope B): never copy the sealed test vault into a
            # worktree. certify/ is born in the archive dir, and a resubmit
            # overlays the parent node's archive, so exclude its whole subtree.
            if rel.parts and rel.parts[0] == "certify":
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
        """Persist the sealed result.json and return the parsed payload (or None).

        Val-firewall (Scope B): the FULL result payload carries the sealed
        ``held_out`` (test) block, so the durable copy lives in the off-limits
        ``archive/<node>/certify/`` subdir, never the agent-visible node-archive
        root. terminal_writer is the sole writer of the root ``result.json`` and
        strips test before writing it. The raw dict is still returned (held_out
        intact) so terminal_writer can route held_out into certify.json.

        L-3 (audit 2026-07-23): two shapes are handled here, because two
        writers exist for the worktree's result.json:

          - NEW: the training script called ``automil.runtime_helpers.
            write_result_json``, which already wrote the FULL payload
            directly into ``AUTOMIL_RESULTS_DIR`` (== ``sealed_dir`` below)
            and a STRIPPED (val-only, no ``held_out``/``summary``) sibling
            into the worktree. Before that helper existed, the worktree copy
            carried the full payload -- test metrics included -- for the
            entire run, and anything reading the project directory during
            search (including the coding agent driving it) could read the
            sealed metrics straight off disk. Now, at most, it can read the
            same validation-only view already shown in the final,
            agent-facing ``archive/<node>/result.json``. Because the sealed
            copy is already correct and authoritative in this case, it is
            read as-is -- the stripped worktree copy is NOT copied over it,
            which would silently overwrite the sealed ``held_out`` with
            nothing.
          - LEGACY: an older script wrote the FULL payload straight into the
            worktree and no sealed copy exists yet. Preserves the original
            behaviour byte-for-byte: copy the worktree file into
            ``sealed_dir`` and read it back from there.

        Neither file existing means the process produced no result at all
        (crash before any write) -- returns None, as before.
        """
        sealed_dir = archive_dir / "certify"
        sealed_file = sealed_dir / "result.json"
        result_file = worktree_path / "result.json"

        if sealed_file.exists():
            source = sealed_file
        elif result_file.exists():
            sealed_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(result_file, sealed_file)
            source = sealed_file
        else:
            return None

        try:
            return json.loads(
                source.read_text(), parse_constant=_reject_nonfinite_constant
            )
        except ValueError as exc:
            # CR-1a: a non-finite (Infinity/NaN) or otherwise malformed result.json
            # cannot be trusted as the selection signal. Degrade to a crash result —
            # the same outcome as a schema-invalid result at terminal_writer
            # ingestion — so it never influences keep/discard or best_node.
            logger.warning(
                "collect_result: rejected result.json for %s (%s) — treating as crash",
                archive_dir.name, exc,
            )
            return {
                "status": "crash",
                "composite": 0.0,
                "metrics": {},
                "error": f"result.json rejected at ingestion: {exc}",
            }

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
