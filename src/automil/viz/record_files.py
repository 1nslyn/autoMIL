"""Where a project's run record lives on disk, and read-only loaders for it.

Every loader returns ``None`` (or an empty value) for a missing or unreadable
file and never raises on content. No loader has a path under
``archive/<node>/certify/``: the sealed directory is unknown to this module.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from automil.firewall import REDACTION, redact_held_out

logger = logging.getLogger(__name__)

NODE_ID_PATTERN = re.compile(r"^node_\d{4,}$")
RUN_LOG_TAIL_LINES = 200
OVERLAY_FILE_MAX_BYTES = 256 * 1024
_CERTIFY_DIRNAME = "certify"


@dataclass(frozen=True)
class RunPaths:
    """The files one project's record is built from."""

    automil_dir: Path

    @property
    def root(self) -> Path:
        return self.automil_dir.parent

    @property
    def config(self) -> Path:
        return self.automil_dir / "config.yaml"

    @property
    def graph(self) -> Path:
        return self.automil_dir / "graph.json"

    @property
    def orchestrator_dir(self) -> Path:
        return self.automil_dir / "orchestrator"

    @property
    def gpu_state(self) -> Path:
        return self.orchestrator_dir / "gpu_state.json"

    @property
    def orchestrator_log(self) -> Path:
        return self.orchestrator_dir / "orchestrator.log"

    @property
    def queue_dir(self) -> Path:
        return self.orchestrator_dir / "queue"

    @property
    def completed_dir(self) -> Path:
        return self.orchestrator_dir / "completed"

    @property
    def archive_dir(self) -> Path:
        return self.orchestrator_dir / "archive"

    @property
    def cells_dir(self) -> Path:
        return self.automil_dir / "cells"

    @property
    def sessions_dir(self) -> Path:
        return self.automil_dir / "sessions"

    @property
    def plan_md(self) -> Path:
        return self.automil_dir / "plan.md"

    @property
    def learnings_md(self) -> Path:
        return self.automil_dir / "learnings.md"

    @property
    def campaign_state(self) -> Path:
        return self.root / "campaign_state.json"

    @property
    def certification(self) -> Path:
        return self.root / "certification" / "certify.json"

    def node_archive(self, node_id: str) -> Path:
        if not NODE_ID_PATTERN.match(node_id):
            raise ValueError(f"not a node id: {node_id!r}")
        return self.archive_dir / node_id


def read_json(path: Path) -> Any:
    """Parse one JSON file; ``None`` when missing, unreadable or not JSON."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        logger.warning("record: cannot read %s: %s", path, exc)
        return None


def read_text(path: Path, *, max_bytes: int | None = None) -> str | None:
    """Read a text file with replacement for bad bytes; ``None`` when missing."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning("record: cannot read %s: %s", path, exc)
        return None
    if max_bytes is not None:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace")


def load_graph(paths: RunPaths) -> dict[str, Any] | None:
    payload = read_json(paths.graph)
    return payload if isinstance(payload, dict) else None


def load_gpu_state(paths: RunPaths) -> dict[str, Any] | None:
    payload = read_json(paths.gpu_state)
    return payload if isinstance(payload, dict) else None


def load_spec(paths: RunPaths, node_id: str) -> dict[str, Any] | None:
    """The launched spec (archive copy first, then the queue entry)."""
    for candidate in (paths.node_archive(node_id) / "spec.json", paths.queue_dir / f"{node_id}.json"):
        payload = read_json(candidate)
        if isinstance(payload, dict):
            return payload
    return None


def load_completed(paths: RunPaths, node_id: str) -> dict[str, Any] | None:
    RunPaths.node_archive(paths, node_id)  # validates the id
    payload = read_json(paths.completed_dir / f"{node_id}.json")
    return payload if isinstance(payload, dict) else None


def load_result(paths: RunPaths, node_id: str) -> dict[str, Any] | None:
    """The agent-facing ``result.json`` at the archive root (validation only)."""
    payload = read_json(paths.node_archive(node_id) / "result.json")
    return payload if isinstance(payload, dict) else None


def load_run_log_tail(paths: RunPaths, node_id: str, *, lines: int = RUN_LOG_TAIL_LINES) -> dict[str, Any] | None:
    """The last lines of a terminal node's run log, redacted again on the way out.

    Callers serve this only for terminal nodes (the orchestrator redacts the
    file at completion); ``redacted_lines`` counts what this pass had to hide.
    """
    text = read_text(paths.node_archive(node_id) / "run.log")
    if text is None:
        return None
    all_lines = text.splitlines()
    tail = all_lines[-lines:]
    clean = redact_held_out("\n".join(tail)).split("\n") if tail else []
    redacted = sum(1 for before, after in zip(tail, clean) if before != after and after == REDACTION)
    return {
        "available": True,
        "tail": clean,
        "n_lines_total": len(all_lines),
        "tail_lines": len(clean),
        "redacted_lines": redacted,
    }


def overlay_files(spec: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The changed files a spec declares (never a directory listing)."""
    if not isinstance(spec, dict):
        return []
    manifest = spec.get("overlay_manifest")
    files = []
    if isinstance(manifest, dict):
        for rel_path, digest in manifest.items():
            text = str(digest)
            files.append({"path": str(rel_path), "sha256": text.split(":", 1)[1] if text.startswith("sha256:") else text})
    elif isinstance(manifest, list):
        for entry in manifest:
            if isinstance(entry, dict) and "path" in entry:
                files.append({"path": str(entry["path"]), "sha256": entry.get("sha256")})
            elif isinstance(entry, str):
                files.append({"path": entry, "sha256": None})
    return files


def load_overlay_file(paths: RunPaths, node_id: str, rel_path: str) -> dict[str, Any] | None:
    """One changed file from a node's overlay, by its declared relative path."""
    archive = paths.node_archive(node_id)
    if not rel_path or rel_path.startswith("/") or ".." in Path(rel_path).parts:
        return None
    if _CERTIFY_DIRNAME in Path(rel_path).parts:
        return None
    target = archive / rel_path
    try:
        target.resolve().relative_to(archive.resolve())
    except ValueError:
        return None
    text = read_text(target, max_bytes=OVERLAY_FILE_MAX_BYTES + 1)
    if text is None:
        return None
    truncated = len(text.encode("utf-8")) > OVERLAY_FILE_MAX_BYTES
    if truncated:
        text = text.encode("utf-8")[:OVERLAY_FILE_MAX_BYTES].decode("utf-8", errors="ignore")
    clean = redact_held_out(text)
    if text.endswith("\n") and not clean.endswith("\n"):
        clean += "\n"
    return {"path": rel_path, "text": clean, "truncated": truncated}


def load_notes(paths: RunPaths) -> dict[str, Any]:
    notes: dict[str, Any] = {"plan_md": read_text(paths.plan_md), "learnings_md": read_text(paths.learnings_md)}
    mtimes = []
    for candidate in (paths.plan_md, paths.learnings_md):
        try:
            mtimes.append(candidate.stat().st_mtime)
        except OSError:
            continue
    notes["mtime"] = max(mtimes) if mtimes else None
    return notes


def load_orchestrator_log(paths: RunPaths) -> list[str]:
    text = read_text(paths.orchestrator_log)
    return text.splitlines() if text else []


def load_campaign_state(paths: RunPaths) -> dict[str, Any] | None:
    payload = read_json(paths.campaign_state)
    return payload if isinstance(payload, dict) else None


def load_certification(paths: RunPaths) -> dict[str, Any] | None:
    """The cell's own certification bundle, present only after ``certify-winner``."""
    payload = read_json(paths.certification)
    return payload if isinstance(payload, dict) else None


def load_config(paths: RunPaths) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:  # pragma: no cover - yaml is a core dependency
        return {}
    text = read_text(paths.config)
    if text is None:
        return {}
    try:
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        logger.warning("record: cannot parse %s: %s", paths.config, exc)
        return {}
    return payload if isinstance(payload, dict) else {}
