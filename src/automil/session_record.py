"""Store the runtime's transcript of an agent session inside the project.

Claude Code writes every session to ``<config>/projects/<cwd-slug>/<id>.jsonl``
with a sidecar directory ``<id>/`` (subagent transcripts, fetched tool
results) in the user's home, and prunes them after its cleanup period. The
project is the record, so the session is copied to
``automil/sessions/<id>/{transcript.jsonl, subagents/..., record.json}``:
once by the ``SessionEnd`` hook, and again by ``automil activity
store-sessions`` for any session the hook missed. Both call the one function
here, and copying is idempotent.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

SESSIONS_DIRNAME = "sessions"
TRANSCRIPT_FILENAME = "transcript.jsonl"
RECORD_FILENAME = "record.json"
_SESSION_ID = re.compile(r"^[0-9a-f-]{8,64}$")


class SessionRecordError(ValueError):
    """A session record request that cannot be honoured as stated."""


@dataclass(frozen=True)
class StoredSession:
    session_id: str
    dir: Path
    transcript: Path
    sidecar: Path | None
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class StoreOutcome:
    session_id: str
    action: Literal["stored", "unchanged", "missing"]
    path: Path | None
    detail: str


def claude_config_dir(env: Mapping[str, str] | None = None) -> Path:
    """Where the runtime keeps its files: ``CLAUDE_CONFIG_DIR`` or ``~/.claude``."""
    environment = os.environ if env is None else env
    configured = environment.get("CLAUDE_CONFIG_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".claude"


def validate_session_id(session_id: object) -> str:
    if not isinstance(session_id, str) or not _SESSION_ID.match(session_id):
        raise SessionRecordError(f"not a session id: {session_id!r}")
    return session_id


def locate_home_transcript(session_id: str, *, config_dir: Path | None = None) -> Path | None:
    """The runtime's own copy of a session, newest first when several exist."""
    validate_session_id(session_id)
    projects = (config_dir or claude_config_dir()) / "projects"
    if not projects.is_dir():
        return None
    candidates = []
    for slug in projects.iterdir():
        candidate = slug / f"{session_id}.jsonl"
        if candidate.is_file():
            candidates.append(candidate)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def session_dir(automil_dir: Path, session_id: str) -> Path:
    return Path(automil_dir) / SESSIONS_DIRNAME / validate_session_id(session_id)


def _sha256(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    lines = 0
    size = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
            lines += chunk.count(b"\n")
            size += len(chunk)
    return digest.hexdigest(), size, lines


def _ends_with_newline(path: Path) -> bool:
    with path.open("rb") as fh:
        try:
            fh.seek(-1, os.SEEK_END)
        except OSError:
            return False
        return fh.read(1) == b"\n"


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(destination.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        shutil.copyfile(source, tmp)
        os.chmod(tmp, 0o666 & ~_current_umask())
        os.replace(tmp, destination)
    finally:
        if tmp.exists():
            tmp.unlink()


def _current_umask() -> int:
    from automil.runtime_helpers import _current_umask as read_umask

    return read_umask()


def _copy_sidecar(source: Path, destination: Path) -> int:
    """Replace ``destination``'s entries with ``source``'s; returns the file count."""
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for entry in sorted(source.iterdir()):
        target = destination / entry.name
        if entry.is_dir():
            tmp = destination / f".tmp-{entry.name}"
            if tmp.exists():
                shutil.rmtree(tmp)
            shutil.copytree(entry, tmp)
            if target.exists():
                shutil.rmtree(target)
            os.replace(tmp, target)
            count += sum(1 for p in target.rglob("*") if p.is_file())
        elif entry.is_file():
            _copy_file_atomic(entry, target)
            count += 1
    return count


def store_session_record(automil_dir: Path, session_id: str, transcript_path: Path) -> StoreOutcome:
    """Copy one session's transcript and sidecar into the project. Idempotent."""
    session_id = validate_session_id(session_id)
    source = Path(transcript_path)
    if source.name != f"{session_id}.jsonl":
        raise SessionRecordError(f"{source} is not the transcript of session {session_id}")
    if not source.is_file():
        return StoreOutcome(session_id, "missing", None, f"transcript not found at {source}")
    target_dir = session_dir(automil_dir, session_id)
    manifest_path = target_dir / RECORD_FILENAME
    digest, size, lines = _sha256(source)
    previous = _read_manifest(manifest_path)
    sidecar_source = source.with_suffix("")
    sidecar_count = sum(1 for p in sidecar_source.rglob("*") if p.is_file()) if sidecar_source.is_dir() else 0
    if (
        previous is not None
        and previous.get("sha256") == digest
        and previous.get("bytes") == size
        and previous.get("sidecar_files") == sidecar_count
        and (target_dir / TRANSCRIPT_FILENAME).is_file()
    ):
        return StoreOutcome(session_id, "unchanged", target_dir, "record already current")
    target_dir.mkdir(parents=True, exist_ok=True)
    _copy_file_atomic(source, target_dir / TRANSCRIPT_FILENAME)
    copied = _copy_sidecar(sidecar_source, target_dir) if sidecar_source.is_dir() else 0
    manifest = {
        "schema": 1,
        "session_id": session_id,
        "source_path": str(source),
        "stored_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "bytes": size,
        "lines": lines,
        "sha256": digest,
        "complete": _ends_with_newline(source),
        "sidecar_files": copied,
    }
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(target_dir))
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.chmod(tmp_name, 0o666 & ~_current_umask())
    os.replace(tmp_name, manifest_path)
    return StoreOutcome(session_id, "stored", target_dir, f"{lines} lines, {copied} sidecar files")


def store_journaled_sessions(automil_dir: Path, *, config_dir: Path | None = None) -> tuple[StoreOutcome, ...]:
    """Store every session the activity journal knows; never raises per session."""
    from automil.cells.activity import journal_sessions

    outcomes = []
    for session in journal_sessions(automil_dir):
        source = locate_home_transcript(session.session_id, config_dir=config_dir)
        if source is None:
            outcomes.append(StoreOutcome(session.session_id, "missing", None, "no transcript under the runtime's projects directory"))
            continue
        outcomes.append(store_session_record(automil_dir, session.session_id, source))
    return tuple(outcomes)


def read_stored_sessions(automil_dir: Path) -> tuple[StoredSession, ...]:
    """Every stored session record under ``automil/sessions``."""
    folder = Path(automil_dir) / SESSIONS_DIRNAME
    if not folder.is_dir():
        return ()
    records = []
    for entry in sorted(folder.iterdir()):
        transcript = entry / TRANSCRIPT_FILENAME
        if not entry.is_dir() or not _SESSION_ID.match(entry.name) or not transcript.is_file():
            continue
        manifest = _read_manifest(entry / RECORD_FILENAME) or {}
        sidecar = entry if (entry / "subagents").is_dir() else None
        records.append(StoredSession(entry.name, entry, transcript, sidecar, manifest))
    return tuple(records)
