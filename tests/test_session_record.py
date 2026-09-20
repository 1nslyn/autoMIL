"""session_record: the runtime's transcript is copied into the project."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from automil.session_record import (
    SessionRecordError,
    claude_config_dir,
    locate_home_transcript,
    read_stored_sessions,
    store_journaled_sessions,
    store_session_record,
)

from tests.viz.conftest import SID, write_mini_session


@pytest.fixture
def home(tmp_path: Path) -> Path:
    """A fake ``~/.claude`` with one project slug holding the mini session."""
    config = tmp_path / "claude"
    slug = config / "projects" / "-data-project"
    write_mini_session(slug)
    return config


def test_store_copies_transcript_sidecar_and_manifest(tmp_path: Path, home: Path):
    automil = tmp_path / "automil"
    automil.mkdir()
    source = home / "projects" / "-data-project" / f"{SID}.jsonl"

    outcome = store_session_record(automil, SID, source)
    assert outcome.action == "stored"
    stored = automil / "sessions" / SID
    assert (stored / "transcript.jsonl").read_bytes() == source.read_bytes()
    assert (stored / "subagents" / "agent-agent1.jsonl").is_file()
    assert (stored / "subagents" / "agent-agent1.meta.json").is_file()
    manifest = json.loads((stored / "record.json").read_text())
    assert manifest["session_id"] == SID and manifest["sidecar_files"] == 2
    assert manifest["lines"] == source.read_bytes().count(b"\n") and manifest["complete"] is True
    assert manifest["bytes"] == source.stat().st_size and len(manifest["sha256"]) == 64

    # a second store with nothing new is a no-op
    assert store_session_record(automil, SID, source).action == "unchanged"

    # a grown source is copied again
    with source.open("ab") as fh:
        fh.write(b'{"type": "mode", "mode": "normal"}\n')
    again = store_session_record(automil, SID, source)
    assert again.action == "stored"
    assert (stored / "transcript.jsonl").read_bytes() == source.read_bytes()

    (record,) = read_stored_sessions(automil)
    assert record.session_id == SID and record.sidecar == stored
    assert record.manifest["lines"] == manifest["lines"] + 1


def test_store_validates_ids_and_reports_a_missing_source(tmp_path: Path):
    automil = tmp_path / "automil"
    automil.mkdir()
    with pytest.raises(SessionRecordError):
        store_session_record(automil, "../etc", tmp_path / "x.jsonl")
    with pytest.raises(SessionRecordError):
        store_session_record(automil, SID, tmp_path / "other.jsonl")
    outcome = store_session_record(automil, SID, tmp_path / f"{SID}.jsonl")
    assert outcome.action == "missing" and outcome.path is None
    assert read_stored_sessions(automil) == ()


def test_locate_honours_the_config_dir_and_newest_copy(tmp_path: Path, home: Path, monkeypatch):
    assert locate_home_transcript(SID, config_dir=home) == home / "projects" / "-data-project" / f"{SID}.jsonl"
    assert locate_home_transcript("0000aaaa-0000-0000-0000-000000000000", config_dir=home) is None
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home))
    assert claude_config_dir() == home
    assert locate_home_transcript(SID) is not None
    monkeypatch.delenv("CLAUDE_CONFIG_DIR")
    assert claude_config_dir() == Path.home() / ".claude"


def test_store_journaled_sessions_reports_each_outcome(tmp_path: Path, home: Path):
    automil = tmp_path / "automil"
    automil.mkdir()
    missing = "1111aaaa-0000-0000-0000-000000000000"
    (automil / ".activity.jsonl").write_text(
        '{"cell_id":null,"event":"session_open","observed_at":1.0,"session_id":"%s"}\n'
        '{"cell_id":null,"event":"session_open","observed_at":2.0,"session_id":"%s"}\n' % (SID, missing)
    )
    outcomes = store_journaled_sessions(automil, config_dir=home)
    assert [(o.session_id, o.action) for o in outcomes] == [(SID, "stored"), (missing, "missing")]
    assert store_journaled_sessions(tmp_path / "nowhere", config_dir=home) == ()
