"""Unit tests for the agent-activity marker (P2.1)."""
from __future__ import annotations

from automil.cells.activity import (
    ACTIVITY_FILENAME,
    read_last_action_at,
    touch_last_action,
)


def test_touch_then_read_roundtrip(tmp_path):
    touch_last_action(tmp_path, now=1234.5)
    assert read_last_action_at(tmp_path) == 1234.5
    assert (tmp_path / ACTIVITY_FILENAME).exists()


def test_touch_defaults_to_now(tmp_path, monkeypatch):
    monkeypatch.setattr("automil.cells.activity.time.time", lambda: 9999.0)
    touch_last_action(tmp_path)
    assert read_last_action_at(tmp_path) == 9999.0


def test_read_missing_returns_none(tmp_path):
    assert read_last_action_at(tmp_path) is None


def test_read_malformed_returns_none(tmp_path):
    (tmp_path / ACTIVITY_FILENAME).write_text("not-a-float\n")
    assert read_last_action_at(tmp_path) is None


def test_touch_never_raises_on_bad_dir(tmp_path):
    # Writing into a non-existent dir is swallowed (best-effort).
    touch_last_action(tmp_path / "does" / "not" / "exist")  # no exception
