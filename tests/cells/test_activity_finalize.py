"""Atomic SessionEnd finalization and operator-attested close.

The two-step observe-then-record SessionEnd was a TOCTOU: any concurrent
equal-value scrape between the steps rewrote the stored sample's
``observed_at`` and permanently stranded the session open once the exporter
died with the runtime. These tests pin the atomic replacement and the
operator escape hatch for runtimes that die without their hook.
"""
from __future__ import annotations

import json

import pytest

from automil.cells.activity import (
    ACTIVITY_SAMPLES_FILENAME,
    ActivityError,
    close_dead_session,
    finalize_session_end,
    ingest_prometheus_metrics,
    read_activity_report,
    record_hook_event,
)


def _open_session(tmp_path, session_id: str = "session-1", at: float = 100.0):
    record_hook_event(
        tmp_path,
        "cell-1",
        {
            "hook_event_name": "SessionStart",
            "session_id": session_id,
            "source": "startup",
        },
        observed_at=at,
    )


def _metrics(session_id: str = "session-1", *, cli: float = 0.0) -> str:
    return (
        "# TYPE claude_code_active_time_total counter\n"
        f'claude_code_active_time_total{{session_id="{session_id}",type="cli"}} {cli}\n'
    )


def _sample(tmp_path, session_id: str = "session-1") -> dict:
    return json.loads((tmp_path / ACTIVITY_SAMPLES_FILENAME).read_text())[
        "sessions"
    ][session_id]


def test_ingest_advances_only_when_the_cumulative_value_moves(tmp_path):
    """An unchanged counter must not rewrite observed_at (churn + TOCTOU fuel)."""
    _open_session(tmp_path)
    ingest_prometheus_metrics(tmp_path, _metrics(cli=5.0), observed_at=110.0)
    assert _sample(tmp_path)["observed_at"] == 110.0

    ingest_prometheus_metrics(tmp_path, _metrics(cli=5.0), observed_at=120.0)
    assert _sample(tmp_path) == {"active_seconds": 5.0, "observed_at": 110.0}

    ingest_prometheus_metrics(tmp_path, _metrics(cli=6.0), observed_at=130.0)
    assert _sample(tmp_path) == {"active_seconds": 6.0, "observed_at": 130.0}


def test_session_end_survives_a_concurrent_equal_value_scrape(tmp_path):
    """The exact race that used to strand a session open forever.

    A daemon tick (or ``budget show``) ingests the same final counter value
    between the hook's observe and its journal append. With value-gated
    advancement plus single-lock finalization the end still records.
    """
    _open_session(tmp_path)
    # The daemon scraped the final value first...
    ingest_prometheus_metrics(tmp_path, _metrics(cli=5.0), observed_at=110.0)
    # ...and again after the hook's own scrape would have run.
    ingest_prometheus_metrics(tmp_path, _metrics(cli=5.0), observed_at=115.0)

    finalize_session_end(
        tmp_path,
        {"hook_event_name": "SessionEnd", "session_id": "session-1"},
        _metrics(cli=5.0),
    )

    report = read_activity_report(tmp_path, "cell-1")
    assert report.complete
    assert report.active_seconds == 5.0


def test_finalize_session_end_ingests_the_final_sample_itself(tmp_path):
    _open_session(tmp_path)

    finalize_session_end(
        tmp_path,
        {"hook_event_name": "SessionEnd", "session_id": "session-1"},
        _metrics(cli=7.5),
    )

    report = read_activity_report(tmp_path, "cell-1")
    assert report.complete
    assert report.active_seconds == 7.5


def test_finalize_session_end_requires_an_exclusive_session_scrape(tmp_path):
    _open_session(tmp_path)
    foreign = _metrics(cli=1.0) + _metrics("session-2", cli=2.0)

    with pytest.raises(ActivityError, match="final Claude active-time metric"):
        finalize_session_end(
            tmp_path,
            {"hook_event_name": "SessionEnd", "session_id": "session-1"},
            foreign,
        )


def test_finalize_session_end_rejects_unopened_and_duplicate(tmp_path):
    with pytest.raises(ActivityError, match="unmatched session_end"):
        finalize_session_end(
            tmp_path,
            {"hook_event_name": "SessionEnd", "session_id": "session-1"},
            _metrics(cli=1.0),
        )

    _open_session(tmp_path)
    finalize_session_end(
        tmp_path,
        {"hook_event_name": "SessionEnd", "session_id": "session-1"},
        _metrics(cli=1.0),
    )
    with pytest.raises(ActivityError, match="conflicting duplicate session_end"):
        finalize_session_end(
            tmp_path,
            {"hook_event_name": "SessionEnd", "session_id": "session-1"},
            _metrics(cli=1.0),
        )


def test_finalize_session_end_rejects_a_regressing_final_value(tmp_path):
    _open_session(tmp_path)
    ingest_prometheus_metrics(tmp_path, _metrics(cli=9.0), observed_at=110.0)

    with pytest.raises(ActivityError, match="active-time regression"):
        finalize_session_end(
            tmp_path,
            {"hook_event_name": "SessionEnd", "session_id": "session-1"},
            _metrics(cli=5.0),
        )


def test_operator_close_finalizes_a_dead_session_from_its_durable_sample(tmp_path):
    _open_session(tmp_path)
    ingest_prometheus_metrics(tmp_path, _metrics(cli=42.0), observed_at=110.0)

    seconds = close_dead_session(tmp_path, "session-1", "runtime OOM-killed")

    assert seconds == 42.0
    report = read_activity_report(tmp_path, "cell-1")
    assert report.complete
    assert report.active_seconds == 42.0
    # The ledger distinguishes an operator close from a native hook, and the
    # journal round-trips through the strict replay validator.
    lines = [
        json.loads(line)
        for line in (tmp_path / ".activity.jsonl").read_text().splitlines()
    ]
    assert lines[-1]["finalized_by"] == "operator-close"
    assert lines[-1]["attestation"] == "runtime OOM-killed"
    assert read_activity_report(tmp_path, "cell-1").complete


def test_operator_close_requires_sample_open_session_and_attestation(tmp_path):
    _open_session(tmp_path)

    with pytest.raises(ActivityError, match="no durable active-time sample"):
        close_dead_session(tmp_path, "session-1", "why")
    with pytest.raises(ActivityError, match="non-empty attestation"):
        close_dead_session(tmp_path, "session-1", "   ")
    with pytest.raises(ActivityError, match="unmatched session_end"):
        close_dead_session(tmp_path, "session-9", "why")

    ingest_prometheus_metrics(tmp_path, _metrics(cli=1.0), observed_at=110.0)
    close_dead_session(tmp_path, "session-1", "died")
    with pytest.raises(ActivityError, match="already ended"):
        close_dead_session(tmp_path, "session-1", "died twice")
