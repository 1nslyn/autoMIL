"""Behavior tests for Claude-native active-time accounting."""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from automil.cells.activity import (
    ACTIVITY_JOURNAL_FILENAME,
    ACTIVITY_SAMPLES_FILENAME,
    ActivityError,
    ActivityHealth,
    ActivityObservation,
    assess_activity,
    bind_activity_session,
    ingest_prometheus_metrics,
    read_activity_report,
    read_unbound_activity_report,
    record_hook_event,
)


def _record(
    tmp_path,
    event: str,
    at: float,
    *,
    cell_id: str = "cell-1",
    session_id: str = "session-1",
    **extra: object,
) -> None:
    payload = {"hook_event_name": event, "session_id": session_id, **extra}
    if event == "SessionStart":
        payload.setdefault("source", "startup")
    record_hook_event(
        tmp_path,
        cell_id,
        payload,
        observed_at=at,
        final_sample_observed_at=at if event == "SessionEnd" else None,
    )


def _metrics(
    session_id: str = "session-1",
    *,
    cli: float = 0.0,
    user: float = 0.0,
) -> str:
    return "\n".join(
        [
            "# TYPE claude_code_active_time_total counter",
            (
                "claude_code_active_time_total"
                f'{{session_id="{session_id}",type="cli"}} {cli}'
            ),
            (
                "claude_code_active_time_total"
                f'{{session_id="{session_id}",type="user"}} {user}'
            ),
            "",
        ]
    )


def test_report_uses_claude_native_active_time_not_session_wall_time(tmp_path):
    _record(tmp_path, "SessionStart", 1_000.0)
    assert ingest_prometheus_metrics(
        tmp_path, _metrics(cli=120.5, user=30.0), observed_at=91_000.0
    ) == ("session-1",)
    _record(tmp_path, "SessionEnd", 91_000.0)

    report = read_activity_report(tmp_path, "cell-1")

    assert report.active_seconds == 150.5
    assert report.sessions == ("session-1",)
    assert report.metered_sessions == ("session-1",)
    assert report.complete is True
    assert report.event_count == 3
    with pytest.raises(FrozenInstanceError):
        report.active_seconds = 0.0


def test_cumulative_samples_replace_prior_total_and_duplicates_are_idempotent(tmp_path):
    _record(tmp_path, "SessionStart", 1.0)
    assert ingest_prometheus_metrics(
        tmp_path, _metrics(cli=10), observed_at=2.0
    ) == ("session-1",)
    assert ingest_prometheus_metrics(
        tmp_path, _metrics(cli=10), observed_at=3.0
    ) == ("session-1",)
    assert ingest_prometheus_metrics(
        tmp_path, _metrics(cli=12, user=3), observed_at=4.0
    ) == ("session-1",)

    report = read_activity_report(tmp_path, "cell-1")
    assert report.active_seconds == 15.0
    assert report.event_count == 2
    assert len((tmp_path / ACTIVITY_JOURNAL_FILENAME).read_text().splitlines()) == 1
    samples = json.loads((tmp_path / ACTIVITY_SAMPLES_FILENAME).read_text())
    assert samples["sessions"]["session-1"]["active_seconds"] == 15.0


def test_active_time_regression_is_rejected(tmp_path):
    _record(tmp_path, "SessionStart", 1.0)
    ingest_prometheus_metrics(tmp_path, _metrics(cli=10), observed_at=2.0)

    with pytest.raises(ActivityError, match="active-time regression"):
        ingest_prometheus_metrics(tmp_path, _metrics(cli=9), observed_at=3.0)


def test_session_end_requires_a_persisted_final_sample(tmp_path):
    _record(tmp_path, "SessionStart", 1.0)
    with pytest.raises(ActivityError, match="final active-time sample"):
        _record(tmp_path, "SessionEnd", 2.0)

    ingest_prometheus_metrics(tmp_path, _metrics(cli=7), observed_at=3.0)
    with pytest.raises(ActivityError, match="stale"):
        _record(tmp_path, "SessionEnd", 4.0)

    # An unchanged counter no longer advances observed_at; the value must move
    # for the stored sample to carry the new timestamp the end will attest.
    ingest_prometheus_metrics(tmp_path, _metrics(cli=8), observed_at=4.0)
    _record(tmp_path, "SessionEnd", 4.0)

    assert read_activity_report(tmp_path, "cell-1").complete is True


def test_ended_session_rejects_a_tampered_final_sample_token(tmp_path):
    _record(tmp_path, "SessionStart", 1.0)
    ingest_prometheus_metrics(tmp_path, _metrics(cli=7), observed_at=2.0)
    _record(tmp_path, "SessionEnd", 2.0)
    samples_path = tmp_path / ACTIVITY_SAMPLES_FILENAME
    samples = json.loads(samples_path.read_text())
    samples["sessions"]["session-1"]["observed_at"] = 1.5
    samples_path.write_text(json.dumps(samples))

    with pytest.raises(ActivityError, match="final sample is stale"):
        read_activity_report(tmp_path, "cell-1")


def test_ended_session_sample_is_immutable(tmp_path):
    _record(tmp_path, "SessionStart", 1.0)
    ingest_prometheus_metrics(tmp_path, _metrics(cli=7), observed_at=3.0)
    _record(tmp_path, "SessionEnd", 3.0)

    assert ingest_prometheus_metrics(
        tmp_path, _metrics(cli=7), observed_at=4.0
    ) == ("session-1",)
    with pytest.raises(ActivityError, match="ended session"):
        ingest_prometheus_metrics(tmp_path, _metrics(cli=8), observed_at=5.0)

    report = read_activity_report(tmp_path, "cell-1")
    assert report.active_seconds == 7.0
    assert report.open_sessions == ()
    assert report.ended_sessions == ("session-1",)


def test_project_session_can_be_opened_unbound_then_bound_once(tmp_path):
    record_hook_event(
        tmp_path,
        None,
        {
            "hook_event_name": "SessionStart",
            "session_id": "session-1",
            "source": "startup",
        },
        observed_at=1.0,
    )

    unbound = read_unbound_activity_report(tmp_path)
    assert unbound.sessions == ("session-1",)
    assert read_activity_report(tmp_path, "cell-1").sessions == ()

    digest = "a" * 64
    bind_activity_session(tmp_path, "cell-1", "session-1", digest, observed_at=2.0)
    assert read_unbound_activity_report(tmp_path).sessions == ()
    assert read_activity_report(tmp_path, "cell-1").bindings == (
        ("session-1", digest),
    )
    with pytest.raises(ActivityError, match="conflicting binding"):
        bind_activity_session(
            tmp_path, "cell-2", "session-1", digest, observed_at=3.0
        )


def test_activity_assessment_distinguishes_live_complete_and_degraded(tmp_path):
    _record(tmp_path, "SessionStart", 1.0)
    ingest_prometheus_metrics(tmp_path, _metrics(cli=7), observed_at=2.0)
    report = read_activity_report(tmp_path, "cell-1")

    live = assess_activity(
        report,
        ActivityObservation(
            available=True, sessions=("session-1",), observed_at=2.0
        ),
    )
    assert live.health is ActivityHealth.OPEN_HEALTHY
    assert live.admissible is True
    assert live.active_seconds == 7.0

    unavailable = assess_activity(
        report,
        ActivityObservation(available=False, error="connection refused"),
    )
    assert unavailable.health is ActivityHealth.DEGRADED
    assert unavailable.admissible is False
    assert unavailable.active_seconds == 7.0

    foreign = assess_activity(
        report,
        ActivityObservation(
            available=True, sessions=("session-2",), observed_at=3.0
        ),
    )
    assert foreign.health is ActivityHealth.DEGRADED
    assert "does not match" in foreign.reason

    ingest_prometheus_metrics(tmp_path, _metrics(cli=8), observed_at=4.0)
    _record(tmp_path, "SessionEnd", 4.0)
    ended = assess_activity(
        read_activity_report(tmp_path, "cell-1"),
        ActivityObservation(available=False, error="endpoint gone"),
    )
    assert ended.health is ActivityHealth.ENDED_COMPLETE
    assert ended.complete is True
    assert ended.admissible is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"available": True, "observed_at": 1.0, "error": "contradiction"},
        {"available": True, "sessions": ("session-1",)},
        {"available": False, "sessions": ("session-1",), "error": "gone"},
    ],
)
def test_activity_observation_rejects_contradictory_states(kwargs):
    with pytest.raises(ActivityError):
        ActivityObservation(**kwargs)


def test_unknown_session_export_is_ignored_until_start_hook_arrives(tmp_path):
    assert ingest_prometheus_metrics(
        tmp_path, _metrics(cli=7), observed_at=1.0
    ) == ("session-1",)
    assert not (tmp_path / ACTIVITY_JOURNAL_FILENAME).exists()

    _record(tmp_path, "SessionStart", 2.0)
    assert ingest_prometheus_metrics(
        tmp_path, _metrics(cli=8), observed_at=3.0
    ) == ("session-1",)
    assert read_activity_report(tmp_path, "cell-1").active_seconds == 8.0


def test_cells_and_sessions_are_isolated(tmp_path):
    _record(tmp_path, "SessionStart", 0.0, cell_id="cell-a", session_id="a")
    _record(tmp_path, "SessionStart", 0.0, cell_id="cell-b", session_id="b")
    ingest_prometheus_metrics(tmp_path, _metrics("a", cli=3), observed_at=1.0)
    ingest_prometheus_metrics(tmp_path, _metrics("b", cli=5), observed_at=1.0)

    report_a = read_activity_report(tmp_path, "cell-a")
    report_b = read_activity_report(tmp_path, "cell-b")
    assert (report_a.active_seconds, report_a.sessions) == (3.0, ("a",))
    assert (report_b.active_seconds, report_b.sessions) == (5.0, ("b",))
    assert report_a.sha256 != report_b.sha256


def test_binding_is_immutable_and_cannot_follow_session_end(tmp_path):
    _record(tmp_path, "SessionStart", 1.0)
    first = "a" * 64
    bind_activity_session(tmp_path, "cell-1", "session-1", first, observed_at=2.0)
    bind_activity_session(tmp_path, "cell-1", "session-1", first, observed_at=3.0)
    assert read_activity_report(tmp_path, "cell-1").bindings == (
        ("session-1", first),
    )
    with pytest.raises(ActivityError, match="conflicting binding"):
        bind_activity_session(
            tmp_path, "cell-1", "session-1", "b" * 64, observed_at=3.0
        )

    ingest_prometheus_metrics(tmp_path, _metrics(cli=1), observed_at=4.0)
    _record(tmp_path, "SessionEnd", 4.0)
    with pytest.raises(ActivityError, match="session_bind after session"):
        bind_activity_session(
            tmp_path, "cell-1", "session-1", first, observed_at=5.0
        )


def test_session_id_cannot_move_between_cells(tmp_path):
    _record(tmp_path, "SessionStart", 0.0, cell_id="cell-a")
    with pytest.raises(ActivityError, match="belongs to another cell"):
        _record(tmp_path, "SessionEnd", 1.0, cell_id="cell-b")


def test_fresh_startup_only_and_unsupported_hooks_fail_loudly(tmp_path):
    with pytest.raises(ActivityError, match="start a fresh session"):
        _record(tmp_path, "SessionStart", 0.0, source="compact")
    _record(tmp_path, "SessionStart", 1.0)
    with pytest.raises(ActivityError, match="unsupported hook event"):
        _record(tmp_path, "PreToolUse", 2.0, tool_name="Monitor")


def test_unmatched_and_conflicting_session_events_are_rejected(tmp_path):
    with pytest.raises(ActivityError, match="unmatched session_end"):
        _record(tmp_path, "SessionEnd", 1.0)
    _record(tmp_path, "SessionStart", 2.0)
    _record(tmp_path, "SessionStart", 2.0)
    with pytest.raises(ActivityError, match="conflicting duplicate"):
        _record(tmp_path, "SessionStart", 3.0)


def test_timestamps_cannot_regress(tmp_path):
    _record(tmp_path, "SessionStart", 10.0)
    with pytest.raises(ActivityError, match="timestamp regression"):
        ingest_prometheus_metrics(tmp_path, _metrics(cli=1), observed_at=9.0)


@pytest.mark.parametrize(
    "payload, message",
    [
        (
            'claude_code_active_time_total{session_id="session-1",type="cli"} -1\n',
            "finite and non-negative",
        ),
        ('claude_code_active_time_total{type="cli"} 2\n', "session.id"),
        (
            'claude_code_active_time_total{session_id="session-1",type="other"} 2\n',
            "unsupported active-time type",
        ),
    ],
)
def test_malformed_active_metric_is_rejected(tmp_path, payload, message):
    _record(tmp_path, "SessionStart", 1.0)
    with pytest.raises(ActivityError, match=message):
        ingest_prometheus_metrics(tmp_path, payload, observed_at=2.0)


def test_export_without_active_metric_is_a_noop(tmp_path):
    _record(tmp_path, "SessionStart", 1.0)
    payload = "unrelated_metric 1\n"
    assert ingest_prometheus_metrics(tmp_path, payload, observed_at=2.0) == ()
    assert read_activity_report(tmp_path, "cell-1").metered_sessions == ()


def test_corrupt_or_truncated_lines_fail_loudly(tmp_path):
    _record(tmp_path, "SessionStart", 1.0)
    journal = tmp_path / ACTIVITY_JOURNAL_FILENAME
    with journal.open("a", encoding="utf-8") as stream:
        stream.write('{"event":"session_end"')

    with pytest.raises(ActivityError, match="truncated final line"):
        read_activity_report(tmp_path, "cell-1")
    with pytest.raises(ActivityError, match="truncated final line"):
        _record(tmp_path, "SessionEnd", 2.0)


def test_empty_report_and_canonical_journal_line(tmp_path):
    report = read_activity_report(tmp_path, "cell-1")
    assert report.active_seconds == 0.0
    assert report.sessions == report.metered_sessions == report.bindings == ()
    assert report.complete is False
    assert report.event_count == 0
    assert report.sha256 == (
        "e3b0c44298fc1c149afbf4c8996fb924"
        "27ae41e4649b934ca495991b7852b855"
    )

    _record(tmp_path, "SessionStart", 1.0)
    assert (tmp_path / ACTIVITY_JOURNAL_FILENAME).read_bytes() == (
        b'{"cell_id":"cell-1","event":"session_open","observed_at":1.0,'
        b'"session_id":"session-1"}\n'
    )


def test_journal_schema_rejects_obsolete_wait_events(tmp_path):
    obsolete = {
        "event": "wait_start",
        "cell_id": "cell-1",
        "session_id": "session-1",
        "tool_use_id": "tool-1",
        "observed_at": 1.0,
    }
    (tmp_path / ACTIVITY_JOURNAL_FILENAME).write_text(json.dumps(obsolete) + "\n")

    with pytest.raises(ActivityError, match="unknown journal event"):
        read_activity_report(tmp_path, "cell-1")
