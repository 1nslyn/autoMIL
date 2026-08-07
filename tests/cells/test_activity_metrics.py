"""Boundary tests for Claude's localhost Prometheus metric scrape."""
from __future__ import annotations

from io import BytesIO
from urllib.error import URLError

from automil import activity_metrics
from automil.cells.activity import read_activity_report, record_hook_event


class _Response(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def test_refresh_persists_claude_native_counter(tmp_path, monkeypatch):
    record_hook_event(
        tmp_path,
        "cell-1",
        {
            "hook_event_name": "SessionStart",
            "session_id": "session-1",
            "source": "startup",
        },
        observed_at=1.0,
    )
    exposition = (
        '# TYPE claude_code_active_time_total counter\n'
        'claude_code_active_time_total{session_id="session-1",type="cli"} 10\n'
        'claude_code_active_time_total{session_id="session-1",type="user"} 2.5\n'
    ).encode()
    monkeypatch.setattr(
        activity_metrics,
        "urlopen",
        lambda url, timeout: _Response(exposition),
    )

    assert activity_metrics.refresh_activity_metrics(tmp_path) == ("session-1",)
    assert read_activity_report(tmp_path, "cell-1").active_seconds == 12.5


def test_refresh_returns_false_when_claude_endpoint_is_absent(tmp_path, monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise URLError("not running")

    monkeypatch.setattr(activity_metrics, "urlopen", unavailable)

    assert activity_metrics.refresh_activity_metrics(tmp_path) is None
