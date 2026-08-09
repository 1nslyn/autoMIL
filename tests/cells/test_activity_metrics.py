"""Boundary tests for Claude's localhost Prometheus metric scrape."""
from __future__ import annotations

from io import BytesIO
from urllib.error import URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler

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
    observation = activity_metrics.observe_activity_metrics(
        tmp_path,
        open_url=lambda url, timeout: _Response(exposition),
        observed_at=2.0,
    )

    assert observation.available is True
    assert observation.sessions == ("session-1",)
    assert observation.observed_at == 2.0
    assert read_activity_report(tmp_path, "cell-1").active_seconds == 12.5


def test_refresh_returns_false_when_claude_endpoint_is_absent(tmp_path, monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise URLError("not running")

    observation = activity_metrics.observe_activity_metrics(
        tmp_path, open_url=unavailable
    )

    assert observation.available is False
    assert observation.sessions == ()
    assert "not running" in observation.error


def test_invalid_live_metric_becomes_degraded_observation(tmp_path):
    """Malformed telemetry holds admission without aborting the daemon tick."""
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
    payload = (
        'claude_code_active_time_total{session_id="session-1",type="cli"} -1\n'
    ).encode()

    observation = activity_metrics.observe_activity_metrics(
        tmp_path,
        open_url=lambda url, timeout: _Response(payload),
        observed_at=2.0,
    )

    assert observation.available is False
    assert observation.observed_at == 2.0
    assert "finite and non-negative" in observation.error
    assert read_activity_report(tmp_path, "cell-1").active_seconds == 0.0


def test_observe_scrapes_the_project_declared_exporter_port(tmp_path):
    (tmp_path / "config.yaml").write_text("activity:\n  exporter_port: 9581\n")
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
    scraped: list[str] = []

    def capture(url, timeout):
        scraped.append(url)
        return _Response(
            'claude_code_active_time_total'
            '{session_id="session-1",type="cli"} 10\n'.encode()
        )

    observation = activity_metrics.observe_activity_metrics(
        tmp_path, open_url=capture, observed_at=2.0,
    )

    assert observation.available is True
    assert scraped == ["http://127.0.0.1:9581/metrics"]


def test_invalid_declared_port_degrades_instead_of_crashing_the_tick(tmp_path):
    (tmp_path / "config.yaml").write_text("activity:\n  exporter_port: 80\n")

    observation = activity_metrics.observe_activity_metrics(
        tmp_path,
        open_url=lambda url, timeout: _Response(b""),
        observed_at=2.0,
    )

    assert observation.available is False
    assert "exporter_port must be an integer" in observation.error


def test_direct_loopback_opener_disables_proxies_and_redirects():
    handlers = activity_metrics._DIRECT_OPENER.handlers

    assert activity_metrics._NO_PROXY_HANDLER.proxies == {}
    assert not any(isinstance(handler, ProxyHandler) for handler in handlers)
    redirect_handler = next(
        handler for handler in handlers if isinstance(handler, HTTPRedirectHandler)
    )
    assert redirect_handler.redirect_request(None, None, None, None, None, None) is None


def test_refresh_compatibility_returns_sessions_or_none(tmp_path, monkeypatch):
    from automil.cells.activity import ActivityObservation

    monkeypatch.setattr(
        activity_metrics,
        "observe_activity_metrics",
        lambda *_args, **_kwargs: ActivityObservation(
            available=True, sessions=("session-1",), observed_at=1.0
        ),
    )
    assert activity_metrics.refresh_activity_metrics(tmp_path) == ("session-1",)

    monkeypatch.setattr(
        activity_metrics,
        "observe_activity_metrics",
        lambda *_args, **_kwargs: ActivityObservation(
            available=False, error="gone"
        ),
    )
    assert activity_metrics.refresh_activity_metrics(tmp_path) is None
