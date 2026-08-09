"""Canonical Claude runtime settings for native activity accounting."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

CLAUDE_ACTIVITY_COMMAND = "uv run automil activity ingest"
ACTIVITY_HOOK_TIMEOUT_SEC = 15
ACTIVITY_METRICS_HOST = "127.0.0.1"
ACTIVITY_METRICS_PORT = 9464
ACTIVITY_METRICS_PATH = "/metrics"
ACTIVITY_METRICS_URL = (
    f"http://{ACTIVITY_METRICS_HOST}:{ACTIVITY_METRICS_PORT}{ACTIVITY_METRICS_PATH}"
)
_PORT_RANGE = range(1024, 65536)


def activity_metrics_url(port: int = ACTIVITY_METRICS_PORT) -> str:
    """Return the loopback exposition URL for one project's exporter port."""
    return f"http://{ACTIVITY_METRICS_HOST}:{port}{ACTIVITY_METRICS_PATH}"


def project_exporter_port(automil_dir: Path | str) -> int:
    """Resolve the project's declared exporter port (default 9464).

    Every consumer of the native counter — the settings written into the
    project, the runtime hooks, the orchestrator scrape, and the launcher
    probe — must resolve the same port for the same project, so the value
    lives in ``config.yaml`` under ``activity.exporter_port`` and everything
    derives from it. Concurrent projects on one host declare distinct ports.
    """
    import yaml

    config_path = Path(automil_dir) / "config.yaml"
    try:
        raw = config_path.read_text()
    except FileNotFoundError:
        # No config means no declaration: the default port, not an error.
        return ACTIVITY_METRICS_PORT
    except OSError as exc:
        raise ValueError(
            f"cannot resolve activity exporter port from {config_path}: {exc}"
        ) from exc
    try:
        config = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        raise ValueError(
            f"cannot resolve activity exporter port from {config_path}: {exc}"
        ) from exc
    section = config.get("activity") or {}
    port = section.get("exporter_port", ACTIVITY_METRICS_PORT)
    if isinstance(port, bool) or not isinstance(port, int) or port not in _PORT_RANGE:
        raise ValueError(
            f"activity.exporter_port must be an integer in "
            f"[{_PORT_RANGE.start}, {_PORT_RANGE.stop}) — got {port!r}"
        )
    return port


def claude_activity_environment(
    port: int = ACTIVITY_METRICS_PORT,
) -> dict[str, str]:
    """Return Claude's native active-time export contract.

    Claude computes ``claude_code.active_time.total`` itself and exposes its
    current cumulative value on a localhost Prometheus endpoint; the port is
    declared explicitly so concurrent sessions on one host never contend for
    one endpoint.
    """
    return {
        "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
        "OTEL_METRICS_EXPORTER": "prometheus",
        "OTEL_METRICS_INCLUDE_SESSION_ID": "true",
        "OTEL_EXPORTER_PROMETHEUS_PORT": str(port),
    }


def claude_activity_hooks(
    command: str = CLAUDE_ACTIVITY_COMMAND,
) -> dict[str, list[dict[str, object]]]:
    """Return fresh Claude settings entries for the activity journal.

    Hooks bind Claude's native active-time metric to an immutable session and
    cell.  Only fresh session startup is recorded: automatic compaction keeps
    the same session open and therefore must not create a second boundary.
    """
    command_hook = {
        "type": "command",
        "command": command,
        "timeout": ACTIVITY_HOOK_TIMEOUT_SEC,
    }

    def entry(*, matcher: str | None = None) -> dict[str, object]:
        value: dict[str, object] = {"hooks": [dict(command_hook)]}
        if matcher is not None:
            value["matcher"] = matcher
        return value

    return {
        "SessionStart": [entry(matcher="startup")],
        "SessionEnd": [entry()],
    }


def claude_activity_settings(
    port: int = ACTIVITY_METRICS_PORT,
) -> dict[str, object]:
    """Return the complete project-local Claude accounting contract."""
    return {
        "env": claude_activity_environment(port),
        "hooks": claude_activity_hooks(),
    }


def missing_claude_activity_hooks(
    settings: object, *, port: int = ACTIVITY_METRICS_PORT,
) -> tuple[str, ...]:
    """Return canonical activity settings absent from Claude settings."""
    if not isinstance(settings, Mapping):
        return ("env", *claude_activity_hooks())
    env = settings.get("env")
    missing: list[str] = []
    if not isinstance(env, Mapping) or any(
        env.get(key) != value
        for key, value in claude_activity_environment(port).items()
    ):
        missing.append("env")
    hooks = settings.get("hooks")
    if not isinstance(hooks, Mapping):
        return (*missing, *claude_activity_hooks())
    for event, expected_entries in claude_activity_hooks().items():
        actual_entries = hooks.get(event)
        if not isinstance(actual_entries, list) or any(
            expected not in actual_entries for expected in expected_entries
        ):
            missing.append(event)
    return tuple(missing)


__all__ = [
    "ACTIVITY_METRICS_HOST",
    "ACTIVITY_METRICS_PATH",
    "ACTIVITY_METRICS_PORT",
    "ACTIVITY_METRICS_URL",
    "ACTIVITY_HOOK_TIMEOUT_SEC",
    "CLAUDE_ACTIVITY_COMMAND",
    "activity_metrics_url",
    "claude_activity_environment",
    "claude_activity_hooks",
    "claude_activity_settings",
    "missing_claude_activity_hooks",
    "project_exporter_port",
]
