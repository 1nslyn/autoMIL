"""Canonical Claude runtime settings for native activity accounting."""
from __future__ import annotations

from typing import Mapping

CLAUDE_ACTIVITY_COMMAND = "uv run automil activity ingest"
ACTIVITY_HOOK_TIMEOUT_SEC = 15
ACTIVITY_METRICS_HOST = "127.0.0.1"
ACTIVITY_METRICS_PORT = 9464
ACTIVITY_METRICS_PATH = "/metrics"
ACTIVITY_METRICS_URL = (
    f"http://{ACTIVITY_METRICS_HOST}:{ACTIVITY_METRICS_PORT}{ACTIVITY_METRICS_PATH}"
)


def claude_activity_environment() -> dict[str, str]:
    """Return Claude's native active-time export contract.

    Claude computes ``claude_code.active_time.total`` itself and exposes its
    current cumulative value on the documented localhost Prometheus endpoint.
    """
    return {
        "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
        "OTEL_METRICS_EXPORTER": "prometheus",
        "OTEL_METRICS_INCLUDE_SESSION_ID": "true",
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


def claude_activity_settings() -> dict[str, object]:
    """Return the complete project-local Claude accounting contract."""
    return {
        "env": claude_activity_environment(),
        "hooks": claude_activity_hooks(),
    }


def missing_claude_activity_hooks(settings: object) -> tuple[str, ...]:
    """Return canonical activity settings absent from Claude settings."""
    if not isinstance(settings, Mapping):
        return ("env", *claude_activity_hooks())
    env = settings.get("env")
    missing: list[str] = []
    if not isinstance(env, Mapping) or any(
        env.get(key) != value
        for key, value in claude_activity_environment().items()
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
    "claude_activity_environment",
    "claude_activity_hooks",
    "claude_activity_settings",
    "missing_claude_activity_hooks",
]
