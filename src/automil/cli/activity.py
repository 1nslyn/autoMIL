"""Hidden adapter for ingesting agent-runtime lifecycle hooks."""
from __future__ import annotations

import json
import sys

import click

from automil.cli import main
from automil.cli._helpers import _find_automil_dir


@main.group(hidden=True)
def activity() -> None:
    """Internal runtime-hook commands."""


@activity.command("ingest", hidden=True)
def ingest() -> None:
    """Read one Claude hook event from stdin and append it to the journal."""
    try:
        payload = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"invalid hook JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise click.ClickException("hook payload must be one JSON object")

    automil_dir = _find_automil_dir()
    from automil.activity_metrics import observe_activity_metrics
    from automil.cells.activity import ActivityError, record_hook_event

    try:
        final_sample_observed_at = None
        if payload.get("hook_event_name") == "SessionEnd":
            observation = observe_activity_metrics(automil_dir)
            expected = (payload.get("session_id"),)
            if not observation.available or observation.sessions != expected:
                raise ActivityError(
                    "cannot read this session's final Claude active-time metric "
                    "before SessionEnd"
                )
            final_sample_observed_at = observation.observed_at
        # Hooks identify the runtime session, not a mutable config-derived cell.
        # Submit binds this project-local session once it resolves final identity.
        record_hook_event(
            automil_dir,
            None,
            payload,
            final_sample_observed_at=final_sample_observed_at,
        )
        if payload.get("hook_event_name") == "SessionStart":
            observe_activity_metrics(automil_dir)
    except ActivityError as exc:
        raise click.ClickException(str(exc)) from exc
