"""Hidden adapter for ingesting agent-runtime lifecycle hooks."""
from __future__ import annotations

import json
import sys

import click
import yaml

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
    config_path = automil_dir / "config.yaml"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise click.ClickException(f"cannot read {config_path}: {exc}") from exc

    from automil.activity_metrics import refresh_activity_metrics
    from automil.cells.activity import ActivityError, record_hook_event
    from automil.cells.identity import CellIdentityError, resolve_cell_identity

    try:
        identity = resolve_cell_identity(config)
        if payload.get("hook_event_name") == "SessionEnd":
            observed_sessions = refresh_activity_metrics(automil_dir)
            if (
                observed_sessions is None
                or payload.get("session_id") not in observed_sessions
            ):
                raise ActivityError(
                    "cannot read this session's final Claude active-time metric "
                    "before SessionEnd"
                )
        record_hook_event(automil_dir, identity.cell_id, payload)
        if payload.get("hook_event_name") == "SessionStart":
            refresh_activity_metrics(automil_dir)
    except (ActivityError, CellIdentityError) as exc:
        raise click.ClickException(str(exc)) from exc
