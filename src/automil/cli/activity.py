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
    from automil.activity_metrics import (
        fetch_activity_exposition,
        observe_activity_metrics,
    )
    from automil.cells.activity import (
        ActivityError,
        finalize_session_end,
        record_hook_event,
    )

    try:
        if payload.get("hook_event_name") == "SessionEnd":
            # Scrape and record under one activity lock: the old two-step
            # (observe, then append with the observation's timestamp) lost a
            # race to any concurrent scrape and stranded the session open
            # forever once the exporter died with the runtime.
            try:
                exposition = fetch_activity_exposition()
            except ActivityError as exc:
                raise ActivityError(
                    "cannot read this session's final Claude active-time "
                    f"metric before SessionEnd: {exc}"
                ) from exc
            finalize_session_end(automil_dir, payload, exposition)
        else:
            # Hooks identify the runtime session, not a mutable config-derived
            # cell. Submit binds this project-local session once it resolves
            # final identity.
            record_hook_event(automil_dir, None, payload)
            if payload.get("hook_event_name") == "SessionStart":
                observe_activity_metrics(automil_dir)
    except ActivityError as exc:
        raise click.ClickException(str(exc)) from exc


@activity.command("close", hidden=True)
@click.option("--session", "session_id", required=True, help="Runtime session id to finalize.")
@click.option(
    "--attest",
    required=True,
    help="Why the operator is closing it; recorded verbatim in the journal.",
)
def close(session_id: str, attest: str) -> None:
    """Operator-attested close for a session whose runtime died without SessionEnd.

    Refuses while the live exporter still serves this session — a live session
    ends through its own hook. The stored durable sample becomes the attested
    final active-time observation; disclose the closure wherever the consuming
    protocol records termination reasons.
    """

    automil_dir = _find_automil_dir()
    from automil.activity_metrics import observe_activity_metrics
    from automil.cells.activity import ActivityError, close_dead_session

    try:
        observation = observe_activity_metrics(automil_dir)
        if observation.available and session_id in observation.sessions:
            raise ActivityError(
                f"session {session_id!r} is still exporting metrics; "
                "end the live session instead of operator-closing it"
            )
        seconds = close_dead_session(automil_dir, session_id, attest)
    except ActivityError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"closed session {session_id} at attested {seconds:.1f} active seconds"
    )
