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
    from automil.activity_hooks import activity_metrics_url, project_exporter_port
    from automil.activity_metrics import (
        fetch_activity_exposition,
        observe_activity_metrics,
    )
    from automil.cells.activity import (
        ActivityError,
        close_dead_session,
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
                port = project_exporter_port(automil_dir)
            except ValueError as exc:
                # A malformed activity.exporter_port declaration, surfaced
                # with the config path rather than as an endpoint failure.
                # Never fall back on this: the exporter was never addressed.
                raise ActivityError(str(exc)) from exc
            try:
                exposition = fetch_activity_exposition(
                    url=activity_metrics_url(port)
                )
            except ActivityError as exc:
                # The exporter dies with the runtime that hosts it, so a
                # SessionEnd scrape races its own teardown and loses often
                # enough to matter. Refusing here stranded the session open
                # and left the cell unfinalizable until an operator ran
                # `activity close` by hand -- for a runtime that had in fact
                # exited cleanly, making every such attestation misstate what
                # happened. The last durable sample is already on disk and is
                # exactly what that manual recovery would have promoted, so
                # promote it here instead and mark who decided.
                close_dead_session(
                    automil_dir,
                    payload.get("session_id"),
                    "SessionEnd hook ran but the activity exporter was already "
                    f"unreachable ({exc}); finalized from the last durable "
                    "active-time sample on disk",
                    finalized_by="hook-exporter-unreachable",
                )
            else:
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
    from automil.activity_hooks import activity_metrics_url, project_exporter_port
    from automil.activity_metrics import fetch_activity_exposition
    from automil.cells.activity import (
        ActivityError,
        close_dead_session,
        parse_active_sessions,
    )

    try:
        try:
            _metrics_url = activity_metrics_url(project_exporter_port(automil_dir))
        except ValueError as exc:
            raise ActivityError(str(exc)) from exc
        # Liveness guard independent of journal-ingest validity: refuse
        # whenever the endpoint answers and the target session is present,
        # even if an unrelated session would make a full observation invalid.
        try:
            exposition = fetch_activity_exposition(url=_metrics_url)
        except ActivityError:
            exposition = None  # endpoint dead — the case close exists for
        if exposition is not None and session_id in parse_active_sessions(
            exposition
        ):
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
