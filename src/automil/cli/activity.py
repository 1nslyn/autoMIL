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

    store_warning = None
    if payload.get("hook_event_name") == "SessionEnd":
        # The transcript is the only record of the agent's reasoning and the
        # runtime prunes its own copy; copy it first, so accounting trouble
        # below never costs the record.
        store_warning = _store_transcript(automil_dir, payload)

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
                try:
                    close_dead_session(
                        automil_dir,
                        payload.get("session_id"),
                        "SessionEnd hook ran but the activity exporter was "
                        f"already unreachable ({exc}); finalized from the last "
                        "durable active-time sample on disk",
                        finalized_by="hook-exporter-unreachable",
                    )
                except ActivityError as fallback:
                    # Both causes or neither: the operator debugging a
                    # still-open session needs to know the scrape failed, not
                    # only that the promotion had nothing to promote.
                    raise ActivityError(
                        "cannot finalize SessionEnd: the exporter was "
                        f"unreachable ({exc}) and the last durable "
                        f"active-time sample could not be promoted: {fallback}"
                    ) from fallback
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
        if store_warning:
            click.echo(store_warning, err=True)
        raise click.ClickException(str(exc)) from exc
    if store_warning:
        click.echo(store_warning, err=True)


def _store_transcript(automil_dir, payload: dict) -> str | None:
    """Copy the session transcript into the project; a problem is a warning."""
    from pathlib import Path

    from automil.session_record import SessionRecordError, store_session_record

    transcript_path = payload.get("transcript_path")
    if not isinstance(transcript_path, str) or not transcript_path:
        return "session record not stored: the hook payload carried no transcript_path"
    try:
        outcome = store_session_record(automil_dir, payload.get("session_id"), Path(transcript_path))
    except (SessionRecordError, OSError) as exc:
        return f"session record not stored: {exc}"
    if outcome.action == "missing":
        return f"session record not stored: {outcome.detail}"
    return None


@activity.command("store-sessions", hidden=True)
@click.option(
    "--root", "root", default=None, type=click.Path(),
    help="Project root (or its automil/ dir). Default: discover from the working directory.",
)
@click.option(
    "--claude-config-dir", "config_dir", default=None, type=click.Path(),
    help="The runtime's config dir holding projects/<slug>/<session>.jsonl (default: CLAUDE_CONFIG_DIR or ~/.claude).",
)
def store_sessions(root: str | None, config_dir: str | None) -> None:
    """Copy every journaled session's transcript into automil/sessions/.

    The SessionEnd hook stores a session as it ends; this covers sessions
    whose runtime was killed first. Exit status 1 when any session has no
    transcript left to copy.
    """
    from pathlib import Path

    from automil.session_record import store_journaled_sessions

    if root is not None:
        candidate = Path(root).resolve()
        automil_dir = candidate if candidate.name == "automil" else candidate / "automil"
        if not (automil_dir / "config.yaml").exists():
            raise click.ClickException(f"{root}: no automil/config.yaml found")
    else:
        automil_dir = _find_automil_dir()
    outcomes = store_journaled_sessions(automil_dir, config_dir=Path(config_dir) if config_dir else None)
    if not outcomes:
        click.echo(f"no session in {automil_dir / '.activity.jsonl'}")
        raise SystemExit(1)
    missing = False
    for outcome in outcomes:
        click.echo(f"{outcome.session_id}: {outcome.action} ({outcome.detail})")
        missing = missing or outcome.action == "missing"
    if missing:
        raise SystemExit(1)


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
