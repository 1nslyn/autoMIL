"""dequeue command: remove a queued/pending node from the orchestrator queue and mark
it cancelled in graph.json. Operates on nodes in proposed/pending or queued state.
Use `automil cancel` for running nodes. Uses locked_update for serialized graph writes
(unlike cancel.py's raw tempfile write — this is the correct pattern per RESEARCH.md).
"""
from __future__ import annotations

import logging
from pathlib import Path

import click

from automil.cli import main
from automil.cli._helpers import _find_automil_dir
from automil.cli.lifecycle._shared import _get_node_or_die

logger = logging.getLogger(__name__)

# WR-05: dequeue is only valid for a not-yet-executed proposal. graph.cancel()
# unconditionally decrements meta.total_proposed and flips status to
# "cancelled" with no type/status guard of its own, so allowing it on any
# other state double-decrements the proposed counter (the node was already
# decremented by mark_failed/promote) and rewrites already-executed results.
# Guard POSITIVELY on the only safe shape rather than enumerating a negative
# TERMINAL_STATES set — the codebase uses crash/oom/timeout/partial/registered
# statuses too (submit.py, graph.py), none of which belong in dequeue.
DEQUEUEABLE_STATES = frozenset({"pending", "queued"})


@main.command("dequeue")
@click.argument("node_id")
def dequeue(node_id: str) -> None:
    """Dequeue a queued or pending node.

    Removes orchestrator/queue/<node>.json if present and marks the graph node
    cancelled via graph.cancel() under locked_update.

    Accepts nodes in proposed/pending or queued state.
    Hard-fails if the node is running (use `automil cancel`) or already terminal.
    Idempotent: a pending node with no queue spec on disk is still marked cancelled,
    clearing orphaned proposals.
    """
    from automil.graph import locked_update  # noqa: PLC0415
    from automil.cli._helpers import _load_technique_map  # noqa: PLC0415

    adir = _find_automil_dir()

    # Step 1: look up node — hard-fail if unknown (prevents graph.cancel KeyError).
    node = _get_node_or_die(adir, node_id)

    # Step 2: positive state guard (WR-05, per D-05). Only a proposed node in a
    # pending/queued (not-yet-launched) state may be dequeued. Everything else —
    # running, or any executed/terminal status — is rejected so graph.cancel()
    # can never double-decrement total_proposed or rewrite executed results.
    state = node.get("status", "")
    node_type = node.get("type", "")
    if state == "running":
        raise click.ClickException(
            f"Node {node_id!r} is running. Use `automil cancel {node_id}` to stop it."
        )
    if not (node_type == "proposed" and state in DEQUEUEABLE_STATES):
        raise click.ClickException(
            f"Node {node_id!r} is {node_type or 'unknown'}/{state or 'unknown'}; "
            f"only pending proposals can be dequeued."
        )

    # Queue spec path (flat — NOT backend-namespaced per RESEARCH §OPS-02).
    # Queue files are orchestrator/queue/<node>.json. D-169 backend namespacing
    # applies to RUNNING specs only (orchestrator/running/<backend>/<node>.json).
    orch_dir = adir / "orchestrator"
    queue_spec = orch_dir / "queue" / f"{node_id}.json"

    # Step 3+4 (WR-04): perform the running-state re-check, the queue-spec
    # unlink, and the graph mark ALL under the same locked_update so the daemon
    # cannot interleave between them. The out-of-lock guard above is only a fast
    # fail; the authoritative decision is re-made here under the lock against a
    # freshly re-read node status.
    #
    # Residual TOCTOU (documented): the daemon's _get_pending()/_launch() globs
    # and unlinks queue specs WITHOUT holding the graph lock (see
    # _orchestrator_daemon.py _launch L921-924), so taking the graph lock here
    # narrows but does not fully close the window — the daemon can still pull a
    # spec out of queue/ concurrently. Re-reading status under the lock means we
    # will at worst observe the node as 'running' (daemon won the race) and
    # refuse, instead of marking a live job 'cancelled'. Fully closing it would
    # require the daemon to also take the graph lock before queue pickup.
    graph_path = adir / "graph.json"
    if graph_path.exists():
        with locked_update(graph_path, technique_map=_load_technique_map(adir)) as graph:
            locked_node = graph.get_node(node_id)
            if not locked_node:
                logger.warning("dequeue: node %s vanished from graph during lock", node_id)
            else:
                locked_state = locked_node.get("status", "")
                locked_type = locked_node.get("type", "")
                if locked_state == "running":
                    raise click.ClickException(
                        f"Node {node_id!r} started running before it could be "
                        f"dequeued. Use `automil cancel {node_id}` to stop it."
                    )
                if not (locked_type == "proposed" and locked_state in DEQUEUEABLE_STATES):
                    raise click.ClickException(
                        f"Node {node_id!r} is {locked_type or 'unknown'}/"
                        f"{locked_state or 'unknown'}; only pending proposals can "
                        f"be dequeued."
                    )
                if queue_spec.exists():
                    try:
                        queue_spec.unlink()
                        logger.debug("dequeue: removed queue spec %s", queue_spec)
                    except OSError as exc:
                        raise click.ClickException(
                            f"Could not remove queue spec at {queue_spec}: {exc}"
                        ) from exc
                graph.cancel(node_id)

    click.echo(f"Dequeued {node_id}.")
