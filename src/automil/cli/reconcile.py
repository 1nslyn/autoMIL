"""reconcile command: sync experiment graph with orchestrator state.

Plan 01 (CLN-06) lifted the unflagged body verbatim from the original
``cli.py:510-524``. Plan 07 (CLI-07) adds ``--recompute-best`` (with
``--dry-run`` sibling) per locked decisions D-10..D-15. Existing
unflagged behaviour is byte-identical (D-14).
"""
from __future__ import annotations

import click

from automil.cli import main
from automil.cli._helpers import _find_automil_dir, _load_technique_map


@main.command()
@click.option(
    "--recompute-best",
    is_flag=True,
    default=False,
    help="Rebuild meta.best_node_id from executed/keep nodes (CLI-07).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="With --recompute-best: print summary, do not write graph.json.",
)
@click.option(
    "--from-archive",
    default=None,
    metavar="NODE_OR_ALL",
    help=(
        "Refresh existing node(s) from archive result.json. "
        "Pass a node_id or 'all'. Skips running nodes (Pitfall 3 guard). "
        "Default reconcile stays missing-node-only (D-11, REC-02)."
    ),
)
def reconcile(recompute_best: bool, dry_run: bool, from_archive: str | None):
    """Sync experiment graph with orchestrator state.

    With ``--recompute-best``: walks ``executed/keep`` nodes, picks the
    max-composite node (lex tie-break on ``node_id``), updates
    ``meta.best_node_id`` and ``meta.best_composite``, and prints a
    one-line summary. ``--dry-run`` prints the same summary without
    writing.
    """
    adir = _find_automil_dir()
    from automil.graph import ExperimentGraph

    # D-11: opt-in refresh of existing nodes from archive result.json.
    # Default reconcile (no --from-archive) stays missing-node-only.
    if from_archive is not None:
        import json as _json
        from automil.graph import locked_update, _accept, _accept_margin
        archive_dir = adir / "orchestrator" / "archive"
        graph_path = adir / "graph.json"

        if from_archive == "all":
            targets = [p.name for p in archive_dir.iterdir() if p.is_dir()] if archive_dir.exists() else []
        else:
            targets = [from_archive]

        refreshed = 0
        with locked_update(str(graph_path), technique_map=_load_technique_map(adir)) as g:
            for nid in targets:
                result_path = archive_dir / nid / "result.json"
                if not result_path.exists():
                    click.echo(f"  skip {nid}: no archive result.json")
                    continue
                gnode = g.get_node(nid)
                if gnode is None:
                    click.echo(f"  skip {nid}: not in graph (use default reconcile for missing nodes)")
                    continue
                # Pitfall 3 guard: never overwrite a live running node
                if gnode.get("status") == "running":
                    click.echo(f"  skip {nid}: currently running")
                    continue
                try:
                    payload = _json.loads(result_path.read_text())
                except (ValueError, OSError) as exc:
                    click.echo(f"  skip {nid}: malformed archive result.json ({exc})")
                    continue
                gnode["composite"] = payload.get("composite", gnode.get("composite", 0.0))

                # CR-03 fix: result.json status enum (completed/budget_killed/crash/
                # partial/cancelled) must NOT be written directly into gnode["status"].
                # Graph node status vocabulary is keep/discard/crash/partial/running/
                # pending/cancelled. Writing "completed" or "budget_killed" corrupts
                # graph semantics: _reevaluate_descendants skips non-keep/discard nodes,
                # recompute_best only counts "keep" nodes, and UCB scoring propagates
                # incorrect potentials from "completed" parents.
                #
                # Mapping: crash/partial/cancelled pass through unchanged.
                # completed/budget_killed are treated like a normal completion: compare
                # the refreshed composite against the parent's composite to determine
                # keep vs discard (same logic as terminal_writer.write_terminal_state).
                raw_result_status = payload.get("status")
                if raw_result_status is not None:
                    _GRAPH_PASSTHROUGH = {"crash", "partial", "cancelled"}
                    _COMPUTE_KEEPDISCARD = {"completed", "budget_killed"}
                    if raw_result_status in _GRAPH_PASSTHROUGH:
                        gnode["status"] = raw_result_status
                    elif raw_result_status in _COMPUTE_KEEPDISCARD:
                        parent_id = gnode.get("parent_id")
                        parent = g.get_node(parent_id) if parent_id else None
                        p_comp = parent.get("composite", 0.0) if parent else 0.0
                        composite = gnode["composite"]  # already updated above
                        gnode["status"] = "keep" if _accept(composite, p_comp, _accept_margin(g.meta)) else "discard"
                    # else: unknown status value — leave gnode["status"] unchanged
                    # Preserve raw result status for traceability (operator-visible).
                    gnode.setdefault("metadata", {})["result_status"] = raw_result_status

                if payload.get("metrics"):
                    gnode["metrics"] = payload["metrics"]
                refreshed += 1
            # g.save() called automatically on context exit

        click.echo(f"Refreshed {refreshed} node(s) from archive.")
        return

    if recompute_best:
        # CLI-07 path: rebuild meta.best_node_id from executed/keep nodes.
        graph_path = adir / "graph.json"
        graph = ExperimentGraph.load(graph_path, technique_map=_load_technique_map(adir))
        old_id, old_c, new_id, new_c = graph.recompute_best()

        old_id_str = old_id if old_id is not None else "None"
        new_id_str = new_id if new_id is not None else "None"
        if old_id == new_id:
            # D-13 verbatim: unchanged-best line.
            click.echo(
                f"best_node_id unchanged: {new_id_str} (composite {new_c:.6f})"
            )
        else:
            # D-13 verbatim: changed-best line with literal Unicode → (U+2192).
            # ASCII fallback is forbidden — silently weakening the locked
            # decision is not allowed. stdout encoding is UTF-8 on Linux
            # (project is Linux-only per PROJECT.md).
            click.echo(
                f"best_node_id: {old_id_str} (composite {old_c:.6f}) "
                f"→ {new_id_str} (composite {new_c:.6f})"
            )

        if not dry_run:
            graph.save()
        return

    # Default path (D-14): orchestrator-state sync. Body byte-identical to
    # Plan 01's lift from the original cli.py:510-524.
    orch = adir / "orchestrator"
    graph = ExperimentGraph(path=str(adir / "graph.json"), technique_map=_load_technique_map(adir))
    graph.reconcile(
        queue_dir=str(orch / "queue"),
        running_dir=str(orch / "running"),
        completed_dir=str(orch / "completed"),
        archive_dir=str(orch / "archive"),
    )
    graph.save()
    click.echo("Graph reconciled with orchestrator state.")
