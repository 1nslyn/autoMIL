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
    max-primary_value node (lex tie-break on ``node_id``), updates
    ``meta.best_node_id`` and ``meta.best_primary_value``, and prints a
    one-line summary. ``--dry-run`` prints the same summary without
    writing.
    """
    adir = _find_automil_dir()
    from automil.graph import ExperimentGraph

    # D-11: opt-in refresh of existing nodes from archive result.json.
    # Default reconcile (no --from-archive) stays missing-node-only.
    if from_archive is not None:
        import json as _json
        from automil.graph import keep_or_discard, locked_update, merged_metadata
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
                # B6 (claims-alignment): same ingest sanitation as the terminal
                # writer — key-guard, then prefer the val-recomputed primary_value
                # and fold-derived SE over the reported values.
                from automil.scoring import ingest_signal as _ingest_signal
                from automil.graph import node_primary_se as _node_se
                _leaking, _comp_rec, _se_rec, _refused = _ingest_signal(
                    payload, (g.meta.get("scoring") or {}).get("formula")
                )
                if _leaking:
                    # Ingest as crash (the ingest_signal contract), not skip:
                    # this is the ONE tool that refreshes existing executed
                    # nodes, so it is also the repair path for a node whose
                    # primary_value was contaminated before the A6 guard existed.
                    click.echo(
                        f"  {nid}: val-firewall violation — held-out-named "
                        f"metrics key(s) {', '.join(_leaking)}; ingesting as crash"
                    )
                    gnode["status"] = "crash"
                    gnode["primary_value"] = 0.0
                    gnode["metrics"] = {}
                    gnode["metadata"] = merged_metadata(gnode, {
                        "result_status": payload.get("status"),
                        "firewall_violation": list(_leaking),
                    })
                    refreshed += 1
                    continue
                # Status-independent, like the terminal writer and the refusal
                # branch below: a partial archive's usable val metrics still
                # beat its reported scalar — even a quarantined node can
                # parent a keep bar, so the reported (possibly test-derived)
                # value must never survive ingest just because the run was
                # partial.
                if _comp_rec is not None:
                    gnode["primary_value"] = _comp_rec
                elif _refused:
                    # Fail-closed (B2/B3), status-independent like every other
                    # ingest mouth: metrics present but unable to support the
                    # declared formula — never keep the reported scalar (even
                    # a quarantined node can still parent a keep bar).
                    click.echo(
                        f"  {nid}: metrics cannot support the declared "
                        "scoring.formula; refusing the reported primary_value "
                        "(scored 0.0)"
                    )
                    gnode["primary_value"] = 0.0
                else:
                    gnode["primary_value"] = payload.get("primary_value", gnode.get("primary_value", 0.0))
                _se_final = _se_rec if _se_rec is not None else _node_se(payload)
                if _se_final is not None:
                    gnode["primary_se"] = _se_final
                # Paired margin: refresh the fold projection from the archive
                # result so the keep/discard below (and this node's future role
                # as a parent) uses the paired basis. Assign-or-CLEAR — the
                # refresh is authoritative; keeping a previous run's folds
                # beside a refreshed primary_value would pair across runs.
                from automil.scoring import fold_primary_value_entries as _fold_entries
                _folds_refresh = _fold_entries(
                    payload, (g.meta.get("scoring") or {}).get("formula")
                )
                if _folds_refresh is not None:
                    gnode["fold_primary_values"] = _folds_refresh
                else:
                    gnode.pop("fold_primary_values", None)
                # Companion-guard evidence, refreshed before the gate below for
                # the same reason as the fold vector: deciding on the previous
                # run's companion metric would gate against stale evidence.
                # Assign-or-CLEAR, like the fold vector — a refresh whose
                # metrics went empty must not leave a superseded companion
                # value for this node's children to be gated against.
                gnode["metrics"] = dict(payload.get("metrics") or {})

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
                # the refreshed primary_value against the parent's primary_value to determine
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
                        gnode["status"] = keep_or_discard(g.meta, parent, gnode)
                    # else: unknown status value — leave gnode["status"] unchanged
                    # Preserve raw result status for traceability (operator-visible).
                    # L-8a: copy-on-write (graph.merged_metadata) — node["metadata"]
                    # can be aliased with another node's dict (gate/evaluate.py
                    # creates gate-eval children via a shallow dict(node) copy).
                    gnode["metadata"] = merged_metadata(gnode, {"result_status": raw_result_status})

                # (`metrics` was refreshed above, before the gate.)
                # A refreshed node's value/status/fold vector is a new gate
                # for everything below it (same contract as terminal_writer):
                # children screened against the old bar must be re-decided,
                # or a parent converted to crash keeps advertising decisions
                # made against a value the refresh just invalidated.
                if gnode.get("status") not in ("partial", "crash"):
                    g._reevaluate_descendants(nid)
                refreshed += 1
            if refreshed:
                # And the best pointer must not keep naming a node the
                # refresh demoted (terminal_writer calls this per ingest).
                g.recompute_best()
            # g.save() called automatically on context exit

        click.echo(f"Refreshed {refreshed} node(s) from archive.")
        return

    if recompute_best:
        # CLI-07 path: rebuild meta.best_node_id from executed/keep nodes.
        graph_path = adir / "graph.json"
        # CR-2 (audit 2026-07-23): persist under the lock; --dry-run stays a
        # read-only load (no lock, no write).
        if dry_run:
            graph = ExperimentGraph.load(graph_path, technique_map=_load_technique_map(adir))
            old_id, old_c, new_id, new_c = graph.recompute_best()
        else:
            from automil.graph import locked_update
            with locked_update(str(graph_path), technique_map=_load_technique_map(adir)) as graph:
                old_id, old_c, new_id, new_c = graph.recompute_best()
                # graph.save() runs on context exit under the lock.

        old_id_str = old_id if old_id is not None else "None"
        new_id_str = new_id if new_id is not None else "None"
        if old_id == new_id:
            # D-13 verbatim: unchanged-best line.
            click.echo(
                f"best_node_id unchanged: {new_id_str} (primary_value {new_c:.6f})"
            )
        else:
            # D-13 verbatim: changed-best line with literal Unicode → (U+2192).
            # ASCII fallback is forbidden — silently weakening the locked
            # decision is not allowed. stdout encoding is UTF-8 on Linux
            # (project is Linux-only per PROJECT.md).
            click.echo(
                f"best_node_id: {old_id_str} (primary_value {old_c:.6f}) "
                f"→ {new_id_str} (primary_value {new_c:.6f})"
            )

        return

    # Default path (D-14): orchestrator-state sync.
    # CR-2 (audit 2026-07-23): this path scans queue/running/completed/archive and
    # rewrites the whole graph — the widest lost-update window against a concurrent
    # daemon completion — so hold the lock across the read-modify-write.
    orch = adir / "orchestrator"
    from automil.graph import locked_update
    with locked_update(
        str(adir / "graph.json"), technique_map=_load_technique_map(adir)
    ) as graph:
        graph.reconcile(
            queue_dir=str(orch / "queue"),
            running_dir=str(orch / "running"),
            completed_dir=str(orch / "completed"),
            archive_dir=str(orch / "archive"),
        )
        # graph.save() runs on context exit under the lock.
    click.echo("Graph reconciled with orchestrator state.")
