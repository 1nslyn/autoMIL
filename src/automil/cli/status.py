"""status command: experiment status summary."""
from __future__ import annotations

import click

from automil.cli import main
from automil.cli._helpers import _find_automil_dir, _load_technique_map


@main.command()
def status():
    """Show experiment status summary."""
    adir = _find_automil_dir()

    queue = list((adir / "orchestrator" / "queue").glob("*.json"))
    completed = list((adir / "orchestrator" / "completed").glob("*.json"))

    graph_path = adir / "graph.json"
    if graph_path.exists():
        from automil.graph import ExperimentGraph
        graph = ExperimentGraph(path=str(graph_path), technique_map=_load_technique_map(adir))
        executed = sum(1 for n in graph.nodes.values() if n.get("type") == "executed")
        proposed = sum(1 for n in graph.nodes.values() if n.get("type") == "proposed")
        best = graph.meta.get("best_primary_value", 0)
        click.echo(f"Executed: {executed}  Proposed: {proposed}  Best: {best:.4f}")

        # GTE-06 / D-144: promotion rate + gate-health diagnostic
        try:
            rate = graph.promotion_rate(days=30)
            nominated_nodes = graph.nominations_in_window(days=30)
            nominated = len(nominated_nodes)
            promoted = sum(1 for n in nominated_nodes if n.get("status") == "registered")
            if nominated > 0:
                from automil.gate.stats import diagnose_gate_health
                click.echo(
                    f"Promotion rate (30d): {rate:.1%} ({promoted}/{nominated}) — "
                    f"{diagnose_gate_health(rate)}"
                )
            else:
                click.echo("Promotion rate (30d): no data (zero nominations in 30-day window)")
        except Exception:
            pass  # soft-fail: status never breaks on gate-helper bugs
    else:
        click.echo("No graph.json found.")

    click.echo(f"Queue: {len(queue)}  Completed (pending read): {len(completed)}")
