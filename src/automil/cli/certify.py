"""certify command: reveal the sealed held-out TEST performance ONCE (val-firewall).

During search the experiment tree selects on the VALIDATION composite and test
metrics are quarantined in a sealed ``archive/<node>/certify.json`` (written by
terminal_writer). ``certify`` is the single, deliberate end-of-run read of that
test performance for the val-selected winner (or an explicit ``--node`` / top-K)
— the honest generalization number that must never feed back into the search.
"""
from __future__ import annotations

import json
import logging

import click

from automil.cli import main
from automil.cli._helpers import _find_automil_dir

logger = logging.getLogger(__name__)


def _sorted_keep_nodes(graph) -> list[dict]:
    """Executed keep-nodes, best validation composite first (id breaks ties)."""
    keeps = [
        n for n in graph.nodes.values()
        if isinstance(n, dict)
        and n.get("type") == "executed"
        and n.get("status") == "keep"
    ]
    keeps.sort(key=lambda n: (n.get("composite", 0.0), n.get("id", "")), reverse=True)
    return keeps


@main.command()
@click.option("--node", "node_id", default=None,
              help="Node id to certify (default: the val-selected best node).")
@click.option("--top-k", "top_k", default=1, type=int,
              help="Certify the top-K keep nodes by validation composite.")
def certify(node_id: str | None, top_k: int):
    """Reveal sealed held-out TEST metrics for the val-selected node(s) — once.

    This is the ONLY sanctioned read of test. Do NOT run it inside the search
    loop: revealing test and acting on it reintroduces the selection leak the
    validation firewall exists to prevent.
    """
    adir = _find_automil_dir()
    graph_path = adir / "graph.json"
    if not graph_path.exists():
        click.echo("No graph.json found. Run some experiments first.")
        return

    from automil.graph import ExperimentGraph
    graph = ExperimentGraph(str(graph_path))
    archive = adir / "orchestrator" / "archive"

    # Which node(s) to certify.
    if node_id is not None:
        targets = [node_id]
    else:
        targets = [n["id"] for n in _sorted_keep_nodes(graph)[:max(1, top_k)]]
        if not targets and graph.meta.get("best_node_id"):
            targets = [graph.meta["best_node_id"]]

    if not targets:
        click.echo("No keep nodes to certify yet.")
        return

    logger.warning(
        "certify: revealing sealed held-out TEST metrics for %d node(s). This is "
        "an end-of-run action — never act on these numbers inside the search loop.",
        len(targets),
    )
    click.echo("Held-out certification (val-selected → honest test):\n")
    for nid in targets:
        node = graph.get_node(nid)
        if node is None:
            click.echo(f"  [{nid}] not found in graph.")
            continue
        val_comp = node.get("composite", 0.0)
        certify_path = archive / nid / "certify.json"
        if not certify_path.exists():
            click.echo(
                f"  [{nid}] val_composite={val_comp:.4f}  —  "
                "no certify.json (test not sealed for this node)."
            )
            continue
        try:
            sealed = json.loads(certify_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            click.echo(f"  [{nid}] failed to read certify.json: {exc}")
            continue
        held = sealed.get("held_out", {}) or {}
        test_str = "  ".join(
            f"{k}={v:.4f}" for k, v in sorted(held.items())
            if isinstance(v, (int, float))
        )
        click.echo(
            f"  [{nid}] val_composite={val_comp:.4f}  |  held-out: {test_str or '(none)'}"
        )

    click.echo(
        "\nReport the held-out numbers as the final generalization result. The "
        "val→test gap is the honest cost of search; do not re-select on it."
    )
