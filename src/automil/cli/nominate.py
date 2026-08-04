"""Top-level nominate command (GTE-05 / D-142 / D-145).

Shortcut at `automil nominate <node_id>` — sibling of `automil submit` —
because operators use it more often than `automil gate nominate` would imply.
The gate subgroup (cli/gate.py) remains intact; this is an ADDITIVE top-level
alias per D-145 design decision.

BCK-04 clean: no os.kill / os.killpg / Popen / .pid references.
Framework purity: generic framework code only — D-148 verified.
"""
from __future__ import annotations

import click

from automil.cli import main


@main.command("nominate")
@click.argument("node_id")
@click.option(
    "--agent",
    is_flag=True,
    default=False,
    hidden=True,
    help="Mark as agent-initiated (auto_nominate path; audit log only).",
)
def nominate_cmd(node_id: str, agent: bool) -> None:
    """Nominate a keep-status node as a gate candidate (D-142).

    Mutates status keep -> candidate. Idempotent. Run `automil promote <node_id>`
    afterwards to evaluate against the parent's pre-registered held-out cells.
    """
    from automil.cli._helpers import _find_automil_dir, _load_technique_map
    from automil.gate import nominate
    from automil.graph import locked_update

    adir = _find_automil_dir()
    graph_path = adir / "graph.json"
    if not graph_path.exists():
        raise click.ClickException(f"No graph.json at {graph_path}")
    # CR-2: lock the read-modify-write so a concurrent daemon completion cannot
    # clobber the status mutation via a stale snapshot.
    with locked_update(str(graph_path), technique_map=_load_technique_map(adir)) as graph:
        try:
            nominate(node_id, graph, agent_initiated=agent)
        except ValueError as exc:
            # Exits without save() (lock released) — no partial write.
            raise click.ClickException(str(exc))
        status = graph.nodes[node_id].get("status")
        # graph.save() runs on context exit under the lock.
    click.echo(f"Nominated {node_id}: status -> {status}")
