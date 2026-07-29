"""certify command: reveal the sealed held-out TEST performance ONCE (val-firewall).

During search the experiment tree selects on the VALIDATION composite and test
metrics are quarantined in a sealed ``archive/<node>/certify/`` subdir (written by
terminal_writer). ``certify`` is the single, deliberate end-of-run read of that
test performance for the val-selected winner (or an explicit ``--node``)
— the honest generalization number that must never feed back into the search.

M-8 (audit 2026-07-23): revealing more than one node is selection on test by
another name. ``--top-k`` > 1 used to be permitted with nothing but a log
warning; it now requires ``--unseal-multiple``, an explicit acknowledgement of
what the extra reveals cost.
"""
from __future__ import annotations

import json
import logging

import click
from click.core import ParameterSource

from automil.cli import main
from automil.cli._helpers import _find_automil_dir

logger = logging.getLogger(__name__)

_MULTI_REVEAL_REFUSAL = (
    "--top-k {k} would unseal the held-out TEST block for {k} nodes.\n\n"
    "That is selection on test by another name. The validation firewall reads "
    "test exactly once, for the node validation already selected; the moment "
    "several nodes' test numbers are on the same page, the choice among them "
    "is a test-set choice — whether or not anyone writes down that they made "
    "it.\n\n"
    "The one legitimate use is a reporting table (e.g. a paper appendix) in "
    "which the K rows are published together and none of them is permitted to "
    "change a recipe, a ranking, or a claim. If that is what this is, re-run "
    "with --unseal-multiple. If you are still searching, do not."
)


def _sorted_keep_nodes(graph) -> list[dict]:
    """Executed keep-nodes, best validation composite first (id breaks ties)."""
    keeps = [
        n for n in graph.nodes.values()
        if isinstance(n, dict)
        and n.get("type") == "executed"
        and n.get("status") == "keep"
    ]
    # D-12 tie-break: highest composite first, lexicographically smaller id wins ties
    # (matches recompute_best / meta.best_node_id so `certify` reports the canonical winner).
    keeps.sort(key=lambda n: (-float(n.get("composite", 0.0)), n.get("id", "")))
    return keeps


def _validate_reveal_scope(
    ctx: click.Context,
    node_id: str | None,
    top_k: int,
    unseal_multiple: bool,
) -> None:
    """Refuse a multi-node reveal that has not been acknowledged (M-8).

    Runs before any project I/O so the refusal never depends on graph state.
    """
    if top_k < 1:
        # max(1, top_k) used to rewrite `--top-k 0` to 1 without saying so.
        raise click.UsageError(f"--top-k must be >= 1; got {top_k}.")
    if node_id is not None and ctx.get_parameter_source("top_k") is ParameterSource.COMMANDLINE:
        raise click.UsageError(
            "--node and --top-k are two different ways to choose what to "
            "unseal; pass only one. --node certifies exactly the node named, "
            "and --top-k was previously ignored when both were given."
        )
    if top_k > 1 and not unseal_multiple:
        raise click.UsageError(_MULTI_REVEAL_REFUSAL.format(k=top_k))


@main.command()
@click.option("--node", "node_id", default=None,
              help="Node id to certify (default: the val-selected best node).")
@click.option("--top-k", "top_k", default=1, type=int,
              help="Certify the top-K keep nodes by validation composite. "
                   "K>1 unseals several nodes' test blocks and so requires "
                   "--unseal-multiple.")
@click.option("--unseal-multiple", "unseal_multiple", is_flag=True, default=False,
              help="Acknowledge that --top-k K>1 unseals K held-out test "
                   "blocks. Comparing them IS selection on test: it spends the "
                   "one honest read the firewall protects, and any recipe, "
                   "ranking or claim chosen afterwards is no longer blind. "
                   "Use only for a published table where all K rows are "
                   "reported together and none may drive a decision.")
@click.pass_context
def certify(ctx: click.Context, node_id: str | None, top_k: int, unseal_multiple: bool):
    """Reveal sealed held-out TEST metrics for the val-selected node — once.

    This is the ONLY sanctioned read of test. Do NOT run it inside the search
    loop: revealing test and acting on it reintroduces the selection leak the
    validation firewall exists to prevent.
    """
    _validate_reveal_scope(ctx, node_id, top_k, unseal_multiple)

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
        targets = [n["id"] for n in _sorted_keep_nodes(graph)[:top_k]]
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
    if len(targets) > 1:
        # The acknowledgement was given at the CLI; say it again on the surface
        # a reader of this output will actually see (M-8).
        click.echo(
            f"WARNING: unsealing {len(targets)} held-out test blocks (--unseal-multiple). "
            "Reporting them together is permitted; choosing among them is not — "
            "that would be selection on test.\n"
        )
    click.echo("Held-out certification (val-selected → honest test):\n")
    for nid in targets:
        node = graph.get_node(nid)
        if node is None:
            click.echo(f"  [{nid}] not found in graph.")
            continue
        val_comp = node.get("composite", 0.0)
        certify_path = archive / nid / "certify" / "certify.json"
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
