"""propose + rank commands: paired by lifecycle (D-01)."""
from __future__ import annotations

import logging

import click

from automil.cli import main
from automil.cli._helpers import _find_automil_dir, _load_technique_map

logger = logging.getLogger(__name__)


@main.command()
@click.option("--n", default=6, help="Number of proposals to return")
@click.option("--max-per-branch", default=2, help="Max proposals per branch")
@click.option(
    "--include-held-out",
    is_flag=True,
    default=False,
    help=(
        "Include held-out gate-eval nodes (D-139; logs WARNING; "
        "do NOT use during the agent search loop)."
    ),
)
def rank(n: int, max_per_branch: int, include_held_out: bool):
    """Show top-ranked proposals from the experiment graph."""
    adir = _find_automil_dir()
    graph_path = adir / "graph.json"

    if not graph_path.exists():
        click.echo("No graph.json found. Run some experiments first.")
        return

    from automil.graph import ExperimentGraph
    graph = ExperimentGraph(path=str(graph_path), technique_map=_load_technique_map(adir))

    # D-139: held-out isolation — filter unless operator explicitly opts in.
    if include_held_out:
        logger.warning(
            "rank --include-held-out: held-out cell composites now visible; "
            "this MUST NOT be used during the agent search loop (D-139)."
        )
    else:
        graph._data["nodes"] = {
            k: v
            for k, v in graph.nodes.items()
            if not (
                isinstance(v, dict)
                and v.get("metadata", {}).get("held_out", False)
            )
        }

    graph.recalculate_scores()
    proposals = graph.rank_proposals(n=n, max_per_branch=max_per_branch)

    if not proposals:
        click.echo("No proposals available. Time to brainstorm!")
        return

    click.echo(f"Top {len(proposals)} proposals:\n")
    for i, node in enumerate(proposals, 1):
        node_id = node["id"]
        parent = node.get("parent_id", "root")
        desc = node.get("description", "")
        score = node.get("potential", 0)
        kind = node.get("kind", "unspecified")
        click.echo(f"  {i}. [{node_id}] [{kind}] (parent: {parent}, score: {score:.4f})")
        click.echo(f"     {desc}")
        click.echo()


#: Free mode exposes every kind. Architecture-preserving mode is narrower than
#: this Click choice and admits only PRESERVING_KINDS at runtime.
PROPOSAL_KINDS = ["architecture", "regularization", "hp", "data", "ensemble"]
STRUCTURAL_KINDS = frozenset({"architecture", "ensemble"})
PRESERVING_KINDS = frozenset({"regularization", "hp"})


@main.command()
@click.option("--parent", required=True, help="Parent node ID")
@click.option("--desc", required=True, help="Proposal description")
@click.option("--techniques", multiple=True, help="Technique tags")
@click.option("--kind", type=click.Choice(PROPOSAL_KINDS), default=None,
              help="Experiment kind. Free mode uses architecture|ensemble for "
                   "its structural quota; architecture-preserving mode permits "
                   "only regularization|hp.")
@click.option("--mil-model", default=None,
              help="MIL model identifier — stored in node metadata so `automil submit` "
                   "can inherit it as a fallback (D-12, REC-04).")
def propose(parent: str, desc: str, techniques: tuple, kind: str | None, mil_model: str | None):
    """Add a new experiment proposal to the graph."""
    adir = _find_automil_dir()
    from automil.admissibility import load_candidate_policy
    candidate_policy = load_candidate_policy(adir)
    if candidate_policy.mode == "architecture-preserving" and (
        kind is None or kind not in PRESERVING_KINDS
    ):
        # B4 (claims-alignment): refuse at the write. kind=None used to slip
        # through as "unspecified" and hard-fail one command later in
        # `automil portfolio` with a message that never named the offender —
        # burning agent-active budget on a loop the skill mandates every batch.
        offender = "Missing --kind" if kind is None else f"Refusing {kind!r} proposal"
        raise click.ClickException(
            f"{offender} in architecture-preserving mode: only 'hp' or "
            "'regularization' are admissible — the executable surface is "
            "declared scalars plus train-only optimizer/update, scheduler, "
            "and stopping policies; there is no data/sampling hook."
        )
    # CR-2 (audit 2026-07-23): propose is the agent's most frequent write. Do the
    # whole read-modify-write under locked_update so a concurrent daemon
    # completion cannot clobber this proposal (or vice versa) via a stale snapshot.
    from automil.graph import locked_update

    desc_norm = desc.strip()
    with locked_update(
        str(adir / "graph.json"), technique_map=_load_technique_map(adir)
    ) as graph:
        # Duplicate guard: refuse exact-description sibling proposals under the
        # same parent that are still pending or running. Prevents waste from
        # accidental double-proposes (the 0063="dup of 0057" case). Exact-match
        # only — fine-grained hyperparameter sweeps with different descriptions
        # are unaffected.
        for n in graph.nodes.values():
            if (n.get("parent_id") == parent
                    and n.get("type") == "proposed"
                    and n.get("status") in ("pending", "running")
                    and (n.get("description", "") or "").strip() == desc_norm):
                # Raising here exits the context without save() (lock released) —
                # no partial write.
                raise click.ClickException(
                    f"Refusing to propose: {n['id']} already exists under "
                    f"--parent {parent} with the same description "
                    f"'{desc_norm[:60]}'. Use a different description, pick a "
                    f"different parent, or wait for {n['id']} to complete."
                )

        node_id = graph.add_proposed(
            parent_id=parent,
            description=desc,
            techniques=list(techniques),
            kind=kind or "unspecified",
        )
        if mil_model:
            from automil.cells.state import normalize_mil_model
            from automil.graph import merged_metadata
            gnode = graph.get_node(node_id)
            # L-8a: copy-on-write (see graph.merged_metadata docstring) — a
            # plain setdefault+assign mutates node["metadata"] in place,
            # which is reachable from another writer via aliasing.
            gnode["metadata"] = merged_metadata(gnode, {"mil_model": normalize_mil_model(mil_model)})
        graph.recalculate_scores()
        # graph.save() runs on context exit under the lock.

    suffix = "" if kind else "  (no --kind; counts as non-structural in portfolio)"
    click.echo(f"Added proposal {node_id} [{kind or 'unspecified'}]: {desc}{suffix}")


@main.command()
@click.option("--threshold", default=0.5, show_default=True,
              help="Minimum structural (architecture+ensemble) fraction of pending proposals.")
def portfolio(threshold: float):
    """Validate the pending proposal mix under the configured search mode.

    Free mode retains the structural-fraction gate. Architecture-preserving
    mode has no architecture quota and instead fails if any architecture or
    ensemble proposal is pending.
    """
    adir = _find_automil_dir()
    graph_path = adir / "graph.json"
    if not graph_path.exists():
        click.echo("No graph.json found. Run some experiments first.")
        return

    from automil.graph import ExperimentGraph
    from automil.admissibility import load_candidate_policy
    graph = ExperimentGraph(path=str(graph_path), technique_map=_load_technique_map(adir))
    candidate_policy = load_candidate_policy(adir)

    pending = [n for n in graph.nodes.values()
               if n.get("type") == "proposed" and n.get("status") == "pending"]
    executed = [n for n in graph.nodes.values() if n.get("type") == "executed"]

    def _counts(nodes: list[dict]) -> dict[str, int]:
        out: dict[str, int] = {}
        for n in nodes:
            out[n.get("kind", "unspecified")] = out.get(n.get("kind", "unspecified"), 0) + 1
        return out

    def _render(label: str, nodes: list[dict]) -> float:
        counts = _counts(nodes)
        total = len(nodes)
        structural = sum(c for k, c in counts.items() if k in STRUCTURAL_KINDS)
        frac = (structural / total) if total else 0.0
        click.echo(f"{label} ({total}):")
        for k in PROPOSAL_KINDS + ["unspecified"]:
            if counts.get(k):
                tag = " (structural)" if k in STRUCTURAL_KINDS else ""
                click.echo(f"  {k:<14} {counts[k]}{tag}")
        click.echo(f"  → structural: {structural}/{total} = {frac:.0%}")
        return frac

    if candidate_policy.mode == "architecture-preserving":
        def _render_recipe(label: str, nodes: list[dict]) -> None:
            counts = _counts(nodes)
            click.echo(f"{label} ({len(nodes)}) — recipe-only mode:")
            for proposal_kind in PROPOSAL_KINDS + ["unspecified"]:
                if counts.get(proposal_kind):
                    tag = (
                        " (forbidden)"
                        if proposal_kind not in PRESERVING_KINDS
                        else ""
                    )
                    click.echo(f"  {proposal_kind:<14} {counts[proposal_kind]}{tag}")

        _render_recipe("Pending proposals", pending)
        if executed:
            click.echo("")
            _render_recipe("Executed so far", executed)
        forbidden = [
            n for n in pending
            if n.get("kind", "unspecified") not in PRESERVING_KINDS
        ]
        if forbidden:
            click.echo("")
            click.echo(
                "PORTFOLIO FORBIDDEN: architecture-preserving mode fixes the "
                "published model and has no data/sampling hook; retain only "
                "hp/regularization proposals that fit the train-only policy seam."
            )
            raise SystemExit(1)
        return

    pending_frac = _render("Pending proposals", pending)
    if executed:
        click.echo("")
        _render("Executed so far", executed)

    if pending and pending_frac < threshold:
        click.echo("")
        click.echo(
            f"PORTFOLIO BELOW TARGET: pending is {pending_frac:.0%} structural "
            f"(target ≥{threshold:.0%}). Propose more architecture/ensemble "
            f"experiments before executing — don't let the batch be a pure HP sweep."
        )
        raise SystemExit(1)
