"""gate subgroup: gate manifest + promotion-rate commands (GTE-01..06 / D-145).

Pattern mirrors src/automil/cli/cell.py exactly:
  @main.group("gate") + subcommands with lazy imports inside command bodies.

Path-traversal defence: parent_id validated against ^node_\\d+$ regex before
any file or git operation (T-05-08-01 in plan threat model).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import click

from automil.cli import main


@main.group("gate")
def gate_group() -> None:
    """Generalization-gate manifest management commands."""
    pass


_PARENT_ID_RE = re.compile(r"^node_\d+$")


def _validate_parent_id(parent_id: str) -> None:
    """Raise ClickException if parent_id doesn't match ^node_\\d+$ (T-05-08-01)."""
    if not _PARENT_ID_RE.match(parent_id):
        raise click.ClickException(
            f"Invalid parent_id {parent_id!r}: must match ^node_\\d+$ "
            f"(path-traversal defence; only 'node_<digits>' accepted)"
        )


def _parse_held_out_cells(
    spec: str,
) -> tuple[tuple[str, str, str, str], ...]:
    """Parse --held-out-cells 'id1:dataset:encoder:task,...' into manifest tuple.

    Validates each entry has exactly 4 colon-separated parts.
    """
    cells: list[tuple[str, str, str, str]] = []
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) != 4:
            raise click.ClickException(
                f"--held-out-cells entry {entry!r}: expected "
                f"'cell_id:dataset:encoder:task' (4 colon-separated parts); "
                f"got {len(parts)} part(s)"
            )
        cells.append(tuple(p.strip() for p in parts))  # type: ignore[arg-type]
    if not cells:
        raise click.ClickException("--held-out-cells must be non-empty")
    return tuple(cells)


@gate_group.command("register-manifest")
@click.argument("parent_id")
@click.option(
    "--strategy",
    type=click.Choice(["stratified", "random", "operator-curated"], case_sensitive=False),
    default="stratified",
    show_default=True,
    help=(
        "Held-out cell selection strategy (D-145, O-03). "
        "'stratified' (recommended) ensures each dataset×encoder combination "
        "is represented. 'operator-curated' requires --held-out-cells. "
        "Auto-selection (--auto-select) is forthcoming (plan 12 calibration pilot)."
    ),
)
@click.option("--K", "K", type=int, default=None,
              help="Min cells that must pass (D-141). Default: gate.K from "
                   "automil/config.yaml, else 2.")
@click.option(
    "--p-threshold",
    type=float,
    default=None,
    help="Pre-Bonferroni alpha; effective threshold is p_threshold/K (O-02). "
         "Default: gate.p_threshold from automil/config.yaml, else 0.05.",
)
@click.option(
    "--bootstrap-reps",
    type=int,
    default=None,
    help="GTE-04: BCa bootstrap resamples for the CI. Default: "
         "gate.bootstrap_reps from automil/config.yaml, else 1000.",
)
@click.option(
    "--held-out-cells",
    default=None,
    help=(
        "Comma-separated 'cell_id:dataset:encoder:task' tuples. "
        "Required for --strategy operator-curated; optional for stratified/random "
        "if --auto-select is provided. "
        "Example: 'abc12345:ccrcc:uni_v2:high_grade,def67890:clwd:ctranspath:subtype'"
    ),
)
@click.option(
    "--auto-select",
    type=int,
    default=None,
    help=(
        "For --strategy stratified/random: auto-select N cells. "
        "STUB — calibration pilot (plan 12) will populate the recipe. "
        "Use --held-out-cells with an explicit list for now."
    ),
)
def register_manifest_cmd(
    parent_id: str,
    strategy: str,
    K: int,
    p_threshold: float | None,
    bootstrap_reps: int | None,
    held_out_cells: str | None,
    auto_select: int | None,
) -> None:
    """Pre-register a gate manifest for PARENT_ID (D-138).

    Validates inputs, atomically writes the manifest, and commits it to git in
    one operation. Refuses to overwrite an existing manifest — use
    retire-manifest first to supersede a previous registration.

    The manifest records the held-out cells, K, p_threshold, and bootstrap_reps
    BEFORE any candidate is nominated or evaluated (pre-registration defence,
    Pitfall 6 / D-138).
    """
    from automil.cli._helpers import _find_automil_dir, _find_git_root  # lazy
    from automil.gate import GateManifest, validate_manifest_dict, write_manifest_committed
    from automil.gate.manifest import SCHEMA_VERSION
    import dataclasses

    _validate_parent_id(parent_id)

    adir = _find_automil_dir()

    # CFG-1: the `gate:` block in automil/config.yaml was read by nothing — every
    # value came from this command's CLI defaults. For a PRE-REGISTERED gate that
    # is backwards: the thresholds are meant to be fixed in git before any
    # held-out evaluation runs, so they must come from the committed config, with
    # an explicit flag as the override rather than the only source.
    import yaml as _yaml

    from automil.gate.config import load_gate_config

    _cfg_path = adir / "config.yaml"
    _raw = _yaml.safe_load(_cfg_path.read_text()) if _cfg_path.exists() else {}
    _gate_cfg = load_gate_config(_raw)
    if K is None:
        K = _gate_cfg.K
    if p_threshold is None:
        p_threshold = _gate_cfg.p_threshold
    if bootstrap_reps is None:
        bootstrap_reps = _gate_cfg.bootstrap_reps
    git_root = _find_git_root()

    # Validate parent_id exists in graph (if graph.json exists)
    graph_path = adir / "graph.json"
    if graph_path.exists():
        try:
            data = json.loads(graph_path.read_text())
            if parent_id not in data.get("nodes", {}):
                raise click.ClickException(
                    f"parent_id {parent_id!r} not found in {graph_path}. "
                    f"Run 'automil submit' and wait for the parent to execute "
                    f"before registering its manifest."
                )
        except (json.JSONDecodeError, OSError) as exc:
            raise click.ClickException(
                f"Could not read {graph_path}: {exc}"
            ) from exc

    # Resolve held-out cells list
    strategy_lower = strategy.lower()
    if strategy_lower == "operator-curated":
        if not held_out_cells:
            raise click.ClickException(
                "--strategy operator-curated requires --held-out-cells "
                "'cell_id1:ds:enc:task,cell_id2:...'"
            )
        cells = _parse_held_out_cells(held_out_cells)
    else:
        if held_out_cells:
            cells = _parse_held_out_cells(held_out_cells)
        elif auto_select is not None:
            raise click.ClickException(
                f"--strategy {strategy} --auto-select {auto_select}: "
                f"automatic cell selection is forthcoming (calibration pilot, plan 12). "
                f"Use --held-out-cells with an explicit comma-separated list for now."
            )
        else:
            raise click.ClickException(
                "Provide either --held-out-cells or --auto-select N "
                "(--auto-select requires calibration pilot, plan 12)."
            )

    manifest = GateManifest(
        parent_id=parent_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        git_committed_at_sha="PENDING",
        held_out_cells=cells,
        K=K,
        p_threshold=p_threshold,
        bootstrap_reps=bootstrap_reps,
        win_definition=(
            "delta_primary_value > 0 AND p <= alpha/K (paired Wilcoxon, "
            "Bonferroni-corrected) AND ci_low > 0 (BCa bootstrap, 95%)"
        ),
        schema_version=SCHEMA_VERSION,
    )

    # Pre-write validation surfaces errors before git operations
    try:
        validate_manifest_dict(dataclasses.asdict(manifest))
    except ValueError as exc:
        raise click.ClickException(f"Manifest validation failed: {exc}") from exc

    manifests_dir = adir / "gate"
    try:
        sha = write_manifest_committed(manifest, manifests_dir, git_root)
    except FileExistsError as exc:
        raise click.ClickException(str(exc)) from exc
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(
        f"Registered manifest for {parent_id} "
        f"(K={K}, p<{p_threshold}, "
        f"held_out={len(cells)} cells, "
        f"sha={sha[:12]})"
    )


@gate_group.command("retire-manifest")
@click.argument("parent_id")
@click.option(
    "--reason",
    required=True,
    help="Why this manifest is being superseded (recorded for audit trail).",
)
def retire_manifest_cmd(parent_id: str, reason: str) -> None:
    """Retire an active manifest, writing a .retired.gate_manifest.json + git commit.

    The retirement reason is recorded in the retired file for audit purposes.
    After retiring, a new manifest can be registered with
    'automil gate register-manifest PARENT_ID ...'.
    """
    from automil.cli._helpers import _find_automil_dir, _find_git_root  # lazy
    from automil.gate import retire_manifest

    _validate_parent_id(parent_id)
    adir = _find_automil_dir()
    git_root = _find_git_root()
    manifests_dir = adir / "gate"

    try:
        sha = retire_manifest(parent_id, reason, manifests_dir, git_root)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    except FileExistsError as exc:
        raise click.ClickException(str(exc)) from exc
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(
        f"Retired manifest for {parent_id} "
        f"(reason={reason!r}, sha={sha[:12]})"
    )


@gate_group.command("status")
@click.argument("parent_id", required=False)
def gate_status_cmd(parent_id: str | None) -> None:
    """Show manifest details + per-candidate gate state.

    With PARENT_ID: detailed view of the manifest for that node.
    Without PARENT_ID: list all active manifests (excludes retired).
    """
    from automil.cli._helpers import _find_automil_dir  # lazy
    from automil.gate import load_manifest

    adir = _find_automil_dir()
    manifests_dir = adir / "gate"

    if not manifests_dir.exists():
        click.echo("No manifests registered.")
        return

    if parent_id:
        _validate_parent_id(parent_id)
        try:
            m = load_manifest(parent_id, manifests_dir)
        except FileNotFoundError as exc:
            raise click.ClickException(str(exc)) from exc

        click.echo(f"parent_id:            {m.parent_id}")
        click.echo(f"created_at:           {m.created_at}")
        click.echo(f"git_committed_at_sha: {m.git_committed_at_sha}")
        click.echo(f"K:                    {m.K}")
        click.echo(f"p_threshold:          {m.p_threshold}")
        click.echo(f"bootstrap_reps:       {m.bootstrap_reps}")
        click.echo(f"held_out_cells ({len(m.held_out_cells)}):")
        for cell_id, ds, enc, task in m.held_out_cells:
            click.echo(f"  {cell_id[:12]}  {ds:<10} {enc:<14} {task}")
    else:
        all_files = sorted(manifests_dir.glob("*.gate_manifest.json"))
        active_files = [
            f for f in all_files
            if not f.name.endswith(".retired.gate_manifest.json")
        ]
        click.echo(f"Active manifests ({len(active_files)}):")
        for f in active_files:
            try:
                d = json.loads(f.read_text())
                click.echo(
                    f"  {d['parent_id']:<12}  "
                    f"K={d['K']:<3}  "
                    f"p={d['p_threshold']}  "
                    f"held_out={len(d['held_out_cells'])} cells"
                )
            except (json.JSONDecodeError, KeyError):
                click.echo(f"  {f.name}  (parse error)")


@gate_group.command("stats")
def gate_stats_cmd() -> None:
    """Show promotion_rate (30d window) + gate-health diagnostic (D-144 / GTE-06).

    Reads graph.json to compute how many nominated candidates were promoted
    in the past 30 days, and interprets the rate via diagnose_gate_health.
    """
    from automil.cli._helpers import _find_automil_dir, _load_technique_map  # lazy
    from automil.gate.stats import diagnose_gate_health
    from automil.graph import ExperimentGraph

    adir = _find_automil_dir()
    graph_path = adir / "graph.json"
    if not graph_path.exists():
        click.echo("No graph.json found — no experiments have run yet.")
        return

    graph = ExperimentGraph(path=str(graph_path), technique_map=_load_technique_map(adir))
    rate_30 = graph.promotion_rate(days=30)
    nominated_nodes = graph.nominations_in_window(days=30)
    nominated = len(nominated_nodes)
    promoted = sum(
        1 for n in nominated_nodes if n.get("status") == "registered"
    )
    click.echo(
        f"Promotion rate (30d): {rate_30:.1%} "
        f"({promoted}/{nominated} nominated candidates)"
    )
    click.echo(f"Health: {diagnose_gate_health(rate_30)}")
