"""cell subgroup: cell budget status and list commands (CAP-06 / D-125)."""
from __future__ import annotations

import json
from pathlib import Path

import click

from automil.cli import main


@main.group("cell")
def cell_group() -> None:
    """Cell budget-cap management commands."""
    pass


def _count_running_in_cell(cell_id: str) -> int:
    """Count running experiments tagged with metadata.cell_id == cell_id.

    Reads automil/orchestrator/running/*.json directly so the CLI can
    report state without instantiating an ExperimentOrchestrator.
    """
    from automil.cli._helpers import _find_automil_dir  # lazy

    try:
        adir = _find_automil_dir()
    except click.ClickException:
        return 0
    running_dir = adir / "orchestrator" / "running"
    if not running_dir.exists():
        return 0
    n = 0
    # D-169: rglob to traverse all backend subdirs (running/local/, running/slurm/, etc.)
    for f in running_dir.rglob("*.json"):
        try:
            spec = json.loads(f.read_text())
            if spec.get("metadata", {}).get("cell_id") == cell_id:
                n += 1
        except (json.JSONDecodeError, OSError):
            continue
    return n


def _format_consumed(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h, rem = divmod(int(max(0, seconds)), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_budget(seconds: int) -> str:
    """Format budget seconds as HH:MM:SS."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_evals(cell) -> str:
    """Render the eval axis as consumed/budget ('-' when the cell is time-only)."""
    budget = cell.eval_budget if cell.eval_budget is not None else "-"
    return f"{cell.consumed_evals}/{budget}"


@cell_group.command("status")
@click.argument("cell_id", required=False)
@click.option("--no-header", is_flag=True, default=False, help="Suppress header row.")
def cell_status(cell_id: str | None, no_header: bool) -> None:
    """Show budget state for one cell (or all cells if CELL_ID omitted)."""
    from datetime import datetime

    from automil.cells import consumed_seconds, get_cell, list_cells  # lazy

    if cell_id is not None:
        # Tolerant prefix match: allow operator to type a short prefix
        cell = get_cell(cell_id) if len(cell_id) == 16 else None
        if cell is None:
            # Try short-prefix match across all cells
            matches = [c for c in list_cells() if c.cell_id.startswith(cell_id)]
            if len(matches) == 0:
                raise click.ClickException(f"No cell found matching id={cell_id!r}")
            if len(matches) > 1:
                raise click.ClickException(
                    f"Ambiguous prefix {cell_id!r}: matched {len(matches)} cells; "
                    f"please use the full 16-char cell_id."
                )
            cell = matches[0]
        cells = [cell]
    else:
        cells = list_cells()

    if not cells:
        click.echo("(no cells)")
        return

    header = (
        f"{'cell_id':<8}  {'dataset':<10}  {'encoder':<10}  {'mil_model':<10}  "
        f"{'started':<19}  {'consumed/budget':<19}  {'evals':<9}  {'usable':<6}  "
        f"{'status':<14}  {'running':<7}"
    )
    if not no_header:
        click.echo(header)
        click.echo("-" * len(header))
    for cell in cells:
        consumed = consumed_seconds(cell)
        consumed_str = _format_consumed(consumed)
        budget_str = _format_budget(cell.budget_seconds)
        cb = f"{consumed_str}/{budget_str}"
        started_str = datetime.fromtimestamp(cell.started_at).strftime("%Y-%m-%d %H:%M:%S")
        running_count = _count_running_in_cell(cell.cell_id)
        click.echo(
            f"{cell.cell_id[:8]:<8}  {cell.dataset[:10]:<10}  {cell.encoder[:10]:<10}  "
            f"{cell.mil_model[:10]:<10}  {started_str:<19}  "
            f"{cb:<19}  {_format_evals(cell):<9}  {cell.completed_evals:<6}  "
            f"{cell.status.value:<14}  {running_count:<7}"
        )


@cell_group.command("list")
@click.option("--no-header", is_flag=True, default=False, help="Pipe-friendly: no header.")
def cell_list(no_header: bool) -> None:
    """Short-form cell listing (cell_id, status, consumed/budget)."""
    from automil.cells import consumed_seconds, list_cells  # lazy

    cells = list_cells()
    if not cells:
        click.echo("(no cells)")
        return
    if not no_header:
        click.echo(
            f"{'cell_id':<8}  {'status':<14}  {'consumed/budget':<19}  {'evals':<9}"
        )
        click.echo("-" * 56)
    for cell in cells:
        consumed_str = _format_consumed(consumed_seconds(cell))
        budget_str = _format_budget(cell.budget_seconds)
        cb = f"{consumed_str}/{budget_str}"
        click.echo(
            f"{cell.cell_id[:8]:<8}  {cell.status.value:<14}  {cb:<19}  "
            f"{_format_evals(cell):<9}"
        )
