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


def _consumed_seconds(cell, activity_inspections) -> float:
    """Resolve current cap usage from the cell's authoritative time source."""
    from automil.cells import consumed_seconds

    if cell.mode == "wall_clock":
        return consumed_seconds(cell)
    inspection = activity_inspections[cell.cell_id]
    if inspection.error is not None:
        raise click.ClickException(
            f"Cell {cell.cell_id[:8]} {inspection.error}"
        )
    return consumed_seconds(
        cell, agent_active_seconds=inspection.active_seconds,
    )


def _echo_registry_errors(errors) -> None:
    """Render invalid journals after healthy rows without disguising them."""
    for error in errors:
        click.echo(f"INVALID  {error.path.name}: {error.message}")


def _finish_inspection(errors: list[str]) -> None:
    """Return a failing CLI status only after every inspectable row was shown."""
    if errors:
        raise click.ClickException(
            f"{len(errors)} cell journal issue(s) require attention"
        )


@cell_group.command("status")
@click.argument("cell_id", required=False)
@click.option("--no-header", is_flag=True, default=False, help="Suppress header row.")
def cell_status(cell_id: str | None, no_header: bool) -> None:
    """Show budget state for one cell (or all cells if CELL_ID omitted)."""
    from datetime import datetime

    from automil.cells import scan_cells  # lazy

    scan = scan_cells()
    registry_errors = list(scan.errors)

    if cell_id is not None:
        # Tolerant prefix match: allow operator to type a short prefix
        matches = [c for c in scan.cells if c.cell_id.startswith(cell_id)]
        invalid_matches = [
            error for error in registry_errors
            if error.path.stem.startswith(cell_id)
        ]
        if len(matches) == 0:
            if invalid_matches:
                _echo_registry_errors(invalid_matches)
                _finish_inspection([str(error) for error in invalid_matches])
            raise click.ClickException(f"No cell found matching id={cell_id!r}")
        if len(matches) > 1:
            raise click.ClickException(
                f"Ambiguous prefix {cell_id!r}: matched {len(matches)} cells; "
                f"please use the full 16-char cell_id."
            )
        cell = matches[0]
        cells = [cell]
    else:
        cells = list(scan.cells)

    if not cells and not registry_errors:
        click.echo("(no cells)")
        return

    from automil.cli._activity_inspection import inspect_agent_activity
    from automil.cli._helpers import _find_automil_dir

    activity_inspections = inspect_agent_activity(_find_automil_dir(), cells)

    header = (
        f"{'cell_id':<8}  {'dataset':<10}  {'encoder':<10}  {'mil_model':<10}  "
        f"{'started':<19}  {'consumed/budget':<19}  {'evals':<9}  {'usable':<6}  "
        f"{'status':<14}  {'running':<7}"
    )
    if not no_header:
        click.echo(header)
        click.echo("-" * len(header))
    inspection_errors: list[str] = []
    for cell in cells:
        try:
            consumed_str = _format_consumed(
                _consumed_seconds(cell, activity_inspections)
            )
        except click.ClickException as exc:
            consumed_str = "DEGRADED"
            inspection_errors.append(str(exc))
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
    if cell_id is None:
        _echo_registry_errors(registry_errors)
        inspection_errors.extend(str(error) for error in registry_errors)
    for message in inspection_errors:
        if "activity " in message:
            click.echo(f"DEGRADED  {message}")
    _finish_inspection(inspection_errors)


@cell_group.command("list")
@click.option("--no-header", is_flag=True, default=False, help="Pipe-friendly: no header.")
def cell_list(no_header: bool) -> None:
    """Short-form cell listing (cell_id, status, consumed/budget)."""
    from automil.cells import scan_cells  # lazy

    scan = scan_cells()
    cells = list(scan.cells)
    if not cells and not scan.errors:
        click.echo("(no cells)")
        return
    from automil.cli._activity_inspection import inspect_agent_activity
    from automil.cli._helpers import _find_automil_dir

    activity_inspections = inspect_agent_activity(_find_automil_dir(), cells)
    if not no_header:
        click.echo(
            f"{'cell_id':<8}  {'status':<14}  {'consumed/budget':<19}  {'evals':<9}"
        )
        click.echo("-" * 56)
    inspection_errors: list[str] = []
    for cell in cells:
        try:
            consumed_str = _format_consumed(
                _consumed_seconds(cell, activity_inspections)
            )
        except click.ClickException as exc:
            consumed_str = "DEGRADED"
            inspection_errors.append(str(exc))
        budget_str = _format_budget(cell.budget_seconds)
        cb = f"{consumed_str}/{budget_str}"
        click.echo(
            f"{cell.cell_id[:8]:<8}  {cell.status.value:<14}  {cb:<19}  "
            f"{_format_evals(cell):<9}"
        )
    _echo_registry_errors(scan.errors)
    inspection_errors.extend(str(error) for error in scan.errors)
    for message in inspection_errors:
        if "activity " in message:
            click.echo(f"DEGRADED  {message}")
    _finish_inspection(inspection_errors)
