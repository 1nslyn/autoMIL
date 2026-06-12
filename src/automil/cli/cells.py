"""cells subcommand group: automil cells migrate (REC-04 / D-15)."""
from __future__ import annotations

import click

from automil.cli import main
from automil.cli._helpers import _find_automil_dir


@main.group()
def cells():
    """Budget-cell management commands."""


@cells.command("migrate")
@click.option("--mil-model", required=True,
              help="MIL model name to assign to all existing cells (D-15, REC-04).")
@click.option("--dry-run", is_flag=True, default=False,
              help="Print migration summary without writing any files.")
def cells_migrate(mil_model: str, dry_run: bool) -> None:
    """Re-key budget cells from parent_id → mil_model (D-15).

    Existing cells/*.json are re-keyed to (dataset, encoder, mil_model). When two
    cells would map to the same new key, their budgets are merged without
    double-counting (agent_active: sum consumed_active_seconds; wall_clock: keep
    earliest started_at). Use --dry-run to preview without writing.
    """
    adir = _find_automil_dir()
    from automil.cells.migrate import migrate_cells

    summaries = migrate_cells(adir / "cells", mil_model=mil_model, dry_run=dry_run)

    for s in summaries:
        action = s["action"]
        old_short = s["old_id"][:8]
        new_short = s["new_id"][:8]
        click.echo(f"  {action}: {old_short} → {new_short}")

    n = len(summaries)
    if dry_run:
        click.echo(f"Dry run: {n} cell(s) would be processed.")
    else:
        click.echo(f"Migrated {n} cell(s).")
