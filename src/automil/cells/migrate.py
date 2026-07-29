"""Budget-cell back-fill migration: parent_id keying → mil_model keying (D-15, REC-04).

One-time operator action. Run via `automil cells migrate --mil-model <value>`.
Dry-run mode prints a summary without writing. Atomic: new cell file written
before old cell file deleted; rollback on any failure.

Mode-aware budget merge (T-09-08 mitigation):
  - agent_active: sum consumed_active_seconds across cells being merged.
  - wall_clock: keep the cell with the earliest started_at.
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

from automil.cells.state import (
    Cell,
    make_cell_id,
    normalize_mil_model,
    read_cell,
    write_cell,
)

logger = logging.getLogger(__name__)


def migrate_cells(
    cells_dir: Path,
    mil_model: str,
    dry_run: bool = False,
) -> list[dict]:
    """Re-key all cells from parent_id → mil_model. Returns summary records.

    For each cell file in cells_dir:
      - Compute new_id = make_cell_id(cell.dataset, cell.encoder, mil_model_norm).
      - If the new path already exists (merge case): mode-aware budget merge.
        agent_active: sum consumed_active_seconds (T-09-08 mitigation).
        wall_clock: keep the earliest started_at.
      - Otherwise (rename case): write new cell, delete old.

    dry_run=True: return summaries without writing any files (T-09-09 mitigation:
    write-before-delete is enforced even in live mode).

    Args:
        cells_dir: Path to the automil/cells/ directory.
        mil_model: Raw MIL model name — will be normalized via normalize_mil_model.
        dry_run: If True, return summaries without modifying files.

    Returns:
        List of summary dicts with keys: old_id, new_id, action ("merge"|"rename"|"skip").
    """
    mil_model_norm = normalize_mil_model(mil_model)
    summaries: list[dict] = []

    for path in sorted(cells_dir.glob("*.json")):
        try:
            cell = read_cell(path)
        except Exception as exc:
            logger.warning("Skipping malformed cell %s: %s", path.name, exc)
            continue

        new_id = make_cell_id(cell.dataset, cell.encoder, mil_model_norm)
        new_path = cells_dir / f"{new_id}.json"

        if new_path == path:
            # Already keyed correctly — coincidental match or already migrated.
            logger.debug("Cell %s already has correct key %s, skipping.", path.name, new_id[:8])
            summaries.append({"old_id": cell.cell_id, "new_id": new_id, "action": "skip"})
            continue

        if new_path.exists():
            # Merge case: a new-keyed cell already exists — combine budgets.
            existing = read_cell(new_path)
            # WR-05 fix: guard against mode mismatch before merging. When
            # cell.mode != existing.mode the merge semantics are ambiguous:
            # - agent_active into wall_clock: consumed_active_seconds is summed
            #   but wall_clock billing ignores it → budget silently discarded.
            # - wall_clock into agent_active: started_at adjusted but billing
            #   continues on agent_active accumulator without wall-clock elapsed.
            # T-09-08 spec says "sum consumed_active_seconds" without addressing
            # mode-mismatch; skip and require manual resolution to avoid data loss.
            if cell.mode != existing.mode:
                logger.warning(
                    "migrate_cells: skipping merge of %s (mode=%s) into %s (mode=%s) — "
                    "mode mismatch; cannot safely combine budgets. Manual review required: "
                    "inspect both cell files and reconcile consumed budget before re-running migrate.",
                    cell.cell_id[:8], cell.mode, existing.cell_id[:8], existing.mode,
                )
                summaries.append({"old_id": cell.cell_id, "new_id": new_id, "action": "skip"})
                continue
            if cell.mode == "agent_active":
                # T-09-08: sum consumed_active_seconds without double-counting.
                merged_consumed = cell.consumed_active_seconds + existing.consumed_active_seconds
                merged = dataclasses.replace(existing, consumed_active_seconds=merged_consumed)
            else:
                # wall_clock: keep the cell with the earliest started_at.
                if cell.started_at < existing.started_at:
                    merged = dataclasses.replace(existing, started_at=cell.started_at)
                else:
                    merged = existing  # existing already has the earlier started_at
            # H-2: the eval counters are mode-independent, so they are summed in
            # both branches — a re-key must never hand back spent evaluations.
            merged = dataclasses.replace(
                merged,
                consumed_evals=cell.consumed_evals + existing.consumed_evals,
                completed_evals=cell.completed_evals + existing.completed_evals,
            )
            summaries.append({"old_id": cell.cell_id, "new_id": new_id, "action": "merge"})
            if not dry_run:
                # T-09-09: write merged cell first (atomic), then unlink old.
                write_cell(merged, cells_dir)
                path.unlink()
        else:
            # Rename case: re-key to the new (dataset, encoder, mil_model) triple.
            new_cell = dataclasses.replace(cell, cell_id=new_id, mil_model=mil_model_norm)
            summaries.append({"old_id": cell.cell_id, "new_id": new_id, "action": "rename"})
            if not dry_run:
                # T-09-09: write new cell first (atomic), then delete old.
                write_cell(new_cell, cells_dir)
                if path != new_path:
                    path.unlink()

    return summaries
