"""REC-04 / D-15: budget-cell back-fill migration without double-counting.

D-15: Existing parent-keyed cells migrate via a back-fill helper
      (`automil cells migrate`) that re-derives mil_model for existing executed
      nodes and merges their elapsed budget into the new (dataset, encoder, mil_model)
      cell without double-counting.

Also tests the read_cell compat shim: legacy cells keyed by parent_id load without TypeError.

RED until Plan 02 ships read_cell compat shim + Plan 05 ships migrate_cells.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import pytest

from automil.cells.state import Cell, CellStatus, make_cell_id, write_cell


def _make_cell(
    cell_id: str,
    dataset: str = "test_ds",
    encoder: str = "enc1",
    parent_id: str | None = None,
    mil_model: str | None = None,
    mode: str = "agent_active",
    consumed_active_seconds: float = 0.0,
    started_at: float | None = None,
    budget_seconds: int = 21600,
    safety_buffer_seconds: int = 1800,
) -> dict:
    """Build a raw cell dict for writing (bypasses Cell dataclass to test legacy keys)."""
    d = {
        "cell_id": cell_id,
        "dataset": dataset,
        "encoder": encoder,
        "started_at": started_at if started_at is not None else time.time(),
        "budget_seconds": budget_seconds,
        "safety_buffer_seconds": safety_buffer_seconds,
        "status": CellStatus.ACTIVE.value,
        "mode": mode,
        "idle_grace_seconds": 300,
        "consumed_active_seconds": consumed_active_seconds,
        "last_tick_at": None,
    }
    # Allow either legacy parent_id or new mil_model key
    if mil_model is not None:
        d["mil_model"] = mil_model
    if parent_id is not None:
        d["parent_id"] = parent_id
    return d


def test_legacy_cell_loads(tmp_path: Path) -> None:
    """D-15 compat shim: a cell JSON with 'parent_id' (no 'mil_model') must load without TypeError.

    read_cell must map parent_id → mil_model for backward compat.
    RED until Plan 02 ships the read_cell compat shim.
    """
    from automil.cells.state import read_cell

    cell_path = tmp_path / "abc1234567890123.json"
    legacy_dict = _make_cell(
        cell_id="abc1234567890123",
        parent_id="node_0042",
        # no mil_model key
    )
    cell_path.write_text(json.dumps(legacy_dict))

    # Must not raise TypeError — compat shim maps parent_id → mil_model
    cell = read_cell(cell_path)
    assert cell.mil_model == "node_0042", (
        f"D-15 compat shim not implemented: expected cell.mil_model='node_0042', "
        f"got {getattr(cell, 'mil_model', 'ATTR_MISSING')!r}."
    )


def test_agent_active_merge_sums_consumed(tmp_path: Path) -> None:
    """D-15: two agent_active cells for same (dataset, encoder, mil_model) → consumed summed.

    migrate_cells must merge elapsed budget without double-counting.
    RED until Plan 05 ships migrate_cells.
    """
    from automil.cells.migrate import migrate_cells  # RED until Plan 05

    cells_dir = tmp_path / "cells"
    cells_dir.mkdir()

    mil_model = "clam_sb"

    # Two cells with same (dataset, encoder, mil_model) — different cell_ids (old keying)
    cell1_id = "aaaa111122223333"
    cell2_id = "bbbb444455556666"

    cell1 = _make_cell(
        cell_id=cell1_id,
        mil_model=mil_model,
        mode="agent_active",
        consumed_active_seconds=3600.0,
    )
    cell2 = _make_cell(
        cell_id=cell2_id,
        mil_model=mil_model,
        mode="agent_active",
        consumed_active_seconds=1800.0,
    )
    (cells_dir / f"{cell1_id}.json").write_text(json.dumps(cell1))
    (cells_dir / f"{cell2_id}.json").write_text(json.dumps(cell2))

    summaries = migrate_cells(cells_dir, mil_model=mil_model, dry_run=False)

    # Verify merge happened
    merged_actions = [s for s in summaries if s.get("action") == "merge"]
    assert merged_actions, (
        "D-15 not implemented: migrate_cells did not merge duplicate cells. "
        "Two cells with same mil_model must be merged."
    )

    # Check the resulting cell has summed consumed_active_seconds
    from automil.cells.state import read_cell
    cell_files = list(cells_dir.glob("*.json"))
    assert len(cell_files) == 1, (
        f"Expected 1 merged cell, got {len(cell_files)}: {[f.name for f in cell_files]}"
    )
    merged = read_cell(cell_files[0])
    expected_consumed = 3600.0 + 1800.0
    assert abs(merged.consumed_active_seconds - expected_consumed) < 0.01, (
        f"D-15: merged cell consumed_active_seconds={merged.consumed_active_seconds}, "
        f"expected {expected_consumed} (sum of both cells)."
    )


def test_wall_clock_merge_keeps_earliest_started_at(tmp_path: Path) -> None:
    """D-15: two wall_clock cells for same mil_model → merged cell has min(started_at).

    RED until Plan 05 ships migrate_cells.
    """
    from automil.cells.migrate import migrate_cells  # RED until Plan 05

    cells_dir = tmp_path / "cells"
    cells_dir.mkdir()

    mil_model = "abmil"
    now = time.time()
    earlier = now - 7200.0   # 2 hours ago
    later = now - 3600.0     # 1 hour ago

    cell1_id = "cccc111122223333"
    cell2_id = "dddd444455556666"

    cell1 = _make_cell(
        cell_id=cell1_id,
        mil_model=mil_model,
        mode="wall_clock",
        started_at=earlier,
    )
    cell2 = _make_cell(
        cell_id=cell2_id,
        mil_model=mil_model,
        mode="wall_clock",
        started_at=later,
    )
    (cells_dir / f"{cell1_id}.json").write_text(json.dumps(cell1))
    (cells_dir / f"{cell2_id}.json").write_text(json.dumps(cell2))

    summaries = migrate_cells(cells_dir, mil_model=mil_model, dry_run=False)

    from automil.cells.state import read_cell
    cell_files = list(cells_dir.glob("*.json"))
    assert len(cell_files) == 1, f"Expected 1 merged cell, got {len(cell_files)}"
    merged = read_cell(cell_files[0])

    assert abs(merged.started_at - earlier) < 0.01, (
        f"D-15: merged wall_clock cell started_at={merged.started_at}, "
        f"expected earliest={earlier}. Min started_at must be kept."
    )


def test_dry_run_does_not_write(tmp_path: Path) -> None:
    """D-15: dry_run=True returns migration summaries but leaves cell files unchanged.

    RED until Plan 05 ships migrate_cells.
    """
    from automil.cells.migrate import migrate_cells  # RED until Plan 05

    cells_dir = tmp_path / "cells"
    cells_dir.mkdir()

    mil_model = "transmil"
    cell_id = "eeee111122223333"
    cell = _make_cell(cell_id=cell_id, mil_model=mil_model, mode="agent_active")
    cell_path = cells_dir / f"{cell_id}.json"
    cell_path.write_text(json.dumps(cell))

    original_content = cell_path.read_text()
    original_files = set(f.name for f in cells_dir.glob("*.json"))

    summaries = migrate_cells(cells_dir, mil_model=mil_model, dry_run=True)

    # Dry run: files must be unchanged
    assert set(f.name for f in cells_dir.glob("*.json")) == original_files, (
        "D-15: dry_run=True modified cell files. Must be a no-op on disk."
    )
    assert cell_path.read_text() == original_content, (
        "D-15: dry_run=True overwrote cell file content."
    )

    # But summaries must still be returned
    assert isinstance(summaries, list), (
        f"D-15: migrate_cells(dry_run=True) must return a list of summaries, got {type(summaries)}"
    )
