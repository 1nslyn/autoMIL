"""Shared fixtures and helpers for automil.cells tests (CAP-01..06).

Provides:
- ``cells_dir`` fixture — isolated tmp directory for cell JSON files
- ``make_cell`` helper — plain function (not a fixture) for constructing
  ``Cell`` instances with arbitrary overrides in test bodies

Tests import: from tests.cells.conftest import make_cell
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from automil.cells.state import Cell, CellStatus


@pytest.fixture
def cells_dir(tmp_path: Path) -> Path:
    """Return an isolated tmp directory for cell JSON files."""
    d = tmp_path / "cells"
    d.mkdir()
    return d


def make_cell(
    cell_id: str = "abc1234567890123",
    dataset: str = "test",
    encoder: str = "enc",
    mil_model: str = "node_0001",
    started_at: float | None = None,
    budget_seconds: int = 21600,
    safety_buffer_seconds: int = 1800,
    status: CellStatus = CellStatus.ACTIVE,
    mode: str = "wall_clock",
    idle_grace_seconds: int = 300,
    consumed_active_seconds: float = 0.0,
    last_tick_at: float | None = None,
) -> Cell:
    """Construct a ``Cell`` with sensible defaults; override any field as needed.

    Plain function (not a fixture) so tests can call it with arbitrary keyword
    args inline without fixture injection ceremony. ``mode`` defaults to the
    legacy ``"wall_clock"`` so existing cap-state tests keep their now-started_at
    semantics; activity-gated tests pass ``mode="agent_active"``.
    """
    return Cell(
        cell_id=cell_id,
        dataset=dataset,
        encoder=encoder,
        mil_model=mil_model,
        started_at=started_at if started_at is not None else time.time(),
        budget_seconds=budget_seconds,
        safety_buffer_seconds=safety_buffer_seconds,
        status=status,
        mode=mode,
        idle_grace_seconds=idle_grace_seconds,
        consumed_active_seconds=consumed_active_seconds,
        last_tick_at=last_tick_at,
    )
