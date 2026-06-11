"""Cell CRUD + lazy registry (CAP-01, CAP-05 / D-107, D-116, D-134).

Cells are persisted to automil/cells/<cell_id>.json. The registry is
"singleton-ish" — module-level functions reading/writing the on-disk
state. No in-memory cache (would conflict with daemon-restart safety).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import click

from automil.cells.state import (
    Cell,
    CellStatus,
    make_cell_id,
    read_cell,
    write_cell,
)
from automil.cli._helpers import _find_automil_dir

logger = logging.getLogger(__name__)


def _cells_dir() -> Path:
    """Locate automil/cells/ relative to the automil/ overlay dir."""
    return _find_automil_dir() / "cells"


def get_or_create_cell(
    dataset: str,
    encoder: str,
    mil_model: str,
    budget_seconds: int,
    safety_buffer_seconds: int,
    idle_grace_seconds: int = 300,
    mode: str = "agent_active",
) -> Cell:
    """Return existing cell or create a new one (lazy + idempotent, D-116, REC-04).

    D-134: all cap parameters (budget_seconds, safety_buffer_seconds,
    idle_grace_seconds, mode) apply ONLY when this call CREATES the cell. If the
    cell already exists, the persisted values are kept and any override is logged
    at INFO. Allowing later submits to extend a cell's budget = sandbagging vector.

    Args:
        dataset: e.g. "ccrcc" — from automil/config.yaml.
        encoder: e.g. "uni-v2" — from automil/config.yaml.
        mil_model: MIL model identifier for budget cell keying (D-13). Must be
            pre-normalized via normalize_mil_model() before calling (D-14).
            Graph parent lineage stays separate — re-parenting does not fork the budget.
        budget_seconds: cap; honored only on creation.
        safety_buffer_seconds: refusing-new lead time; honored only on creation.
        idle_grace_seconds: agent-active idle grace; honored only on creation.
        mode: "agent_active" or "wall_clock"; honored only on creation.
    """
    cells_dir = _cells_dir()
    cell_id = make_cell_id(dataset, encoder, mil_model)
    path = cells_dir / f"{cell_id}.json"
    if path.exists():
        cell = read_cell(path)
        if (cell.budget_seconds != budget_seconds
                or cell.safety_buffer_seconds != safety_buffer_seconds
                or cell.idle_grace_seconds != idle_grace_seconds
                or cell.mode != mode):
            logger.info(
                "Cell %s already open (budget=%ds buffer=%ds idle_grace=%ds mode=%s); "
                "ignoring override (budget=%ds buffer=%ds idle_grace=%ds mode=%s) per D-134.",
                cell_id[:8], cell.budget_seconds, cell.safety_buffer_seconds,
                cell.idle_grace_seconds, cell.mode,
                budget_seconds, safety_buffer_seconds, idle_grace_seconds, mode,
            )
        return cell

    # First submit for this (dataset, encoder, mil_model) triple → open the cell.
    cell = Cell(
        cell_id=cell_id,
        dataset=dataset,
        encoder=encoder,
        mil_model=mil_model,
        started_at=time.time(),  # set ONCE at creation; never updated (D-111)
        budget_seconds=budget_seconds,
        safety_buffer_seconds=safety_buffer_seconds,
        status=CellStatus.ACTIVE,
        mode=mode,
        idle_grace_seconds=idle_grace_seconds,
        consumed_active_seconds=0.0,
        last_tick_at=None,
    )
    write_cell(cell, cells_dir)
    logger.info(
        "Opened cell %s: dataset=%s encoder=%s mil_model=%s budget=%ds buffer=%ds mode=%s",
        cell_id[:8], dataset, encoder, mil_model, budget_seconds, safety_buffer_seconds, mode,
    )
    return cell


def get_cell(cell_id: str) -> Cell | None:
    """Return Cell with the given cell_id, or None if not found.

    Returns None gracefully when called from a non-project cwd (the
    ``_cells_dir()`` lookup raises ``click.ClickException`` when no
    ``automil/config.yaml`` is findable upward). Production callers that
    *know* they're inside a project should pre-validate via
    ``_find_automil_dir()`` if they want loud failures on misconfig.
    """
    try:
        cells_dir = _cells_dir()
    except click.ClickException:
        return None
    path = cells_dir / f"{cell_id}.json"
    if not path.exists():
        return None
    try:
        return read_cell(path)
    except (json.JSONDecodeError, OSError, KeyError) as exc:
        logger.warning("Could not read cell %s: %s", cell_id[:8], exc)
        return None


def list_cells(cells_dir: Path | None = None) -> list[Cell]:
    """Return all cells under ``cells_dir`` (or ``automil/cells/``), sorted
    by cell_id.

    Pass ``cells_dir`` explicitly from a callsite that already knows its
    project root (the orchestrator daemon has ``self.automil_dir``) — the
    ``_find_automil_dir()`` cwd-walk fallback only works when invoked from
    inside the consumer project, not from a tmp-path sandbox or another
    cwd. Malformed cell files are skipped with ``logger.warning``. Returns
    [] when invoked from a non-project cwd (cwd-walk failure is silent).
    """
    if cells_dir is None:
        try:
            cells_dir = _cells_dir()
        except click.ClickException:
            return []
    if not cells_dir.exists():
        return []
    cells: list[Cell] = []
    for p in sorted(cells_dir.glob("*.json")):
        try:
            cells.append(read_cell(p))
        except (json.JSONDecodeError, OSError, KeyError, ValueError) as exc:
            logger.warning("Skipping malformed cell file %s: %s", p, exc)
    return cells


def is_refusing_new(cell: Cell) -> bool:
    """True iff cell's status blocks new submits (D-116)."""
    return cell.status in (
        CellStatus.REFUSING_NEW,
        CellStatus.TERMINATING,
        CellStatus.FINALIZED,
    )
