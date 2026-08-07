"""Cell CRUD + lazy registry (CAP-01, CAP-05 / D-107, D-116, D-134).

Cells are persisted to automil/cells/<cell_id>.json. The registry is
"singleton-ish" — module-level functions reading/writing the on-disk
state. No in-memory cache (would conflict with daemon-restart safety).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import click

from automil.cells.cap import evals_exhausted
from automil.cells.state import (
    BLOCKING_STATUSES,
    Cell,
    CellStatus,
    make_cell_id,
    read_cell,
    write_cell,
)
from automil.cli._helpers import _find_automil_dir

logger = logging.getLogger(__name__)


class CellSchemaError(ValueError):
    """One persisted cell file cannot be interpreted as the current schema.

    The filename is retained so callers that inspect an entire registry can
    report the broken row without hiding healthy cells.  Cell journals are
    accounting evidence: obsolete layouts are rejected explicitly and are
    never migrated as a side effect of reading them.
    """

    def __init__(self, path: Path, detail: str, *, obsolete: bool = False) -> None:
        self.path = path
        self.detail = detail
        self.obsolete = obsolete
        kind = "obsolete cell schema" if obsolete else "invalid cell schema"
        self.message = f"{kind}: {detail}"
        super().__init__(f"{path}: {self.message}")


@dataclass(frozen=True)
class CellRegistryScan:
    """All readable cells plus every per-file schema error in one scan."""

    cells: tuple[Cell, ...]
    errors: tuple[CellSchemaError, ...]


def _cells_dir() -> Path:
    """Locate automil/cells/ relative to the automil/ overlay dir."""
    return _find_automil_dir() / "cells"


def get_or_create_cell(
    dataset: str,
    encoder: str,
    mil_model: str,
    budget_seconds: int,
    safety_buffer_seconds: int,
    mode: str = "agent_active",
    task: str | None = None,
    eval_budget: int | None = None,
    cells_dir: Path | None = None,
) -> Cell:
    """Return existing cell or create a new one (lazy + idempotent, D-116, REC-04).

    D-134: all cap parameters (budget_seconds, safety_buffer_seconds, mode)
    apply ONLY when this call CREATES the cell. If the
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
        mode: "agent_active" or "wall_clock"; honored only on creation.
        task: M-14 — participates in cell identity so a cohort's classification
            and survival searches do not share (and starve) one budget. None
            reproduces the legacy 3-tuple id.
        eval_budget: H-2 — evaluation-count cap (the primary equal-effort axis);
            honored only on creation, same as every other cap parameter. None
            leaves the cell time-only.
        cells_dir: explicit registry directory for a controller that already
            owns a project path; defaults to the current project's registry.
    """
    cells_dir = cells_dir if cells_dir is not None else _cells_dir()
    cell_id = make_cell_id(dataset, encoder, mil_model, task)
    path = cells_dir / f"{cell_id}.json"
    if path.exists():
        cell = _read_cell_checked(path)
        if (cell.budget_seconds != budget_seconds
                or cell.safety_buffer_seconds != safety_buffer_seconds
                or cell.mode != mode
                or cell.eval_budget != eval_budget):
            logger.info(
                "Cell %s already open (budget=%ds buffer=%ds mode=%s "
                "eval_budget=%s); ignoring override (budget=%ds buffer=%ds "
                "mode=%s eval_budget=%s) per D-134.",
                cell_id[:8], cell.budget_seconds, cell.safety_buffer_seconds,
                cell.mode, cell.eval_budget,
                budget_seconds, safety_buffer_seconds, mode,
                eval_budget,
            )
        return cell

    # First lifecycle owner for this identity opens the cell.
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
        eval_budget=eval_budget,
        consumed_evals=0,
        completed_evals=0,
    )
    write_cell(cell, cells_dir)
    logger.info(
        "Opened cell %s: dataset=%s encoder=%s mil_model=%s budget=%ds buffer=%ds "
        "mode=%s eval_budget=%s",
        cell_id[:8], dataset, encoder, mil_model, budget_seconds, safety_buffer_seconds,
        mode, eval_budget,
    )
    return cell


def get_cell(cell_id: str, cells_dir: Path | None = None) -> Cell | None:
    """Return Cell with the given cell_id, or None if not found.

    Returns None gracefully when called from a non-project cwd (the
    ``_cells_dir()`` lookup raises ``click.ClickException`` when no
    ``automil/config.yaml`` is findable upward). Production callers that
    *know* they're inside a project should pre-validate via
    ``_find_automil_dir()`` if they want loud failures on misconfig. Existing
    but obsolete/invalid files raise ``CellSchemaError`` so accounting evidence
    cannot be mistaken for an absent pristine cell.
    """
    if cells_dir is None:
        try:
            cells_dir = _cells_dir()
        except click.ClickException:
            return None
    path = cells_dir / f"{cell_id}.json"
    if not path.exists():
        return None
    return _read_cell_checked(path)


def _read_cell_checked(path: Path) -> Cell:
    """Read one cell and normalize low-level decode errors at the registry seam."""
    try:
        return read_cell(path)
    except TypeError as exc:
        # Exact dataclass construction intentionally rejects fields from the
        # pre-PR mutable-clock schema.  Surface that rejection as evidence, not
        # as a bare TypeError that can abort the daemon tick.
        raise CellSchemaError(path, str(exc), obsolete=True) from exc
    except (json.JSONDecodeError, OSError, KeyError, ValueError) as exc:
        raise CellSchemaError(path, str(exc)) from exc


def scan_cells(cells_dir: Path | None = None) -> CellRegistryScan:
    """Return every valid cell and every invalid journal, sorted by filename.

    This is the inspection-oriented interface.  Scheduling callers that only
    consume valid cells can retain ``list_cells``; operator commands use this
    richer result so one broken file cannot hide the rest of the registry.
    """
    if cells_dir is None:
        try:
            cells_dir = _cells_dir()
        except click.ClickException:
            return CellRegistryScan((), ())
    if not cells_dir.exists():
        return CellRegistryScan((), ())

    cells: list[Cell] = []
    errors: list[CellSchemaError] = []
    for path in sorted(cells_dir.glob("*.json")):
        try:
            cells.append(_read_cell_checked(path))
        except CellSchemaError as exc:
            errors.append(exc)
    return CellRegistryScan(tuple(cells), tuple(errors))


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
    scan = scan_cells(cells_dir)
    for exc in scan.errors:
        logger.warning("Skipping malformed cell file %s: %s", exc.path, exc.message)
    return list(scan.cells)


def is_refusing_new(cell: Cell) -> bool:
    """True iff cell's *status* blocks new submits (D-116).

    Status-only, and therefore as fresh as the last daemon tick. Callers
    deciding whether to admit new work should prefer ``blocks_new_work``.
    """
    return cell.status in BLOCKING_STATUSES


def blocks_new_work(cell: Cell) -> bool:
    """True iff this cell must refuse a new experiment — either cap axis (H-2).

    The single admission predicate shared by ``automil submit``, the
    orchestrator launch path (CAP-1) and the gate's held-out evaluations.

    Consults the eval COUNTER as well as the status because the two advance at
    different moments: ``consumed_evals`` increments at launch, while ``status``
    only advances on the next daemon tick. A status-only check would let a whole
    batch of already-queued specs launch past an exhausted eval budget in the
    window between the two.
    """
    return is_refusing_new(cell) or evals_exhausted(cell)
