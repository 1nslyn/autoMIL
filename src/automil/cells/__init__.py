"""Cell budget-cap subpackage (CAP-01..06 / D-107..D-134).

Public surface (populated incrementally across Phase 4 plans):
    04-01: Cell, CellStatus, consumed_seconds, write_cell, make_cell_id, read_cell
    04-02: aggregate_folds (early stub, replaced by 04-04)
    04-03: cap state machine — next_status (Wave 2)
    04-04: aggregate_folds final, reconcile_budget_kill (Wave 3)
    04-05: registry — get_or_create_cell, get_cell, list_cells, is_refusing_new (Wave 4)
"""
from __future__ import annotations

import logging

from automil.cells.activity import (
    ACTIVITY_FILENAME,
    read_last_action_at,
    touch_last_action,
)
from automil.cells.cap import accrue_active, next_status
from automil.cells.capconfig import (
    CapResolved,
    format_duration,
    parse_duration,
    resolve_cap_config,
)
from automil.cells.reconcile import aggregate_folds, reconcile_budget_kill
from automil.cells.registry import (
    get_cell,
    get_or_create_cell,
    is_refusing_new,
    list_cells,
)
from automil.cells.state import (
    Cell,
    CellStatus,
    consumed_seconds,
    make_cell_id,
    read_cell,
    write_cell,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ACTIVITY_FILENAME",
    "CapResolved",
    "Cell",
    "CellStatus",
    "accrue_active",
    "aggregate_folds",
    "consumed_seconds",
    "format_duration",
    "get_cell",
    "get_or_create_cell",
    "is_refusing_new",
    "list_cells",
    "make_cell_id",
    "next_status",
    "parse_duration",
    "read_cell",
    "read_last_action_at",
    "reconcile_budget_kill",
    "resolve_cap_config",
    "touch_last_action",
    "write_cell",
]
