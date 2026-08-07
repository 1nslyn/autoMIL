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
    ACTIVITY_JOURNAL_FILENAME,
    ActivityAssessment,
    ActivityError,
    ActivityHealth,
    ActivityObservation,
    ActivityReport,
    assess_activity,
    bind_activity_session,
    ingest_prometheus_metrics,
    read_activity_report,
    read_unbound_activity_report,
    record_hook_event,
)
from automil.cells.cap import (
    evals_exhausted,
    next_status,
    remaining_evals,
)
from automil.cells.capconfig import (
    CapResolved,
    format_duration,
    parse_duration,
    parse_eval_budget,
    resolve_cap_config,
)
from automil.cells.reconcile import aggregate_folds, reconcile_budget_kill
from automil.cells.identity import (
    CellIdentity,
    CellIdentityError,
    resolve_cell_identity,
)
from automil.cells.registry import (
    CellRegistryScan,
    CellSchemaError,
    blocks_new_work,
    get_cell,
    get_or_create_cell,
    is_refusing_new,
    list_cells,
    scan_cells,
)
from automil.cells.state import (
    BLOCKING_STATUSES,
    Cell,
    CellStatus,
    consumed_seconds,
    make_cell_id,
    read_cell,
    write_cell,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ACTIVITY_JOURNAL_FILENAME",
    "ActivityAssessment",
    "ActivityError",
    "ActivityHealth",
    "ActivityObservation",
    "ActivityReport",
    "BLOCKING_STATUSES",
    "CapResolved",
    "Cell",
    "CellRegistryScan",
    "CellSchemaError",
    "CellStatus",
    "aggregate_folds",
    "assess_activity",
    "bind_activity_session",
    "blocks_new_work",
    "consumed_seconds",
    "evals_exhausted",
    "format_duration",
    "get_cell",
    "get_or_create_cell",
    "ingest_prometheus_metrics",
    "is_refusing_new",
    "list_cells",
    "scan_cells",
    "make_cell_id",
    "next_status",
    "parse_duration",
    "parse_eval_budget",
    "read_cell",
    "read_activity_report",
    "read_unbound_activity_report",
    "record_hook_event",
    "reconcile_budget_kill",
    "remaining_evals",
    "resolve_cap_config",
    "write_cell",
    "CellIdentity",
    "CellIdentityError",
    "resolve_cell_identity",
]
