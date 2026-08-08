"""Cell state primitives — frozen dataclass, str Enum, atomic IO (CAP-01, CAP-05 / D-107..D-112)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class CellStatus(str, Enum):
    """Cap state machine lifecycle values (D-110).

    String-valued so ``json.dumps(CellStatus.ACTIVE)`` returns ``'"active"'``
    without a custom encoder.  Four values exhaust the two-tier cap machine.
    """

    ACTIVE = "active"
    REFUSING_NEW = "refusing-new"
    TERMINATING = "terminating"
    FINALIZED = "finalized"


#: Statuses in which a cell must refuse new work (D-116). Declared once here so
#: the registry predicate and the cap predicates cannot drift apart.
BLOCKING_STATUSES: tuple[CellStatus, ...] = (
    CellStatus.REFUSING_NEW,
    CellStatus.TERMINATING,
    CellStatus.FINALIZED,
)


@dataclass(frozen=True)
class Cell:
    """Immutable snapshot of a (dataset, encoder, mil_model) budget cell (D-108, REC-04).

    Frozen so ``Cell`` instances cannot be mutated mid-tick.  Status transitions
    go through ``dataclasses.replace(cell, status=new_status)`` followed by
    ``write_cell(cell, cells_dir)`` — atomic on-disk replacement, never
    in-place mutation.  Hashable + JSON-serialisable via
    ``dataclasses.asdict(cell)``.
    """

    cell_id: str
    """16-char hex derived from sha256(dataset|encoder|mil_model)[:16]."""

    dataset: str
    """Dataset identifier, e.g. ``"ccrcc"`` — from automil/config.yaml."""

    encoder: str
    """Encoder identifier, e.g. ``"uni-v2"`` — from automil/config.yaml."""

    mil_model: str
    """MIL model identifier (normalized: stripped, lowercased, whitespace-collapsed).
    Used as the third dimension of the budget cell key (dataset, encoder, mil_model).
    Graph parent lineage stays separate from budget identity (D-13)."""

    started_at: float
    """Unix epoch seconds (UTC) when the cell was created — absolute wall-clock,
    NOT relative.  Written ONCE at cell creation; never updated."""

    budget_seconds: int
    """Consumer-supplied time budget. The generic framework fallback is 21600
    seconds (6h); publication campaigns and other consumers pin their own
    values."""

    safety_buffer_seconds: int
    """Consumer-supplied pre-termination warning window.  Framework fallback: 1800 (30 min).
    Must be < budget_seconds.  At T - safety_buffer the cell transitions
    ACTIVE → REFUSING_NEW."""

    status: CellStatus
    """Current cap lifecycle state (D-110)."""

    mode: str = "agent_active"
    """Time source for the safety cap.

    ``agent_active`` is derived from Claude Code's native cumulative active-time
    metric, stored in the session-bound activity journal. ``wall_clock`` is the
    continuous interval since ``started_at``. Neither mode stores a mutable time
    accumulator in the Cell; the journal or timestamp remains authoritative.
    """

    # --- Eval-count budget axis (H-2) -----------------------------------------
    # ORTHOGONAL to the time cap, not a third mode: ``mode`` still selects how
    # SECONDS are metered, and the time cap remains the safety wall. The eval
    # count is the primary comparison axis because it is the only one portable
    # across LLM runtimes — a wall-clock comparison is not reproducible by
    # anyone running a different agent harness. All three fields are
    # default-valued so cells written before this feature deserialize unchanged
    # (same contract as the P2.2 block above).

    eval_budget: int | None = None
    """Maximum number of experiment LAUNCHES this cell may dispatch.
    ``None`` = no eval cap: the cell is time-only, exactly as before H-2."""

    consumed_evals: int = 0
    """Evaluations dispatched by this cell — the counter the eval cap is checked
    against. Counted at LAUNCH, so crashed, partial and budget-killed nodes all
    count. "Equal effort" must mean equal ATTEMPTS, not equal successes: if
    crashes were free, an agent that writes buggy code would get unlimited
    retries and the budget would stop being a budget. Advanced by the
    orchestrator at dispatch — never by an agent-reported value."""

    completed_evals: int = 0
    """Evaluations that reached a terminal status of ``completed`` or ``partial``
    — i.e. produced usable results. REPORTED ONLY: this is never the cap, so the
    paper can quote both attempts and usable results per cell."""

    billed_node_ids: list = dataclasses.field(default_factory=list)
    """Node ids already billed to this cell — the A9 exactly-once key.
    ``_launch`` can legitimately re-process one node (daemon crash inside its
    archive→queue-unlink window, or a failed queue unlink retried next tick);
    billing keyed on membership here stays exactly-once across those retries,
    keeping ``consumed_evals`` equal to the archived non-cap-refused census the
    campaign freeze requires. Bounded by ``eval_budget`` in practice; absent
    from pre-A9 cell files (default-valued, same contract as the blocks
    above)."""


def make_cell_id(dataset: str, encoder: str, mil_model: str,
                 task: str | None = None) -> str:
    """Return a 16-char deterministic hex id for a budget cell.

    mil_model must be pre-normalized via normalize_mil_model() before calling (D-14).
    Same input always maps to the same id — re-submits join the existing cell.
    Collision space: ~6.4×10¹⁹ (sha256 prefix, 64-bit).

    M-14 (audit 2026-07-23): ``task`` participates in the identity when supplied.
    Without it, a cohort's classification search and its survival search shared one
    budget, so whichever ran first drained the clock and starved the other. ``None``
    reproduces the legacy 3-tuple id exactly, so existing cells keep their ids.

    >>> make_cell_id("ccrcc", "uni-v2", "clam_sb") == make_cell_id("ccrcc", "uni-v2", "clam_sb")
    True
    >>> make_cell_id("luad", "uni_v2", "clam mb", "kras") != make_cell_id("luad", "uni_v2", "clam mb", "os")
    True
    """
    key = f"{dataset}|{encoder}|{mil_model}"
    if task:
        key = f"{key}|{task}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def normalize_mil_model(raw: str) -> str:
    """Strip, lowercase, normalize separators, collapse internal whitespace (D-14, REC-04).

    Ensures CLAM_SB, clam_sb, and ' clam sb ' all hash to the same cell:
    underscores are treated as word separators (replaced with a single space)
    before whitespace collapsing, so ``CLAM_SB`` and ``clam_sb`` and
    ``' clam sb '`` all normalize to ``'clam sb'``.

    No registry validation — autoMIL is generic and cannot enumerate a
    consumer's models (PROJECT.md).
    """
    # Replace underscores with spaces so CLAM_SB == clam_sb == ' clam sb '
    normalized = raw.strip().lower().replace("_", " ")
    return " ".join(normalized.split())


def consumed_seconds(
    cell: Cell,
    now: float | None = None,
    *,
    agent_active_seconds: float | None = None,
) -> float:
    """Return consumed cap seconds from the cell's authoritative time source.

    Wall-clock cells derive time from their immutable creation timestamp.
    Agent-active cells require the caller to supply the replayed activity-journal
    total; silently substituting zero would turn missing telemetry into unlimited
    budget.
    """
    if cell.mode == "wall_clock":
        n = now if now is not None else time.time()
        return max(0.0, n - cell.started_at)
    if cell.mode != "agent_active":
        raise ValueError(f"unknown cell billing mode: {cell.mode!r}")
    if agent_active_seconds is None:
        raise ValueError("agent_active cells require an activity-journal total")
    return max(0.0, float(agent_active_seconds))


def write_cell(cell: Cell, cells_dir: Path) -> None:
    """Atomically write cell state to ``cells_dir/<cell_id>.json`` (D-112).

    Uses ``tempfile.mkstemp(dir=str(cells_dir))`` to keep the temp file on the
    same filesystem as the destination so ``os.replace`` is an atomic POSIX rename
    (Pitfall 2 defence — cross-filesystem renames are NOT atomic).

    On failure the temp file is cleaned up and the exception re-raised.
    """
    cells_dir.mkdir(parents=True, exist_ok=True)
    path = cells_dir / f"{cell.cell_id}.json"
    payload = json.dumps(dataclasses.asdict(cell), indent=2)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(cells_dir), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as fh:
            fh.write(payload)
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def read_cell(path: Path) -> Cell:
    """Deserialise a cell from ``cells/<cell_id>.json``.

    Re-hydrates ``CellStatus`` from its string value so the returned ``Cell``
    is fully typed — ``cell.status == CellStatus.ACTIVE``, not ``"active"``.

    The persisted schema is exact. Obsolete cell layouts are rejected instead
    of silently rewritten because budget identity and time provenance must not
    change during deserialization.
    """
    data = json.loads(path.read_text())
    if "mode" not in data:
        # A pre-native-metering cell file would otherwise inherit the dataclass
        # default and be silently reinterpreted as agent_active — rewriting its
        # billing provenance. Obsolete layouts are rejected, never coerced.
        raise ValueError(
            f"{path.name}: obsolete cell layout without an explicit 'mode'"
        )
    data["status"] = CellStatus(data["status"])
    return Cell(**data)
