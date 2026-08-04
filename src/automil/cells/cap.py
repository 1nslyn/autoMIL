"""Two-tier cap state machine — pure function (CAP-02 / D-113).

No I/O. Caller persists the result via state.write_cell().
Side-effect-free → unit-testable without filesystem.
"""
from __future__ import annotations

from dataclasses import replace

from automil.cells.state import Cell, CellStatus, consumed_seconds


def accrue_active(cell: Cell, now_epoch: float, last_action_at: float | None) -> Cell:
    """Return a cell with its ``agent_active`` budget advanced for this tick.

    Pure function — no I/O. The caller (daemon ``_tick_cells``) persists the
    result. A no-op for ``wall_clock`` cells and for non-billable statuses
    (TERMINATING/FINALIZED).

    The clock advances by the wall-clock since the last tick ONLY when the agent
    acted within ``idle_grace_seconds`` (``last_action_at`` fresh). The per-tick
    delta is capped at ``idle_grace_seconds`` so a daemon restart gap cannot dump
    a large accrual in one tick. ``last_tick_at`` is always advanced so a
    following active tick measures only ~one poll interval.

    Args:
        cell: current cell state (immutable).
        now_epoch: current wall-clock (caller passes time.time()).
        last_action_at: epoch of the agent's last action, or None if unknown.
    """
    if cell.mode != "agent_active":
        return cell
    if cell.status not in (CellStatus.ACTIVE, CellStatus.REFUSING_NEW):
        return cell

    last_tick = cell.last_tick_at if cell.last_tick_at is not None else cell.started_at
    elapsed = min(max(0.0, now_epoch - last_tick), float(cell.idle_grace_seconds))
    agent_active = (
        last_action_at is not None
        and (now_epoch - last_action_at) < cell.idle_grace_seconds
    )
    new_consumed = cell.consumed_active_seconds + (elapsed if agent_active else 0.0)
    return replace(cell, consumed_active_seconds=new_consumed, last_tick_at=now_epoch)


def remaining_evals(cell: Cell) -> int | None:
    """Return the evaluations this cell may still dispatch, or ``None`` if uncapped.

    The eval-count axis (H-2) is ORTHOGONAL to the time cap — not a third
    ``mode``. It is the primary equal-effort comparison axis because evaluation
    count is portable across LLM runtimes, whereas agent-worktime is not.
    ``None`` (no ``eval_budget``) reproduces the pre-H-2 time-only behaviour.

    Counting policy: ``consumed_evals`` is incremented at LAUNCH, so crashed,
    partial and budget-killed nodes all count. Equal effort must mean equal
    ATTEMPTS, not equal successes — if crashes were free, an agent that writes
    buggy code would get unlimited retries and the budget would stop being a
    budget. ``completed_evals`` tracks usable results as a reported secondary
    and is never consulted here.
    """
    if cell.eval_budget is None:
        return None
    return cell.eval_budget - cell.consumed_evals


def evals_exhausted(cell: Cell) -> bool:
    """True iff the eval axis is spent. Always False when there is no eval cap.

    Monotone: ``consumed_evals`` only ever increases, so a cell that reports
    True here can never return to admitting work.
    """
    left = remaining_evals(cell)
    return left is not None and left <= 0


def next_status(cell: Cell, now_epoch: float, running_count: int) -> CellStatus:
    """Return the next CellStatus given current time and running experiment count.

    Pure function — no I/O, no global state. Idempotent: FINALIZED always
    returns FINALIZED.

    Two orthogonal axes drive ONE state machine (H-2):

      - TIME (``budget_seconds``) — the safety wall. It alone escalates to
        TERMINATING while work is in flight, because that transition fires
        cancels.
      - EVALS (``eval_budget``) — the primary comparison axis. Exhausting it
        refuses new work but NEVER kills a running experiment: killing the
        evaluation you already paid for would waste the very unit the budget is
        denominated in. An eval-exhausted cell closes (TERMINATING → FINALIZED)
        only once its in-flight work has drained on its own.

    Whichever axis exhausts first trips REFUSING_NEW.

    Args:
        cell: Current cell state (immutable).
        now_epoch: Current wall-clock (caller passes time.time()); explicit
            injection makes the function testable without monkeypatch.
        running_count: Count of in-cell experiments NOT in terminal state.
    """
    consumed = consumed_seconds(cell, now_epoch)
    remaining = cell.budget_seconds - consumed
    evals_spent = evals_exhausted(cell)

    if cell.status == CellStatus.ACTIVE:
        if remaining <= cell.safety_buffer_seconds or evals_spent:
            return CellStatus.REFUSING_NEW
        return CellStatus.ACTIVE

    if cell.status == CellStatus.REFUSING_NEW:
        if remaining <= 0:
            return CellStatus.TERMINATING
        if evals_spent and running_count == 0:
            # Eval axis: nothing left to launch and nothing left to wait for, so
            # the cell can close. running_count == 0 means the TERMINATING
            # branch has no cancels to fire — this never kills in-flight work.
            return CellStatus.TERMINATING
        return CellStatus.REFUSING_NEW

    if cell.status == CellStatus.TERMINATING:
        if running_count == 0:
            return CellStatus.FINALIZED
        return CellStatus.TERMINATING

    return cell.status  # FINALIZED is terminal
