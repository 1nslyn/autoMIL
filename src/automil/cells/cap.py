"""Two-tier cap state machine — pure function (CAP-02 / D-113).

No I/O. Caller persists the result via state.write_cell().
Side-effect-free → unit-testable without filesystem.
"""
from __future__ import annotations

from automil.cells.state import Cell, CellStatus, consumed_seconds


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


def next_status(
    cell: Cell,
    now_epoch: float,
    running_count: int,
    *,
    agent_active_seconds: float | None = None,
) -> CellStatus:
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
        agent_active_seconds: Replay result from the bound activity journal.
            Required only for ``agent_active`` cells.
    """
    consumed = consumed_seconds(
        cell, now_epoch, agent_active_seconds=agent_active_seconds,
    )
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
