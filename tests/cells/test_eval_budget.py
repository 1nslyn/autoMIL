"""Eval-count budget axis — the primary equal-effort comparison axis (H-2).

Before this axis existed the cap machinery was TIME-ONLY: a cell could be given
6h and nothing else, so "equal effort" could only ever mean equal wall-clock —
which is not portable across LLM runtimes and therefore not reproducible. These
tests pin the second, orthogonal axis:

  - persisted ``Cell`` fields (``eval_budget`` / ``consumed_evals`` /
    ``completed_evals``) with defaults that keep every pre-existing
    ``cells/<id>.json`` readable;
  - ``remaining_evals`` / ``evals_exhausted`` alongside the remaining-seconds
    predicate;
  - the shared state machine tripping REFUSING_NEW on whichever axis exhausts
    first, while only the TIME axis is ever allowed to kill in-flight work;
  - ``cap.eval_budget`` resolution (absent → None → today's time-only behaviour).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from automil.cells.cap import evals_exhausted, next_status, remaining_evals
from automil.cells.capconfig import resolve_cap_config
from automil.cells.migrate import migrate_cells
from automil.cells.registry import blocks_new_work, is_refusing_new
from automil.cells.state import Cell, CellStatus, read_cell, write_cell
from tests.cells.conftest import make_cell

FAKE_NOW = 1_000_000.0
BUDGET = 21600
BUFFER = 1800

# A cell that is nowhere near its time wall — so any transition observed in the
# tests below is attributable to the eval axis alone.
_PLENTY_OF_TIME = dict(started_at=FAKE_NOW - 60, budget_seconds=BUDGET,
                       safety_buffer_seconds=BUFFER)


# ---------------------------------------------------------------------------
# Persisted schema + backward compatibility
# ---------------------------------------------------------------------------


def test_new_cell_defaults_to_no_eval_cap():
    """A Cell built without the eval kwargs is time-only — today's behaviour."""
    cell = make_cell()
    assert cell.eval_budget is None
    assert cell.consumed_evals == 0
    assert cell.completed_evals == 0


def test_read_cell_backward_compatible_with_pre_h2_json(cells_dir: Path):
    """An existing cells/<id>.json (written before H-2) still deserialises.

    This is the exact key set a P2.2-era cell has on disk. If the new fields
    were not default-valued, every already-open cell would raise TypeError on
    the first daemon tick after upgrade.
    """
    legacy = {
        "cell_id": "legacy0000000001",
        "dataset": "ccrcc",
        "encoder": "uni-v2",
        "mil_model": "clam sb",
        "started_at": FAKE_NOW - 3600,
        "budget_seconds": BUDGET,
        "safety_buffer_seconds": BUFFER,
        "status": "active",
        "mode": "agent_active",
        "idle_grace_seconds": 300,
        "consumed_active_seconds": 1234.5,
        "last_tick_at": FAKE_NOW - 5,
    }
    path = cells_dir / "legacy0000000001.json"
    path.write_text(json.dumps(legacy, indent=2))

    cell = read_cell(path)

    assert cell.eval_budget is None, "legacy cells must stay uncapped on the eval axis"
    assert cell.consumed_evals == 0
    assert cell.completed_evals == 0
    # Pre-existing fields survive untouched.
    assert cell.consumed_active_seconds == 1234.5
    assert cell.status is CellStatus.ACTIVE


def test_write_then_read_round_trips_eval_fields(cells_dir: Path):
    """The new fields persist through the atomic writer."""
    cell = make_cell(cell_id="roundtrip0000001", eval_budget=40,
                     consumed_evals=7, completed_evals=5)
    write_cell(cell, cells_dir)

    reloaded = read_cell(cells_dir / "roundtrip0000001.json")
    assert reloaded.eval_budget == 40
    assert reloaded.consumed_evals == 7
    assert reloaded.completed_evals == 5
    assert reloaded == cell


# ---------------------------------------------------------------------------
# remaining_evals / evals_exhausted
# ---------------------------------------------------------------------------


def test_remaining_evals_is_none_without_an_eval_cap():
    assert remaining_evals(make_cell(consumed_evals=99)) is None


def test_remaining_evals_counts_down_from_the_budget():
    assert remaining_evals(make_cell(eval_budget=10, consumed_evals=3)) == 7


@pytest.mark.parametrize(
    ("eval_budget", "consumed", "expected"),
    [
        (None, 0, False),      # no eval cap → never exhausted
        (None, 1000, False),
        (10, 9, False),        # one left
        (10, 10, True),        # boundary — the budget IS spent at equality
        (10, 11, True),        # overshoot (mid-tick double launch) still exhausted
        (0, 0, True),          # degenerate cap: nothing may launch
    ],
)
def test_evals_exhausted_boundary(eval_budget, consumed, expected):
    cell = make_cell(eval_budget=eval_budget, consumed_evals=consumed)
    assert evals_exhausted(cell) is expected


# ---------------------------------------------------------------------------
# next_status — the eval axis moves the SAME state machine
# ---------------------------------------------------------------------------


def test_eval_exhaustion_trips_refusing_new_with_time_remaining():
    """H-2 reproducer: the budget must bind on evaluations, not only on seconds.

    Time-only machinery leaves this cell ACTIVE (it has ~6h left) even though it
    has already spent every evaluation it was allotted.
    """
    cell = make_cell(status=CellStatus.ACTIVE, eval_budget=10, consumed_evals=10,
                     **_PLENTY_OF_TIME)
    assert next_status(cell, now_epoch=FAKE_NOW, running_count=0) == CellStatus.REFUSING_NEW


def test_unspent_eval_budget_leaves_cell_active():
    cell = make_cell(status=CellStatus.ACTIVE, eval_budget=10, consumed_evals=9,
                     **_PLENTY_OF_TIME)
    assert next_status(cell, now_epoch=FAKE_NOW, running_count=0) == CellStatus.ACTIVE


def test_time_axis_still_trips_when_it_is_the_binding_one():
    """Regression: adding the eval axis must not disarm the time safety wall."""
    cell = make_cell(
        status=CellStatus.ACTIVE,
        started_at=FAKE_NOW - (BUDGET - BUFFER),  # remaining == buffer exactly
        budget_seconds=BUDGET,
        safety_buffer_seconds=BUFFER,
        eval_budget=1000,      # eval axis nowhere near exhausted
        consumed_evals=1,
    )
    assert next_status(cell, now_epoch=FAKE_NOW, running_count=0) == CellStatus.REFUSING_NEW


def test_eval_exhaustion_never_kills_in_flight_work():
    """Eval-exhausted + time remaining + running work → stay REFUSING_NEW.

    Killing the evaluation you just paid for would waste the very unit the
    budget is denominated in, and would corrupt attempted-vs-usable accounting.
    Only the TIME wall escalates to TERMINATING (which fires cancels).
    """
    cell = make_cell(status=CellStatus.REFUSING_NEW, eval_budget=10,
                     consumed_evals=10, **_PLENTY_OF_TIME)
    assert next_status(cell, now_epoch=FAKE_NOW, running_count=2) == CellStatus.REFUSING_NEW


def test_eval_exhausted_cell_closes_once_drained():
    """Eval-exhausted + nothing running → TERMINATING (then FINALIZED next tick).

    running_count == 0 means the TERMINATING branch has nothing to cancel, so the
    cell closes cleanly instead of idling in REFUSING_NEW until its time wall.
    """
    cell = make_cell(status=CellStatus.REFUSING_NEW, eval_budget=10,
                     consumed_evals=10, **_PLENTY_OF_TIME)
    assert next_status(cell, now_epoch=FAKE_NOW, running_count=0) == CellStatus.TERMINATING

    terminating = make_cell(status=CellStatus.TERMINATING, eval_budget=10,
                            consumed_evals=10, **_PLENTY_OF_TIME)
    assert next_status(terminating, now_epoch=FAKE_NOW, running_count=0) == CellStatus.FINALIZED


def test_time_axis_still_terminates_a_cell_with_evals_left():
    cell = make_cell(
        status=CellStatus.REFUSING_NEW,
        started_at=FAKE_NOW - BUDGET,  # remaining == 0
        budget_seconds=BUDGET,
        safety_buffer_seconds=BUFFER,
        eval_budget=1000,
        consumed_evals=1,
    )
    assert next_status(cell, now_epoch=FAKE_NOW, running_count=3) == CellStatus.TERMINATING


@pytest.mark.parametrize(
    "status", [CellStatus.ACTIVE, CellStatus.REFUSING_NEW, CellStatus.TERMINATING],
)
def test_no_eval_cap_preserves_time_only_behaviour(status):
    """eval_budget=None reproduces the pre-H-2 machine exactly."""
    cell = make_cell(status=status, consumed_evals=10_000, **_PLENTY_OF_TIME)
    expected = {
        CellStatus.ACTIVE: CellStatus.ACTIVE,
        CellStatus.REFUSING_NEW: CellStatus.REFUSING_NEW,
        CellStatus.TERMINATING: CellStatus.FINALIZED,  # drained
    }[status]
    assert next_status(cell, now_epoch=FAKE_NOW, running_count=0) == expected


# ---------------------------------------------------------------------------
# blocks_new_work — the admission predicate every caller shares
# ---------------------------------------------------------------------------


def test_blocks_new_work_true_when_evals_spent_though_status_still_active():
    """Admission must read the counter, not only the (tick-lagged) status.

    ``consumed_evals`` advances at launch; ``status`` only advances on the next
    daemon tick. Between the two, a status-only check would let a whole batch of
    queued specs launch past the budget.
    """
    cell = make_cell(status=CellStatus.ACTIVE, eval_budget=2, consumed_evals=2,
                     **_PLENTY_OF_TIME)
    assert is_refusing_new(cell) is False, "status alone still reads ACTIVE"
    assert blocks_new_work(cell) is True


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (CellStatus.ACTIVE, False),
        (CellStatus.REFUSING_NEW, True),
        (CellStatus.TERMINATING, True),
        (CellStatus.FINALIZED, True),
    ],
)
def test_blocks_new_work_matches_status_check_without_an_eval_cap(status, expected):
    cell = make_cell(status=status, **_PLENTY_OF_TIME)
    assert blocks_new_work(cell) is expected


# ---------------------------------------------------------------------------
# cap.eval_budget resolution
# ---------------------------------------------------------------------------


def test_eval_budget_absent_resolves_to_none():
    """Omitting the key preserves today's behaviour: time-only."""
    assert resolve_cap_config({"cap": {"budget": "6h"}}).eval_budget is None


def test_eval_budget_null_resolves_to_none():
    """The shipped template writes `eval_budget: null` — must mean 'no eval cap'."""
    assert resolve_cap_config({"cap": {"eval_budget": None}}).eval_budget is None


def test_eval_budget_parsed_from_config():
    assert resolve_cap_config({"cap": {"eval_budget": 40}}).eval_budget == 40
    assert resolve_cap_config({"cap": {"eval_budget": "40"}}).eval_budget == 40


def test_eval_budget_override_wins():
    cfg = {"cap": {"eval_budget": 40}}
    assert resolve_cap_config(cfg, eval_budget_override=12).eval_budget == 12


@pytest.mark.parametrize("bad", [0, -1, "0", "-3"])
def test_eval_budget_rejects_non_positive(bad):
    with pytest.raises(ValueError, match="eval_budget"):
        resolve_cap_config({"cap": {"eval_budget": bad}})


@pytest.mark.parametrize("bad", ["forty", "6h", 4.5, True, [40]])
def test_eval_budget_rejects_non_integer(bad):
    with pytest.raises(ValueError, match="eval_budget"):
        resolve_cap_config({"cap": {"eval_budget": bad}})


# ---------------------------------------------------------------------------
# Merge safety: re-keying cells must not reset spent evaluations
# ---------------------------------------------------------------------------


def test_migrate_merge_sums_eval_counters(cells_dir: Path):
    """Merging two cells sums the eval counters — a merge is not a budget reset."""
    from automil.cells.state import make_cell_id, normalize_mil_model

    target_id = make_cell_id("ds", "enc", normalize_mil_model("clam_sb"))
    write_cell(make_cell(cell_id=target_id, dataset="ds", encoder="enc",
                         mil_model="clam sb", eval_budget=40,
                         consumed_evals=6, completed_evals=4), cells_dir)
    write_cell(make_cell(cell_id="oldkey0000000001", dataset="ds", encoder="enc",
                         mil_model="node_0007", eval_budget=40,
                         consumed_evals=9, completed_evals=7), cells_dir)

    summaries = migrate_cells(cells_dir, "clam_sb")

    assert any(s["action"] == "merge" for s in summaries), summaries
    merged = read_cell(cells_dir / f"{target_id}.json")
    assert merged.consumed_evals == 15, "spent evaluations must survive a re-key"
    assert merged.completed_evals == 11
