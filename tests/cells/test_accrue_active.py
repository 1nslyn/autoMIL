"""Unit tests for activity-gated budget accrual (P2.2).

accrue_active() advances an agent_active cell's consumed_active_seconds by the
wall-clock since the last tick, but ONLY while the agent acted within
idle_grace_seconds. next_status() then meters against that accumulator.
"""
from __future__ import annotations

from automil.cells.cap import accrue_active, next_status
from automil.cells.state import CellStatus
from tests.cells.conftest import make_cell


def _agent_cell(**kw):
    base = dict(
        mode="agent_active",
        status=CellStatus.ACTIVE,
        started_at=1000.0,
        last_tick_at=1000.0,
        idle_grace_seconds=300,
        consumed_active_seconds=0.0,
        budget_seconds=21600,
        safety_buffer_seconds=1800,
    )
    base.update(kw)
    return make_cell(**base)


class TestAccrueActive:
    def test_recent_action_advances_clock(self):
        cell = _agent_cell()
        out = accrue_active(cell, now_epoch=1005.0, last_action_at=1004.0)
        assert out.consumed_active_seconds == 5.0
        assert out.last_tick_at == 1005.0

    def test_stale_action_freezes_clock_but_advances_tick(self):
        cell = _agent_cell()
        # last action 405s ago > idle_grace 300 → idle → no accrual.
        out = accrue_active(cell, now_epoch=1005.0, last_action_at=600.0)
        assert out.consumed_active_seconds == 0.0
        assert out.last_tick_at == 1005.0  # tick still advances

    def test_no_marker_freezes_clock(self):
        cell = _agent_cell()
        out = accrue_active(cell, now_epoch=1005.0, last_action_at=None)
        assert out.consumed_active_seconds == 0.0
        assert out.last_tick_at == 1005.0

    def test_accumulates_across_ticks(self):
        cell = _agent_cell()
        cell = accrue_active(cell, 1005.0, 1005.0)   # +5
        cell = accrue_active(cell, 1010.0, 1009.0)   # +5
        cell = accrue_active(cell, 1100.0, 600.0)    # idle → +0
        assert cell.consumed_active_seconds == 10.0

    def test_per_tick_delta_capped_at_idle_grace(self):
        # Daemon restart: 4000s gap but agent active → cap accrual at idle_grace.
        cell = _agent_cell(last_tick_at=1000.0, idle_grace_seconds=300)
        out = accrue_active(cell, now_epoch=5000.0, last_action_at=4999.0)
        assert out.consumed_active_seconds == 300.0

    def test_last_tick_none_uses_started_at(self):
        cell = _agent_cell(started_at=1000.0, last_tick_at=None)
        out = accrue_active(cell, now_epoch=1004.0, last_action_at=1004.0)
        assert out.consumed_active_seconds == 4.0

    def test_wall_clock_mode_is_noop(self):
        cell = make_cell(mode="wall_clock", status=CellStatus.ACTIVE)
        out = accrue_active(cell, now_epoch=9999.0, last_action_at=9999.0)
        assert out == cell  # unchanged

    def test_terminating_status_is_noop(self):
        cell = _agent_cell(status=CellStatus.TERMINATING)
        out = accrue_active(cell, now_epoch=1005.0, last_action_at=1004.0)
        assert out == cell

    def test_refusing_new_still_accrues(self):
        cell = _agent_cell(status=CellStatus.REFUSING_NEW)
        out = accrue_active(cell, now_epoch=1005.0, last_action_at=1004.0)
        assert out.consumed_active_seconds == 5.0


class TestNextStatusAgentActive:
    """next_status meters an agent_active cell against consumed_active_seconds,
    NOT wall-clock — so a cell created hours ago stays ACTIVE if the agent
    barely worked."""

    def test_old_cell_low_activity_stays_active(self):
        # Created 10h ago, but only 1h of agent-active time → ACTIVE.
        cell = _agent_cell(
            started_at=1000.0,
            consumed_active_seconds=3600,
            budget_seconds=21600,
            safety_buffer_seconds=1800,
        )
        assert next_status(cell, now_epoch=1000.0 + 36000, running_count=0) == CellStatus.ACTIVE

    def test_active_to_refusing_on_active_budget(self):
        cell = _agent_cell(consumed_active_seconds=95, budget_seconds=100, safety_buffer_seconds=10)
        assert next_status(cell, now_epoch=9_999_999.0, running_count=0) == CellStatus.REFUSING_NEW

    def test_refusing_to_terminating_on_exhaustion(self):
        cell = _agent_cell(
            status=CellStatus.REFUSING_NEW,
            consumed_active_seconds=100, budget_seconds=100, safety_buffer_seconds=10,
        )
        assert next_status(cell, now_epoch=9_999_999.0, running_count=0) == CellStatus.TERMINATING
