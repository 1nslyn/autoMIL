"""Shared live-health inspection for agent-active CLI rows."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from automil.cells.activity import (
    ActivityError,
    assess_activity,
    read_activity_report,
)
from automil.cells.state import Cell


@dataclass(frozen=True)
class ActivityInspection:
    """Authentic seconds plus any operator-visible accounting problem."""

    active_seconds: float
    error: str | None = None


def inspect_agent_activity(
    automil_dir: Path,
    cells: Iterable[Cell],
) -> dict[str, ActivityInspection]:
    """Assess all agent-active rows against one shared live observation."""

    reports = {}
    inspections: dict[str, ActivityInspection] = {}
    for cell in cells:
        if cell.mode != "agent_active":
            continue
        try:
            reports[cell.cell_id] = read_activity_report(
                automil_dir, cell.cell_id,
            )
        except ActivityError as exc:
            inspections[cell.cell_id] = ActivityInspection(
                active_seconds=0.0,
                error=f"activity journal is invalid: {exc}",
            )

    observation = None
    if any(report.open_sessions for report in reports.values()):
        from automil.activity_metrics import observe_activity_metrics

        observation = observe_activity_metrics(automil_dir)

    for cell_id, report in reports.items():
        assessment = assess_activity(report, observation)
        inspections[cell_id] = ActivityInspection(
            active_seconds=assessment.active_seconds,
            error=(
                None
                if assessment.admissible
                else f"activity telemetry is degraded: {assessment.reason}"
            ),
        )
    return inspections


__all__ = ["ActivityInspection", "inspect_agent_activity"]
