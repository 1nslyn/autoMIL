"""CLI integration tests for automil cell status / list commands (CAP-06 / D-125).

Tests construct a fake automil overlay in tmp_path, populate cells/<id>.json via
write_cell(), then invoke commands via click.testing.CliRunner.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from automil.cells.activity import (
    ActivityObservation,
    ingest_prometheus_metrics,
    record_hook_event,
)
from automil.cells.state import Cell, CellStatus, write_cell
from automil.cli.cell import cell_group


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _setup_automil_overlay(tmp_path: Path) -> Path:
    """Create a minimal automil/ overlay with just config.yaml.

    Returns the automil/ directory.
    """
    automil_dir = tmp_path / "automil"
    automil_dir.mkdir(parents=True)
    (automil_dir / "config.yaml").write_text("{}")
    (automil_dir / "cells").mkdir()
    (automil_dir / "orchestrator" / "running").mkdir(parents=True)
    return automil_dir


def _make_cell(
    cell_id: str = "abc1234567890123",
    dataset: str = "ccrcc",
    encoder: str = "uni-v2",
    mil_model: str = "node_0042",
    started_at: float | None = None,
    budget_seconds: int = 21600,
    safety_buffer_seconds: int = 1800,
    status: CellStatus = CellStatus.ACTIVE,
    mode: str = "wall_clock",
) -> Cell:
    return Cell(
        cell_id=cell_id,
        dataset=dataset,
        encoder=encoder,
        mil_model=mil_model,
        started_at=started_at if started_at is not None else time.time(),
        budget_seconds=budget_seconds,
        safety_buffer_seconds=safety_buffer_seconds,
        status=status,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Test 1: cell list — empty cells dir
# ---------------------------------------------------------------------------


def test_cell_list_empty(tmp_path, monkeypatch):
    """cell list shows '(no cells)' when the cells/ dir is empty."""
    automil_dir = _setup_automil_overlay(tmp_path)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cell_group, ["list"])

    assert result.exit_code == 0, result.output
    assert "(no cells)" in result.output


# ---------------------------------------------------------------------------
# Test 2: cell list — two cells with different statuses
# ---------------------------------------------------------------------------


def test_cell_list_with_cells(tmp_path, monkeypatch):
    """cell list shows both cells with status and consumed/budget columns."""
    automil_dir = _setup_automil_overlay(tmp_path)
    cells_dir = automil_dir / "cells"
    monkeypatch.chdir(tmp_path)

    cell_a = _make_cell(
        cell_id="aaaa1234567890ab",
        dataset="ccrcc",
        encoder="uni-v2",
        mil_model="node_0001",
        status=CellStatus.ACTIVE,
    )
    cell_b = _make_cell(
        cell_id="bbbb1234567890cd",
        dataset="clwd",
        encoder="conch",
        mil_model="node_0002",
        status=CellStatus.REFUSING_NEW,
    )
    write_cell(cell_a, cells_dir)
    write_cell(cell_b, cells_dir)

    runner = CliRunner()
    result = runner.invoke(cell_group, ["list"])

    assert result.exit_code == 0, result.output
    assert "aaaa1234" in result.output
    assert "bbbb1234" in result.output
    assert "active" in result.output
    assert "refusing-new" in result.output


# ---------------------------------------------------------------------------
# Test 3: cell status — no arg lists all cells
# ---------------------------------------------------------------------------


def test_cell_status_lists_all_when_no_arg(tmp_path, monkeypatch):
    """cell status with no arg shows all 3 cells with header."""
    automil_dir = _setup_automil_overlay(tmp_path)
    cells_dir = automil_dir / "cells"
    monkeypatch.chdir(tmp_path)

    for i in range(3):
        cell_id = f"cell{i:012d}abcd"
        cell = _make_cell(
            cell_id=cell_id,
            dataset=f"ds{i}",
            encoder="uni-v2",
            mil_model=f"node_{i:04d}",
            status=CellStatus.ACTIVE,
        )
        write_cell(cell, cells_dir)

    runner = CliRunner()
    result = runner.invoke(cell_group, ["status"])

    assert result.exit_code == 0, result.output
    # Header row must be present
    assert "cell_id" in result.output
    assert "dataset" in result.output
    assert "consumed/budget" in result.output
    assert "running" in result.output
    # All 3 cells visible
    for i in range(3):
        assert f"cell{i:012d}"[:8] in result.output


# ---------------------------------------------------------------------------
# Test 4: cell status — specific full 16-char id
# ---------------------------------------------------------------------------


def test_cell_status_specific_id_full(tmp_path, monkeypatch):
    """cell status <full_16_char_id> shows only that cell."""
    automil_dir = _setup_automil_overlay(tmp_path)
    cells_dir = automil_dir / "cells"
    monkeypatch.chdir(tmp_path)

    cell = _make_cell(
        cell_id="zzzz1234567890ef",
        dataset="ccrcc",
        encoder="uni-v2",
        mil_model="node_0099",
        status=CellStatus.ACTIVE,
    )
    write_cell(cell, cells_dir)

    runner = CliRunner()
    result = runner.invoke(cell_group, ["status", "zzzz1234567890ef"])

    assert result.exit_code == 0, result.output
    assert "zzzz1234" in result.output
    assert "ccrcc" in result.output


# ---------------------------------------------------------------------------
# Test 5: cell status — short prefix match
# ---------------------------------------------------------------------------


def test_cell_status_specific_id_short_prefix(tmp_path, monkeypatch):
    """cell status <8_char_prefix> matches and shows the cell."""
    automil_dir = _setup_automil_overlay(tmp_path)
    cells_dir = automil_dir / "cells"
    monkeypatch.chdir(tmp_path)

    cell = _make_cell(
        cell_id="deadbeef12345678",
        dataset="clwd",
        encoder="conch",
        mil_model="node_0010",
        status=CellStatus.ACTIVE,
    )
    write_cell(cell, cells_dir)

    runner = CliRunner()
    result = runner.invoke(cell_group, ["status", "deadbeef"])

    assert result.exit_code == 0, result.output
    assert "deadbeef" in result.output
    assert "clwd" in result.output


# ---------------------------------------------------------------------------
# Test 6: cell status — unknown id exits non-zero
# ---------------------------------------------------------------------------


def test_cell_status_unknown_id_errors(tmp_path, monkeypatch):
    """cell status <unknown_id> exits non-zero with 'No cell found' message."""
    automil_dir = _setup_automil_overlay(tmp_path)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(cell_group, ["status", "nonexistent_id"])

    assert result.exit_code != 0
    assert "No cell found" in result.output


# ---------------------------------------------------------------------------
# Test 7: cell status — ambiguous prefix errors
# ---------------------------------------------------------------------------


def test_cell_status_ambiguous_prefix_errors(tmp_path, monkeypatch):
    """cell status <ambiguous_prefix> exits non-zero with 'Ambiguous prefix' message."""
    automil_dir = _setup_automil_overlay(tmp_path)
    cells_dir = automil_dir / "cells"
    monkeypatch.chdir(tmp_path)

    # Two cells sharing 3-char prefix "abc"
    cell_a = _make_cell(
        cell_id="abc1234567890123",
        dataset="ccrcc",
        encoder="uni-v2",
        mil_model="node_0001",
    )
    cell_b = _make_cell(
        cell_id="abc4567890123456",
        dataset="clwd",
        encoder="conch",
        mil_model="node_0002",
    )
    write_cell(cell_a, cells_dir)
    write_cell(cell_b, cells_dir)

    runner = CliRunner()
    result = runner.invoke(cell_group, ["status", "abc"])

    assert result.exit_code != 0
    assert "Ambiguous prefix" in result.output
    assert "matched 2 cells" in result.output


# ---------------------------------------------------------------------------
# Test 8: cell list --no-header
# ---------------------------------------------------------------------------


def test_cell_list_no_header_pipe_friendly(tmp_path, monkeypatch):
    """cell list --no-header omits the header row; cells are still listed."""
    automil_dir = _setup_automil_overlay(tmp_path)
    cells_dir = automil_dir / "cells"
    monkeypatch.chdir(tmp_path)

    cell_a = _make_cell(
        cell_id="hdrtest0123456789",
        dataset="ccrcc",
        encoder="uni-v2",
        mil_model="node_0001",
        status=CellStatus.ACTIVE,
    )
    cell_b = _make_cell(
        cell_id="hdrtest9876543210",
        dataset="clwd",
        encoder="conch",
        mil_model="node_0002",
        status=CellStatus.REFUSING_NEW,
    )
    write_cell(cell_a, cells_dir)
    write_cell(cell_b, cells_dir)

    runner = CliRunner()
    result = runner.invoke(cell_group, ["list", "--no-header"])

    assert result.exit_code == 0, result.output
    # Header line "cell_id" must NOT appear
    lines = result.output.strip().splitlines()
    assert not any(line.startswith("cell_id") for line in lines), (
        "Header should be suppressed with --no-header"
    )
    # Both cells must still be listed (first 8 chars)
    assert "hdrtest0" in result.output
    assert "hdrtest9" in result.output
    assert "active" in result.output
    assert "refusing-new" in result.output


# ---------------------------------------------------------------------------
# Test 9: cell status — running count from disk
# ---------------------------------------------------------------------------


def test_cell_status_running_count_from_disk(tmp_path, monkeypatch):
    """running column counts specs with matching metadata.cell_id from running/ dir."""
    automil_dir = _setup_automil_overlay(tmp_path)
    cells_dir = automil_dir / "cells"
    running_dir = automil_dir / "orchestrator" / "running"
    monkeypatch.chdir(tmp_path)

    target_cell_id = "abc1234567890123"
    other_cell_id = "xyz9876543210987"

    cell = _make_cell(
        cell_id=target_cell_id,
        dataset="ccrcc",
        encoder="uni-v2",
        mil_model="node_0042",
    )
    write_cell(cell, cells_dir)

    # 2 running specs in target cell
    for i, node_id in enumerate(["node_001", "node_002"]):
        spec = {"id": node_id, "metadata": {"cell_id": target_cell_id}}
        (running_dir / f"{node_id}.json").write_text(json.dumps(spec))

    # 1 running spec in different cell — must NOT count
    other_spec = {"id": "node_003", "metadata": {"cell_id": other_cell_id}}
    (running_dir / "node_003.json").write_text(json.dumps(other_spec))

    runner = CliRunner()
    result = runner.invoke(cell_group, ["status"])

    assert result.exit_code == 0, result.output
    # The running count for target cell should be 2
    # Check that "2" appears in the output (running column)
    assert "2" in result.output


# ---------------------------------------------------------------------------
# Test 10: cell status — consumed time grows with started_at
# ---------------------------------------------------------------------------


def test_cell_status_consumed_grows_with_started_at(tmp_path, monkeypatch):
    """consumed/budget column shows ~01:00 for a cell started 1 hour ago."""
    automil_dir = _setup_automil_overlay(tmp_path)
    cells_dir = automil_dir / "cells"
    monkeypatch.chdir(tmp_path)

    started_at = time.time() - 3600  # 1 hour ago
    cell = _make_cell(
        cell_id="timetest1234567a",
        dataset="ccrcc",
        encoder="uni-v2",
        mil_model="node_0001",
        started_at=started_at,
        budget_seconds=21600,
    )
    write_cell(cell, cells_dir)

    runner = CliRunner()
    result = runner.invoke(cell_group, ["status"])

    assert result.exit_code == 0, result.output
    # consumed should be approximately 01:00:XX — check HH:MM prefix
    assert "01:00:" in result.output, (
        f"Expected '01:00:' in output for 1-hour-old cell.\nGot:\n{result.output}"
    )


@pytest.mark.parametrize("command", ["list", "status"])
def test_cell_inspection_shows_valid_and_obsolete_rows_then_exits_nonzero(
    tmp_path, monkeypatch, command,
):
    """One obsolete journal is visible but cannot hide the healthy cells."""
    automil_dir = _setup_automil_overlay(tmp_path)
    cells_dir = automil_dir / "cells"
    monkeypatch.chdir(tmp_path)

    valid = _make_cell(cell_id="healthy0123456789")
    write_cell(valid, cells_dir)
    obsolete = json.loads((cells_dir / f"{valid.cell_id}.json").read_text())
    obsolete["last_tick_at"] = time.time()
    (cells_dir / "broken-journal.json").write_text(json.dumps(obsolete))

    result = CliRunner().invoke(cell_group, [command])

    assert result.exit_code != 0
    assert "healthy0" in result.output
    assert "broken-journal.json" in result.output
    assert "obsolete" in result.output.lower()
    assert "last_tick_at" in result.output


def test_cell_list_marks_unmetered_activity_row_degraded_without_hiding_healthy_cell(
    tmp_path, monkeypatch,
):
    """Structurally incomplete activity evidence is reported per row."""
    automil_dir = _setup_automil_overlay(tmp_path)
    cells_dir = automil_dir / "cells"
    monkeypatch.chdir(tmp_path)
    write_cell(_make_cell(cell_id="healthy0123456789"), cells_dir)
    agent_cell = _make_cell(
        cell_id="agentbad12345678",
        mode="agent_active",
    )
    write_cell(agent_cell, cells_dir)
    record_hook_event(
        automil_dir,
        agent_cell.cell_id,
        {
            "hook_event_name": "SessionStart",
            "session_id": "session-unmetered",
            "source": "startup",
        },
        observed_at=1.0,
    )

    result = CliRunner().invoke(cell_group, ["list"])

    assert result.exit_code != 0
    assert "healthy0" in result.output
    assert "agentbad" in result.output
    assert "DEGRADED" in result.output
    assert "sample" in result.output.lower()


@pytest.mark.parametrize("command", ["list", "status"])
def test_cell_inspection_marks_wrong_live_session_as_telemetry_degraded(
    tmp_path, monkeypatch, command,
):
    """Historical samples cannot disguise a mismatched live Claude session."""
    automil_dir = _setup_automil_overlay(tmp_path)
    cell = _make_cell(cell_id="agentlive1234567", mode="agent_active")
    write_cell(cell, automil_dir / "cells")
    record_hook_event(
        automil_dir,
        cell.cell_id,
        {
            "hook_event_name": "SessionStart",
            "session_id": "expected-session",
            "source": "startup",
        },
        observed_at=1.0,
    )
    ingest_prometheus_metrics(
        automil_dir,
        'claude_code_active_time_total'
        '{session_id="expected-session",type="cli"} 7\n',
        observed_at=2.0,
    )
    monkeypatch.setattr(
        "automil.activity_metrics.observe_activity_metrics",
        lambda *_args, **_kwargs: ActivityObservation(
            available=True,
            sessions=("foreign-session",),
            observed_at=3.0,
        ),
    )
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cell_group, [command])

    assert result.exit_code != 0
    assert "agentliv" in result.output
    assert "DEGRADED" in result.output
    assert "does not match" in result.output
