"""Tests for automil.cells.registry — get_or_create_cell, get_cell, list_cells, is_refusing_new.

CAP-01, CAP-05 / D-107, D-116, D-134.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pytest

from automil.cells.state import Cell, CellStatus, make_cell_id, write_cell
from automil.cells.registry import get_or_create_cell, get_cell, list_cells, is_refusing_new


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_automil_dir(monkeypatch, tmp_path):
    """Patch _find_automil_dir so registry reads/writes under tmp_path.

    Creates automil/ dir (the overlay root) and returns it.
    """
    automil_dir = tmp_path / "automil"
    automil_dir.mkdir()
    monkeypatch.setattr("automil.cells.registry._find_automil_dir", lambda: automil_dir, raising=True)
    return automil_dir


# ---------------------------------------------------------------------------
# Task 1 tests (RED → GREEN target)
# ---------------------------------------------------------------------------


def test_get_or_create_creates_new_cell(fake_automil_dir):
    """First call creates a new Cell with expected fields (CAP-01 / D-116)."""
    before = time.time()
    cell = get_or_create_cell("ccrcc", "uni-v2", "node_0042", 21600, 1800)
    after = time.time()

    assert cell.dataset == "ccrcc"
    assert cell.encoder == "uni-v2"
    assert cell.mil_model == "node_0042"
    assert cell.budget_seconds == 21600
    assert cell.safety_buffer_seconds == 1800
    assert cell.status == CellStatus.ACTIVE
    # started_at should be approximately now (within 5 s)
    assert before - 1.0 <= cell.started_at <= after + 1.0

    # File on disk
    cell_id = make_cell_id("ccrcc", "uni-v2", "node_0042")
    json_path = fake_automil_dir / "cells" / f"{cell_id}.json"
    assert json_path.exists(), f"Expected cell JSON at {json_path}"


def test_get_or_create_returns_existing_cell_on_second_call(fake_automil_dir):
    """Second call with same args returns the SAME cell (same cell_id, same started_at)."""
    cell_1 = get_or_create_cell("ccrcc", "uni-v2", "node_0042", 21600, 1800)
    cell_2 = get_or_create_cell("ccrcc", "uni-v2", "node_0042", 21600, 1800)

    assert cell_1.cell_id == cell_2.cell_id
    assert cell_1.started_at == cell_2.started_at, (
        "Second call must return persisted started_at, NOT a fresh time.time()"
    )


def test_get_or_create_ignores_budget_override_on_existing_cell(fake_automil_dir, caplog):
    """D-134: second call with different budget_seconds is ignored; original value preserved."""
    cell_1 = get_or_create_cell("ccrcc", "uni-v2", "node_0042", 21600, 1800)

    # Second call with DIFFERENT budget — should be ignored
    with caplog.at_level(logging.INFO, logger="automil.cells.registry"):
        cell_2 = get_or_create_cell("ccrcc", "uni-v2", "node_0042", 1800, 300)

    # Original values preserved
    assert cell_2.budget_seconds == 21600, (
        "budget_seconds should be the original 21600, not the override 1800"
    )
    assert cell_2.safety_buffer_seconds == 1800, (
        "safety_buffer_seconds should be the original 1800, not the override 300"
    )

    # INFO log warning about ignored override
    assert any("ignoring" in record.message.lower() or "ignored" in record.message.lower()
               for record in caplog.records), (
        "Expected an INFO log message about the ignored budget override"
    )


def test_get_cell_returns_existing_cell_by_id(fake_automil_dir):
    """get_cell(cell_id) returns the same cell that was created."""
    cell = get_or_create_cell("ccrcc", "uni-v2", "node_0042", 21600, 1800)
    loaded = get_cell(cell.cell_id)

    assert loaded is not None
    assert loaded.cell_id == cell.cell_id
    assert loaded.started_at == cell.started_at
    assert loaded.status == cell.status


def test_get_cell_returns_none_for_missing(fake_automil_dir):
    """get_cell with unknown id returns None without raising."""
    result = get_cell("nonexistent_id_that_does_not_exist")
    assert result is None


def test_list_cells_returns_all_sorted(fake_automil_dir):
    """list_cells() returns all 3 created cells sorted by cell_id."""
    cell_a = get_or_create_cell("dataset_alpha", "uni-v2", "node_0001", 21600, 1800)
    cell_b = get_or_create_cell("dataset_beta", "uni-v2", "node_0002", 21600, 1800)
    cell_c = get_or_create_cell("dataset_gamma", "uni-v2", "node_0003", 21600, 1800)

    cells = list_cells()
    assert len(cells) == 3

    # Sorted by cell_id (which sorts .json filenames alphabetically)
    cell_ids = [c.cell_id for c in cells]
    assert cell_ids == sorted(cell_ids), "list_cells() must be sorted by cell_id"

    # All 3 cells present
    created_ids = {cell_a.cell_id, cell_b.cell_id, cell_c.cell_id}
    assert {c.cell_id for c in cells} == created_ids


def test_list_cells_returns_empty_when_no_cells_dir(fake_automil_dir):
    """list_cells() returns [] when automil/cells/ does not exist."""
    # fake_automil_dir exists but has no cells/ subdir yet
    cells_subdir = fake_automil_dir / "cells"
    assert not cells_subdir.exists(), "Precondition: cells/ dir should not exist yet"

    result = list_cells()
    assert result == []


def test_list_cells_skips_malformed_files(fake_automil_dir, caplog):
    """list_cells() returns 1 valid cell when one file is malformed JSON (T-04-14)."""
    # Create a valid cell
    valid_cell = get_or_create_cell("ccrcc", "uni-v2", "node_0042", 21600, 1800)

    # Inject a malformed JSON file directly
    cells_subdir = fake_automil_dir / "cells"
    malformed_path = cells_subdir / "zzzz_malformed_file.json"
    malformed_path.write_text("{invalid json !!!")

    with caplog.at_level(logging.WARNING, logger="automil.cells.registry"):
        cells = list_cells()

    # Only the valid cell is returned
    assert len(cells) == 1
    assert cells[0].cell_id == valid_cell.cell_id

    # Warning logged for malformed file
    assert any("malformed" in record.message.lower() or "skipping" in record.message.lower()
               for record in caplog.records), (
        "Expected a WARNING log about the malformed cell file"
    )


@pytest.mark.parametrize(
    "status, expected_refusing",
    [
        (CellStatus.ACTIVE, False),
        (CellStatus.REFUSING_NEW, True),
        (CellStatus.TERMINATING, True),
        (CellStatus.FINALIZED, True),
    ],
)
def test_is_refusing_new_each_status(status, expected_refusing):
    """is_refusing_new() returns True iff status blocks new submits (D-116)."""
    from tests.cells.conftest import make_cell
    cell = make_cell(status=status)
    assert is_refusing_new(cell) == expected_refusing


def test_started_at_persists_across_reload(fake_automil_dir):
    """CAP-05 restart-safety: started_at read from disk equals the value set at creation."""
    # First call creates the cell
    original = get_or_create_cell("ccrcc", "uni-v2", "node_0042", 21600, 1800)
    original_started_at = original.started_at

    # Simulate daemon restart: read the cell fresh from disk via get_cell
    reloaded = get_cell(original.cell_id)

    assert reloaded is not None
    assert reloaded.started_at == original_started_at, (
        "started_at must be the same after reload — it is persisted ONCE and never updated"
    )


def test_cell_id_deterministic_across_calls(fake_automil_dir):
    """make_cell_id produces the same id across both get_or_create_cell calls."""
    dataset, encoder, mil_model = "ccrcc", "uni-v2", "node_0042"
    expected_id = make_cell_id(dataset, encoder, mil_model)

    cell_1 = get_or_create_cell(dataset, encoder, mil_model, 21600, 1800)
    cell_2 = get_or_create_cell(dataset, encoder, mil_model, 21600, 1800)

    assert cell_1.cell_id == expected_id
    assert cell_2.cell_id == expected_id
