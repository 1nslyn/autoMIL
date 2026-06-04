"""Integration tests for submit + cell layer (CAP-01, CAP-02 / D-116, D-117, D-134).

Covers:
  1. test_submit_opens_cell_on_first_call     — first submit creates cell file (active)
  2. test_submit_writes_metadata_cell_id      — queue spec has metadata.cell_id
  3. test_submit_rejects_when_cell_refusing_new — refusing-new cell → ClickException
  4. test_submit_cli_budget_override_on_creation — --budget-seconds honored on creation
  5. test_submit_cli_budget_override_ignored_on_existing_cell — override ignored on existing cell
  6. test_submit_validation_fails_on_invalid_buffer — bad flags rejected before cell lookup
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from automil.cli import main
from automil.cells.state import Cell, CellStatus, make_cell_id, write_cell
import time


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _init_git_repo(path: Path) -> None:
    """Initialize a bare git repo with one initial commit."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, capture_output=True)
    (path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, capture_output=True, check=True)


def _setup_project(tmp_path: Path, monkeypatch) -> tuple[CliRunner, Path]:
    """Full automil init + minimal config with dataset/encoder names.

    Returns (runner, adir) where adir = tmp_path/automil.
    """
    _init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["init"])
    assert result.exit_code == 0, f"init failed: {result.output}"

    adir = tmp_path / "automil"
    # Pin dataset/encoder identity so the cell gets deterministic IDs in all
    # tests. Cell identity keys off project.name + encoders.primary (the real
    # config schema submit.py reads), overriding the dir-stamped defaults.
    config_path = adir / "config.yaml"
    cfg = yaml.safe_load(config_path.read_text()) or {}
    cfg["project"] = {**(cfg.get("project") or {}), "name": "test_ds"}
    cfg["encoders"] = {**(cfg.get("encoders") or {}), "primary": "test_enc"}
    config_path.write_text(yaml.safe_dump(cfg))

    return runner, adir


def _make_model_file(tmp_path: Path, content: str = "print('model')\n") -> None:
    """Write model.py so submit has something to snapshot."""
    (tmp_path / "model.py").write_text(content)


def _submit_node(
    runner: CliRunner,
    node: str,
    parent: str | None = None,
    extra_args: list[str] | None = None,
) -> object:
    """Helper to invoke automil submit with a model.py file."""
    args = ["submit", "--node", node, "--desc", f"test {node}", "--files", "model.py"]
    if parent:
        args += ["--parent", parent]
    if extra_args:
        args += extra_args
    return runner.invoke(main, args)


def _cells_dir(adir: Path) -> Path:
    return adir / "cells"


def _cell_id_for(dataset: str = "test_ds", encoder: str = "test_enc",
                 parent_id: str = "root") -> str:
    return make_cell_id(dataset, encoder, parent_id)


def _read_cell_json(adir: Path, cell_id: str) -> dict:
    path = _cells_dir(adir) / f"{cell_id}.json"
    return json.loads(path.read_text())


def _read_queue_spec(adir: Path, node: str) -> dict:
    path = adir / "orchestrator" / "queue" / f"{node}.json"
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSubmitCellLayer:

    def test_submit_opens_cell_on_first_call(self, tmp_path, monkeypatch):
        """First submit for a (dataset, encoder, parent_id) tuple creates cells/<id>.json."""
        runner, adir = _setup_project(tmp_path, monkeypatch)
        _make_model_file(tmp_path)

        result = _submit_node(runner, "node_0001")
        assert result.exit_code == 0, f"submit failed: {result.output}"

        # Cell file must exist.
        cell_id = _cell_id_for()
        cell_path = _cells_dir(adir) / f"{cell_id}.json"
        assert cell_path.exists(), f"Expected cell file at {cell_path}"

        data = json.loads(cell_path.read_text())
        assert data["status"] == "active"
        assert data["dataset"] == "test_ds"
        assert data["encoder"] == "test_enc"
        assert data["parent_id"] == "root"
        assert data["budget_seconds"] == 21600   # framework fallback — no config cap:
        assert data["safety_buffer_seconds"] == 1800

    def test_submit_writes_metadata_cell_id(self, tmp_path, monkeypatch):
        """Queue spec must include metadata.cell_id equal to the cell's cell_id (D-117)."""
        runner, adir = _setup_project(tmp_path, monkeypatch)
        _make_model_file(tmp_path)

        result = _submit_node(runner, "node_0001")
        assert result.exit_code == 0, f"submit failed: {result.output}"

        spec = _read_queue_spec(adir, "node_0001")
        assert "metadata" in spec
        assert "cell_id" in spec["metadata"]

        expected_cell_id = _cell_id_for()
        assert spec["metadata"]["cell_id"] == expected_cell_id

    def test_submit_rejects_when_cell_refusing_new(self, tmp_path, monkeypatch):
        """Submit against a refusing-new cell must raise ClickException with cell_id + budget context."""
        runner, adir = _setup_project(tmp_path, monkeypatch)
        _make_model_file(tmp_path)

        # Manually pre-write a cell in REFUSING_NEW status.
        cell_id = _cell_id_for()
        refusing_cell = Cell(
            cell_id=cell_id,
            dataset="test_ds",
            encoder="test_enc",
            parent_id="root",
            started_at=time.time() - 20000,   # 5.5h ago — well past safety buffer
            budget_seconds=21600,
            safety_buffer_seconds=1800,
            status=CellStatus.REFUSING_NEW,
        )
        cells_dir = _cells_dir(adir)
        cells_dir.mkdir(parents=True, exist_ok=True)
        write_cell(refusing_cell, cells_dir)

        result = _submit_node(runner, "node_0001")
        assert result.exit_code != 0, "Expected submit to fail for refusing-new cell"
        combined = (result.output or "") + (result.exception and str(result.exception) or "")
        # Error message must mention "refusing-new" and "budget exhausted" (Pitfall-9 defence).
        assert "refusing-new" in combined or "budget exhausted" in combined, (
            f"Expected refusal message in output; got: {result.output}"
        )
        # Cell_id[:8] prefix must appear so the operator knows which cell.
        assert cell_id[:8] in combined, (
            f"Expected cell_id[:8]={cell_id[:8]} in output; got: {result.output}"
        )

    def test_submit_cli_budget_override_on_creation(self, tmp_path, monkeypatch):
        """--budget-seconds / --safety-buffer-seconds honored on first submit (D-134)."""
        runner, adir = _setup_project(tmp_path, monkeypatch)
        _make_model_file(tmp_path)

        result = _submit_node(
            runner, "node_0001",
            extra_args=["--budget-seconds", "60", "--safety-buffer-seconds", "10"],
        )
        assert result.exit_code == 0, f"submit failed: {result.output}"

        cell_id = _cell_id_for()
        data = _read_cell_json(adir, cell_id)
        assert data["budget_seconds"] == 60, f"Expected 60, got {data['budget_seconds']}"
        assert data["safety_buffer_seconds"] == 10, f"Expected 10, got {data['safety_buffer_seconds']}"

    def test_submit_cli_budget_override_ignored_on_existing_cell(self, tmp_path, monkeypatch):
        """--budget-seconds on second submit is silently ignored — D-134 first-submit-wins."""
        runner, adir = _setup_project(tmp_path, monkeypatch)
        _make_model_file(tmp_path)

        # First submit opens the cell with default (21600).
        result = _submit_node(runner, "node_0001")
        assert result.exit_code == 0, f"first submit failed: {result.output}"

        cell_id = _cell_id_for()
        data_before = _read_cell_json(adir, cell_id)
        assert data_before["budget_seconds"] == 21600

        # Second submit with --budget-seconds 60 --safety-buffer-seconds 10 must succeed
        # but NOT change the cell (D-134: override only honored on first/creation submit).
        _make_model_file(tmp_path, "print('v2')\n")
        result2 = _submit_node(
            runner, "node_0002",
            extra_args=["--budget-seconds", "60", "--safety-buffer-seconds", "10"],
        )
        assert result2.exit_code == 0, f"second submit failed: {result2.output}"

        data_after = _read_cell_json(adir, cell_id)
        assert data_after["budget_seconds"] == 21600, (
            f"Expected cell budget unchanged at 21600 after override, "
            f"got {data_after['budget_seconds']}"
        )

    def test_submit_validation_fails_on_invalid_buffer(self, tmp_path, monkeypatch):
        """Validation guard: buffer >= budget and budget <= 0 are both rejected."""
        runner, adir = _setup_project(tmp_path, monkeypatch)
        _make_model_file(tmp_path)

        # buffer > budget
        r1 = _submit_node(
            runner, "node_v1",
            extra_args=["--budget-seconds", "100", "--safety-buffer-seconds", "200"],
        )
        assert r1.exit_code != 0, "Expected failure when buffer >= budget"
        assert "0 < buffer < budget" in r1.output, (
            f"Expected validation message; got: {r1.output}"
        )

        # budget <= 0
        r2 = _submit_node(
            runner, "node_v2",
            extra_args=["--budget-seconds", "-1"],
        )
        assert r2.exit_code != 0, "Expected failure when budget <= 0"
        assert "must be > 0" in r2.output, (
            f"Expected 'must be > 0' in output; got: {r2.output}"
        )
