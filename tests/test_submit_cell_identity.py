"""Cap-cell identity must key off the real config schema.

Regression: `submit` resolved the cell's (dataset, encoder) from non-existent
top-level `dataset`/`encoder` keys, so every cell collapsed to
(unknown, unknown) — defeating per-lineage budget enforcement. Real configs
expose this via `project.name` and `encoders.primary`.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from automil.cli import main


@pytest.fixture
def cli_runner():
    return CliRunner()


def _init_git_repo(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, capture_output=True)
    (path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, capture_output=True, check=True)


def _set_config(tmp_path: Path, **sections) -> None:
    cfg_path = tmp_path / "automil" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    for key, value in sections.items():
        cfg[key] = value
    cfg_path.write_text(yaml.safe_dump(cfg))


def _submit_and_read_cell(tmp_path: Path, cli_runner: CliRunner, node: str = "node_0001") -> dict:
    (tmp_path / "model.py").write_text("print('changed')\n")
    result = cli_runner.invoke(
        main,
        ["submit", "--node", node, "--desc", "cell identity", "--files", "model.py"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    cell_files = list((tmp_path / "automil" / "cells").glob("*.json"))
    assert len(cell_files) == 1, f"expected exactly one cell, got {cell_files}"
    return json.loads(cell_files[0].read_text())


class TestSubmitCellIdentity:
    def test_cell_keys_off_project_and_encoder(self, cli_runner, tmp_path, monkeypatch):
        """A config with project.name + encoders.primary keys a distinct cell."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])
        _set_config(
            tmp_path,
            project={"name": "tcga_luad_egfr"},
            encoders={"primary": "hoptimus1"},
        )

        cell = _submit_and_read_cell(tmp_path, cli_runner)

        assert cell["dataset"] == "tcga_luad_egfr", cell
        assert cell["encoder"] == "hoptimus1", cell

    def test_legacy_dataset_encoder_keys_still_honored(self, cli_runner, tmp_path, monkeypatch):
        """Back-compat: top-level dataset.name / encoder.name still resolve."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])
        # Strip the real-schema keys, supply only the legacy ones.
        _set_config(
            tmp_path,
            project={},
            encoders={},
            dataset={"name": "legacy_ds"},
            encoder={"name": "legacy_enc"},
        )

        cell = _submit_and_read_cell(tmp_path, cli_runner)

        assert cell["dataset"] == "legacy_ds", cell
        assert cell["encoder"] == "legacy_enc", cell

    def test_missing_identity_falls_back_to_unknown_with_warning(self, cli_runner, tmp_path, monkeypatch):
        """No project.name, task.name, or encoders.primary → 'unknown' + a stderr warning."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])
        # Clear every identity source, including task.name (the last fallback
        # before 'unknown', stamped from the project dir name at init).
        _set_config(tmp_path, project={}, encoders={}, task={})

        (tmp_path / "model.py").write_text("print('changed')\n")
        result = cli_runner.invoke(
            main,
            ["submit", "--node", "node_0001", "--desc", "no identity", "--files", "model.py"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert "falling back to" in result.output
        cell = json.loads(next((tmp_path / "automil" / "cells").glob("*.json")).read_text())
        assert cell["dataset"] == "unknown", cell
        assert cell["encoder"] == "unknown", cell
