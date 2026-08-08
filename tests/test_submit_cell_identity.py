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
    cfg.setdefault("cap", {})["mode"] = "wall_clock"
    for key, value in sections.items():
        cfg[key] = value
    cfg_path.write_text(yaml.safe_dump(cfg))


def _submit_and_read_cell(tmp_path: Path, cli_runner: CliRunner, node: str = "node_0001") -> dict:
    (tmp_path / "model.py").write_text("print('changed')\n")
    result = cli_runner.invoke(
        main,
        ["submit", "--node", node, "--desc", "cell identity", "--files", "model.py",
         "--mil-model", "test_model"],  # D-12: --mil-model now required
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

    def test_legacy_dataset_encoder_keys_are_rejected(self, cli_runner, tmp_path, monkeypatch):
        """Removed top-level dataset/encoder aliases do not identify a cell."""
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

        (tmp_path / "model.py").write_text("print('changed')\n")
        result = cli_runner.invoke(
            main,
            ["submit", "--node", "node_0001", "--desc", "legacy identity",
             "--files", "model.py", "--mil-model", "test_model"],
            catch_exceptions=False,
        )
        assert result.exit_code != 0
        assert "config.project.name" in result.output
        assert not list((tmp_path / "automil" / "cells").glob("*.json"))

    def test_missing_identity_is_rejected(self, cli_runner, tmp_path, monkeypatch):
        """Incomplete current-schema identity fails before a cell is opened."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])
        # Clear every identity source, including task.name (the last fallback
        # before 'unknown', stamped from the project dir name at init).
        _set_config(tmp_path, project={}, encoders={}, task={})

        (tmp_path / "model.py").write_text("print('changed')\n")
        result = cli_runner.invoke(
            main,
            ["submit", "--node", "node_0001", "--desc", "no identity", "--files", "model.py",
             "--mil-model", "test_model"],  # D-12: required; this test is about dataset/encoder identity
            catch_exceptions=False,
        )
        assert result.exit_code != 0
        assert "config.project.name" in result.output
        assert not list((tmp_path / "automil" / "cells").glob("*.json"))


# ---------------------------------------------------------------------------
# Phase 9 / REC-04 extensions (D-12, D-13, D-14) — RED until Plans 02+05
# ---------------------------------------------------------------------------


class TestMilModelCellIdentity:
    """Tests for --mil-model flag and (dataset, encoder, mil_model) cell keying.

    All tests in this class are RED until Plans 02+05 ship the mil_model changes.
    """

    def test_explicit_mil_model_flag_keys_cell(self, cli_runner, tmp_path, monkeypatch):
        """D-12: --mil-model flag is used for cell keying; cell['mil_model'] == 'clam_sb'.

        RED until Plan 05 adds --mil-model to submit.
        """
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])
        _set_config(
            tmp_path,
            project={"name": "tcga_luad"},
            encoders={"primary": "hoptimus1"},
        )

        (tmp_path / "model.py").write_text("print('changed')\n")
        result = cli_runner.invoke(
            main,
            ["submit", "--node", "node_0001", "--desc", "mil model flag",
             "--files", "model.py", "--mil-model", "clam_sb"],
        )

        # D-12: must succeed and produce a cell with mil_model == 'clam_sb'
        assert result.exit_code == 0, (
            f"D-12 not implemented: submit --mil-model failed with exit_code={result.exit_code}. "
            f"Output: {result.output!r}"
        )
        cell_files = list((tmp_path / "automil" / "cells").glob("*.json"))
        assert cell_files, "No cell file created after submit --mil-model"
        cell = json.loads(cell_files[0].read_text())
        # normalize_mil_model("clam_sb") → "clam sb" (underscores treated as word separators, D-14)
        assert cell.get("mil_model") == "clam sb", (
            f"D-12/D-14: expected cell['mil_model']='clam sb' (normalized form), "
            f"got {cell.get('mil_model')!r}."
        )

    def test_reparent_joins_same_cell(self, cli_runner, tmp_path, monkeypatch):
        """D-13: re-parenting to a different node but same mil_model reuses existing cell.

        submit node_0001 --mil-model clam_sb → cell A
        submit node_0002 --parent node_0001 --mil-model clam_sb → must join cell A (same cell_id)

        RED until Plans 02+05 ship the mil_model key change.
        """
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])
        _set_config(
            tmp_path,
            project={"name": "tcga_luad"},
            encoders={"primary": "hoptimus1"},
        )

        (tmp_path / "model.py").write_text("print('v1')\n")
        result1 = cli_runner.invoke(
            main,
            ["submit", "--node", "node_0001", "--desc", "first",
             "--files", "model.py", "--mil-model", "clam_sb"],
        )

        # D-13: must succeed
        assert result1.exit_code == 0, (
            f"D-13 not implemented: submit --mil-model failed: {result1.output!r}"
        )
        cell_files_after_first = list((tmp_path / "automil" / "cells").glob("*.json"))
        assert len(cell_files_after_first) == 1
        cell_a_id = json.loads(cell_files_after_first[0].read_text())["cell_id"]

        (tmp_path / "model.py").write_text("print('v2')\n")
        result2 = cli_runner.invoke(
            main,
            ["submit", "--node", "node_0002", "--desc", "second",
             "--files", "model.py", "--mil-model", "clam_sb"],
        )
        assert result2.exit_code == 0, f"Second submit failed: {result2.output!r}"

        cell_files_after_second = list((tmp_path / "automil" / "cells").glob("*.json"))
        assert len(cell_files_after_second) == 1, (
            "D-13 not implemented: re-parenting created a new cell instead of joining "
            f"existing cell A ({cell_a_id[:8]})."
        )
        cell_b_id = json.loads(cell_files_after_second[0].read_text())["cell_id"]
        assert cell_a_id == cell_b_id, (
            f"D-13 not implemented: cell_id changed from {cell_a_id[:8]} to {cell_b_id[:8]}. "
            "Re-parenting with same mil_model must join the existing cell."
        )

    def test_missing_mil_model_raises(self, cli_runner, tmp_path, monkeypatch):
        """D-12: submit without --mil-model and no config fallback → exit_code != 0.

        RED until Plan 05 makes --mil-model required-with-inference.
        """
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])
        _set_config(
            tmp_path,
            project={"name": "tcga_luad"},
            encoders={"primary": "hoptimus1"},
            run={},  # explicitly empty — no run.mil_model
        )

        (tmp_path / "model.py").write_text("print('changed')\n")
        result = cli_runner.invoke(
            main,
            ["submit", "--node", "node_0001", "--desc", "no mil model",
             "--files", "model.py"],
            catch_exceptions=False,
        )

        # RED until D-12: currently exits 0 because --mil-model isn't required
        assert result.exit_code != 0, (
            "D-12 not implemented: submit succeeded without --mil-model. "
            "When run.mil_model is absent from config and --mil-model flag is not provided, "
            "submit must fail with a ClickException."
        )
        assert "required" in result.output.lower() or "mil" in result.output.lower(), (
            f"D-12: error message does not mention 'required' or 'mil': {result.output!r}"
        )

    def test_config_mil_model_default_resolves_without_flag(self, cli_runner, tmp_path, monkeypatch):
        """A freshly-init'd project resolves run.mil_model from config, so submit
        works without --mil-model.

        config.yaml.j2 ships run.mil_model="default", so `automil init` scaffolds a
        config that `automil submit` can use directly (no per-submit --mil-model flag).
        """
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])
        _set_config(tmp_path)
        # Deliberately do NOT pass --mil-model or override run: rely on the template default.
        (tmp_path / "model.py").write_text("print('changed')\n")
        result = cli_runner.invoke(
            main,
            ["submit", "--node", "node_0001", "--desc", "config-resolved mil_model",
             "--files", "model.py"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, (
            f"submit without --mil-model should resolve run.mil_model from the init'd "
            f"config template (default), got exit_code={result.exit_code}. Output: {result.output!r}"
        )
        cell_files = list((tmp_path / "automil" / "cells").glob("*.json"))
        assert cell_files, "No cell file created after submit resolving config mil_model"
        cell = json.loads(cell_files[0].read_text())
        assert cell.get("mil_model") == "default", (
            f"expected cell['mil_model']='default' (normalized template value), "
            f"got {cell.get('mil_model')!r}"
        )

    def test_mil_model_normalization_same_cell(self, cli_runner, tmp_path, monkeypatch):
        """D-14: 'CLAM_SB' and ' clam sb ' normalize to the same cell_id.

        RED until Plans 02+05 ship normalize_mil_model + --mil-model in submit.
        """
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])
        _set_config(
            tmp_path,
            project={"name": "tcga_luad"},
            encoders={"primary": "hoptimus1"},
        )

        # Submit with CLAM_SB (uppercase)
        (tmp_path / "model.py").write_text("print('v1')\n")
        result1 = cli_runner.invoke(
            main,
            ["submit", "--node", "node_0001", "--desc", "upper",
             "--files", "model.py", "--mil-model", "CLAM_SB"],
        )

        assert result1.exit_code == 0, (
            f"D-14 not implemented: submit --mil-model CLAM_SB failed: {result1.output!r}"
        )
        cell_files = list((tmp_path / "automil" / "cells").glob("*.json"))
        assert cell_files, "No cell created for first submit"
        cell_id_upper = json.loads(cell_files[0].read_text())["cell_id"]

        # Submit with ' clam sb ' (spaces, lowercase) — should join same cell
        (tmp_path / "model.py").write_text("print('v2')\n")
        result2 = cli_runner.invoke(
            main,
            ["submit", "--node", "node_0002", "--desc", "lower spaced",
             "--files", "model.py", "--mil-model", " clam sb "],
        )
        assert result2.exit_code == 0, f"Second submit failed: {result2.output!r}"

        cell_files_after = list((tmp_path / "automil" / "cells").glob("*.json"))
        assert len(cell_files_after) == 1, (
            "D-14 not implemented: normalization failed — 'CLAM_SB' and ' clam sb ' "
            "created two different cells instead of joining one."
        )
        cell_id_lower = json.loads(cell_files_after[0].read_text())["cell_id"]
        assert cell_id_upper == cell_id_lower, (
            f"D-14: 'CLAM_SB' → cell {cell_id_upper[:8]} but ' clam sb ' → {cell_id_lower[:8]}. "
            "Must normalize to same cell_id."
        )
