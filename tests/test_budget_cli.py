"""Tests for `automil budget` show/set and the comment-preserving editor (P2.3)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from automil.cli import main
from automil.cli.budget import _apply_cap_updates
from automil.cells.capconfig import resolve_cap_config


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


class TestApplyCapUpdates:
    def test_replaces_existing_key_in_place(self):
        text = "cap:\n  budget: 6h               # keep this comment\n  mode: agent_active\n"
        out = _apply_cap_updates(text, {"budget": "2h"})
        assert "budget: 2h" in out
        assert "mode: agent_active" in out  # untouched sibling preserved

    def test_drops_legacy_twin_and_inserts_new(self):
        text = "cap:\n  budget_seconds: 21600\n  safety_buffer_seconds: 1800\n"
        out = _apply_cap_updates(text, {"budget": "2h"}, frozenset({"budget_seconds"}))
        assert "budget: 2h" in out
        assert "budget_seconds" not in out
        assert "safety_buffer_seconds: 1800" in out  # untouched legacy sibling kept

    def test_preserves_surrounding_comments_and_blocks(self):
        text = (
            "# top comment\n"
            "cap:\n"
            "  # inner comment\n"
            "  budget: 6h\n"
            "  mode: agent_active\n"
            "hardware:\n"
            "  gpu_count: 0\n"
        )
        out = _apply_cap_updates(text, {"budget": "30m"})
        assert "# top comment" in out
        assert "# inner comment" in out
        assert "hardware:" in out and "gpu_count: 0" in out
        assert "budget: 30m" in out

    def test_appends_cap_block_when_absent(self):
        text = "project:\n  name: x\n"
        out = _apply_cap_updates(text, {"budget": "6h"})
        assert "cap:" in out
        assert "budget: 6h" in out
        # original content preserved
        assert "name: x" in out

    def test_result_is_parseable_yaml(self):
        text = "cap:\n  budget_seconds: 21600\n  safety_buffer_seconds: 1800\n"
        out = _apply_cap_updates(text, {"budget": "2h", "mode": "wall_clock"},
                                 frozenset({"budget_seconds"}))
        cfg = yaml.safe_load(out)
        assert cfg["cap"]["budget"] == "2h"
        assert cfg["cap"]["mode"] == "wall_clock"


class TestBudgetCLI:
    def test_budget_show_reports_resolved_cap(self, cli_runner, tmp_path, monkeypatch):
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        result = cli_runner.invoke(main, ["budget", "show"], catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert "budget:" in result.output
        assert "6h" in result.output
        assert "mode:" in result.output
        assert "agent_active" in result.output

    def test_budget_set_updates_config_and_preserves_comments(self, cli_runner, tmp_path, monkeypatch):
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        cfg_path = tmp_path / "automil" / "config.yaml"
        before = cfg_path.read_text()
        assert "autoMIL-paper" in before  # template comment present

        result = cli_runner.invoke(main, ["budget", "set", "2h"], catch_exceptions=False)
        assert result.exit_code == 0, result.output

        after = cfg_path.read_text()
        assert "autoMIL-paper" in after, "template comments must survive budget set"
        cap = resolve_cap_config(yaml.safe_load(after))
        assert cap.budget_seconds == 7200

    def test_budget_set_with_mode_and_buffer(self, cli_runner, tmp_path, monkeypatch):
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        result = cli_runner.invoke(
            main, ["budget", "set", "1h", "--safety-buffer", "5m", "--mode", "wall_clock"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        cfg = yaml.safe_load((tmp_path / "automil" / "config.yaml").read_text())
        cap = resolve_cap_config(cfg)
        assert cap.budget_seconds == 3600
        assert cap.safety_buffer_seconds == 300
        assert cap.mode == "wall_clock"

    def test_budget_set_rejects_buffer_ge_budget(self, cli_runner, tmp_path, monkeypatch):
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        result = cli_runner.invoke(main, ["budget", "set", "1h", "--safety-buffer", "2h"])
        assert result.exit_code != 0
        assert "0 < buffer < budget" in result.output

    def test_budget_show_reports_the_eval_axis(self, cli_runner, tmp_path, monkeypatch):
        """H-2: the eval budget is a first-class axis, so `show` must surface it."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        result = cli_runner.invoke(main, ["budget", "show"], catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert "eval_budget:" in result.output
        assert "none (time-only)" in result.output, (
            "the shipped template sets eval_budget: null — show must say so plainly"
        )

    def test_budget_set_writes_the_eval_budget(self, cli_runner, tmp_path, monkeypatch):
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        result = cli_runner.invoke(main, ["budget", "set", "6h", "--eval-budget", "40"],
                                   catch_exceptions=False)
        assert result.exit_code == 0, result.output
        cfg = yaml.safe_load((tmp_path / "automil" / "config.yaml").read_text())
        assert resolve_cap_config(cfg).eval_budget == 40

    def test_budget_set_can_clear_the_eval_budget(self, cli_runner, tmp_path, monkeypatch):
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])
        cli_runner.invoke(main, ["budget", "set", "6h", "--eval-budget", "40"])

        result = cli_runner.invoke(main, ["budget", "set", "6h", "--eval-budget", "none"],
                                   catch_exceptions=False)
        assert result.exit_code == 0, result.output
        cfg = yaml.safe_load((tmp_path / "automil" / "config.yaml").read_text())
        assert resolve_cap_config(cfg).eval_budget is None

    def test_budget_set_rejects_a_non_count_eval_budget(self, cli_runner, tmp_path, monkeypatch):
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        result = cli_runner.invoke(main, ["budget", "set", "6h", "--eval-budget", "6h"])
        assert result.exit_code != 0
        assert "eval_budget" in result.output

    def test_budget_set_rejects_bad_duration(self, cli_runner, tmp_path, monkeypatch):
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        result = cli_runner.invoke(main, ["budget", "set", "soon"])
        assert result.exit_code != 0
        assert "invalid duration" in result.output
