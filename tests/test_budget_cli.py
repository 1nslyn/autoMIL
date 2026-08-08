"""Tests for `automil budget` show/set and the comment-preserving editor (P2.3)."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from automil.cli import main
from automil.cli.budget import _apply_cap_updates
from automil.cells.capconfig import resolve_cap_config
from automil.cells.activity import (
    ActivityObservation,
    ingest_prometheus_metrics,
    record_hook_event,
)
from automil.cells.state import Cell, CellStatus, write_cell


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
        text = "cap:\n  budget: 6h\n  safety_buffer: 30m\n"
        out = _apply_cap_updates(text, {"budget": "2h", "mode": "wall_clock"})
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
        assert "framework fallback" in before  # template comment present

        result = cli_runner.invoke(main, ["budget", "set", "2h"], catch_exceptions=False)
        assert result.exit_code == 0, result.output

        after = cfg_path.read_text()
        assert "framework fallback" in after, "template comments must survive budget set"
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

    def test_budget_show_lists_healthy_and_obsolete_cell_rows_before_failing(
        self, cli_runner, tmp_path, monkeypatch,
    ):
        """A broken accounting journal is a visible row, not a hidden registry abort."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])
        adir = tmp_path / "automil"
        cfg = yaml.safe_load((adir / "config.yaml").read_text()) or {}
        cfg.setdefault("cap", {})["mode"] = "wall_clock"
        (adir / "config.yaml").write_text(yaml.safe_dump(cfg))

        cell = Cell(
            cell_id="healthy0123456789",
            dataset="dataset",
            encoder="encoder",
            mil_model="model",
            started_at=time.time(),
            budget_seconds=21600,
            safety_buffer_seconds=1800,
            status=CellStatus.ACTIVE,
            mode="wall_clock",
        )
        cells_dir = adir / "cells"
        write_cell(cell, cells_dir)
        obsolete = json.loads((cells_dir / f"{cell.cell_id}.json").read_text())
        obsolete["consumed_active_seconds"] = 10.0
        (cells_dir / "old-cell.json").write_text(json.dumps(obsolete))

        result = cli_runner.invoke(main, ["budget", "show"])

        assert result.exit_code != 0
        assert "healthy0" in result.output
        assert "old-cell.json" in result.output
        assert "obsolete" in result.output.lower()
        assert "consumed_active_seconds" in result.output

    def test_budget_show_reports_live_activity_session_mismatch(
        self, cli_runner, tmp_path, monkeypatch,
    ):
        """The inspection command checks live health for open sessions."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])
        adir = tmp_path / "automil"
        cell = Cell(
            cell_id="agentlive1234567",
            dataset="dataset",
            encoder="encoder",
            mil_model="model",
            started_at=time.time(),
            budget_seconds=21600,
            safety_buffer_seconds=1800,
            status=CellStatus.ACTIVE,
            mode="agent_active",
        )
        write_cell(cell, adir / "cells")
        record_hook_event(
            adir,
            cell.cell_id,
            {
                "hook_event_name": "SessionStart",
                "session_id": "expected-session",
                "source": "startup",
            },
            observed_at=1.0,
        )
        ingest_prometheus_metrics(
            adir,
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

        result = cli_runner.invoke(main, ["budget", "show"])

        assert result.exit_code != 0
        assert "agentliv" in result.output
        assert "DEGRADED" in result.output
        assert "does not match" in result.output
