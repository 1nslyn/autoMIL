"""Tests for `propose --kind` and the `automil portfolio` gate (P1.2)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
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


def _proposed_kinds(tmp_path: Path) -> list[str]:
    graph = json.loads((tmp_path / "automil" / "graph.json").read_text())
    return sorted(
        n["kind"] for n in graph["nodes"].values() if n.get("type") == "proposed"
    )


class TestProposeKind:
    def test_kind_persisted_in_graph(self, cli_runner, tmp_path, monkeypatch):
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        r = cli_runner.invoke(
            main, ["propose", "--parent", "root", "--kind", "architecture",
                   "--desc", "gated attention head"], catch_exceptions=False)
        assert r.exit_code == 0, r.output
        assert "[architecture]" in r.output
        assert _proposed_kinds(tmp_path) == ["architecture"]

    def test_missing_kind_is_unspecified(self, cli_runner, tmp_path, monkeypatch):
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        r = cli_runner.invoke(
            main, ["propose", "--parent", "root", "--desc", "untyped idea"],
            catch_exceptions=False)
        assert r.exit_code == 0, r.output
        assert "[unspecified]" in r.output
        assert _proposed_kinds(tmp_path) == ["unspecified"]

    def test_invalid_kind_rejected(self, cli_runner, tmp_path, monkeypatch):
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        r = cli_runner.invoke(
            main, ["propose", "--parent", "root", "--kind", "bogus", "--desc", "x"])
        assert r.exit_code != 0


class TestPortfolioGate:
    def _propose(self, cli_runner, kind: str, desc: str):
        return cli_runner.invoke(
            main, ["propose", "--parent", "root", "--kind", kind, "--desc", desc],
            catch_exceptions=False)

    def test_at_threshold_passes(self, cli_runner, tmp_path, monkeypatch):
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])
        self._propose(cli_runner, "architecture", "new pooling")
        self._propose(cli_runner, "hp", "lr=2e-4")

        r = cli_runner.invoke(main, ["portfolio"], catch_exceptions=False)
        assert r.exit_code == 0, r.output  # 1/2 = 50% ≥ 50%
        assert "structural: 1/2" in r.output

    def test_below_threshold_exits_nonzero(self, cli_runner, tmp_path, monkeypatch):
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])
        self._propose(cli_runner, "architecture", "new pooling")
        self._propose(cli_runner, "hp", "lr=2e-4")
        self._propose(cli_runner, "hp", "wd=1e-3")

        r = cli_runner.invoke(main, ["portfolio"])
        assert r.exit_code == 1  # 1/3 = 33% < 50%
        assert "BELOW TARGET" in r.output

    def test_ensemble_counts_as_structural(self, cli_runner, tmp_path, monkeypatch):
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])
        self._propose(cli_runner, "ensemble", "bag of 3 heads")
        self._propose(cli_runner, "regularization", "rdrop")

        r = cli_runner.invoke(main, ["portfolio"], catch_exceptions=False)
        assert r.exit_code == 0, r.output  # ensemble is structural → 1/2

    def test_threshold_override(self, cli_runner, tmp_path, monkeypatch):
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])
        self._propose(cli_runner, "architecture", "a")
        self._propose(cli_runner, "hp", "b")
        self._propose(cli_runner, "hp", "c")
        # 1/3 structural; with --threshold 0.3 it passes.
        r = cli_runner.invoke(main, ["portfolio", "--threshold", "0.3"],
                              catch_exceptions=False)
        assert r.exit_code == 0, r.output
