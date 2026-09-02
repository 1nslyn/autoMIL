"""CR-2 (audit 2026-07-23): every graph.json read-modify-write must run under
``locked_update`` so a concurrent daemon completion cannot clobber it via a stale
snapshot. Previously propose / nominate / reconcile did a bare
``ExperimentGraph(...).save()`` with no lock.

These tests spy on ``automil.graph.locked_update`` to prove each write path holds
the lock, and confirm functional correctness is preserved.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

import automil.graph as graph_mod
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


def _spy_lock(monkeypatch):
    """Wrap the real locked_update, counting invocations (still functional)."""
    calls = {"n": 0}
    real = graph_mod.locked_update

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(graph_mod, "locked_update", spy)
    return calls


def _setup(cli_runner, tmp_path, monkeypatch) -> Path:
    _init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    cli_runner.invoke(main, ["init"])
    return tmp_path / "automil"


def test_propose_holds_lock(cli_runner, tmp_path, monkeypatch):
    adir = _setup(cli_runner, tmp_path, monkeypatch)
    calls = _spy_lock(monkeypatch)
    r = cli_runner.invoke(
        main, ["propose", "--parent", "root", "--kind", "hp", "--desc", "x"],
        catch_exceptions=False,
    )
    assert r.exit_code == 0, r.output
    assert calls["n"] == 1  # the whole read-modify-write ran under the lock
    graph = json.loads((adir / "graph.json").read_text())
    assert any(n.get("type") == "proposed" for n in graph["nodes"].values())


def test_reconcile_default_holds_lock(cli_runner, tmp_path, monkeypatch):
    _setup(cli_runner, tmp_path, monkeypatch)
    calls = _spy_lock(monkeypatch)
    r = cli_runner.invoke(main, ["reconcile"], catch_exceptions=False)
    assert r.exit_code == 0, r.output
    assert calls["n"] == 1


def test_recompute_best_locks_only_when_persisting(cli_runner, tmp_path, monkeypatch):
    _setup(cli_runner, tmp_path, monkeypatch)
    calls = _spy_lock(monkeypatch)
    # non-dry-run persists → must lock
    r = cli_runner.invoke(main, ["reconcile", "--recompute-best"], catch_exceptions=False)
    assert r.exit_code == 0, r.output
    assert calls["n"] == 1
    # dry-run is read-only → must NOT acquire the write lock
    r2 = cli_runner.invoke(
        main, ["reconcile", "--recompute-best", "--dry-run"], catch_exceptions=False
    )
    assert r2.exit_code == 0, r2.output
    assert calls["n"] == 1


def test_nominate_holds_lock(cli_runner, tmp_path, monkeypatch):
    adir = _setup(cli_runner, tmp_path, monkeypatch)
    # Build a keep node to nominate.
    g = graph_mod.ExperimentGraph(path=str(adir / "graph.json"))
    nid = g.add_proposed(parent_id="root", description="x", techniques=[], kind="hp")
    node = g.get_node(nid)
    node["type"] = "executed"
    node["status"] = "keep"
    node["primary_value"] = 0.9
    g.save()

    calls = _spy_lock(monkeypatch)
    r = cli_runner.invoke(main, ["nominate", nid], catch_exceptions=False)
    assert r.exit_code == 0, r.output
    assert calls["n"] == 1
    graph = json.loads((adir / "graph.json").read_text())
    assert graph["nodes"][nid]["status"] == "candidate"


def test_locked_update_preserves_preexisting_lock_file_content(tmp_path):
    """The lock sidecar must be opened "a+", never "w" -- "w" truncates on
    open, so a lock file that already carries content would lose it before
    the flock is even acquired.
    """
    graph_path = tmp_path / "automil" / "graph.json"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = graph_path.with_suffix(graph_path.suffix + ".lock")
    lock_path.write_text("pre-existing lock marker\n")

    with graph_mod.locked_update(graph_path) as g:
        g.add_proposed(parent_id=None, description="x", techniques=[], kind="hp")

    assert lock_path.read_text().startswith("pre-existing lock marker")
