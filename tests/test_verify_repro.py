"""Coverage for `automil verify-repro` (CLI-09 / REG-09 / D-39 / D-50)."""
from __future__ import annotations

import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner


def _init_git_repo(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, capture_output=True)


def _setup_synthetic_project(tmp_path: Path) -> tuple[Path, Path]:
    """Copy the synthetic_consumer fixture into tmp_path + git init + commit.
    Returns (project_root, automil_dir)."""
    fixture_src = Path(__file__).parent / "fixtures" / "synthetic_consumer"
    for item in fixture_src.iterdir():
        dst = tmp_path / item.name
        if item.is_dir():
            shutil.copytree(item, dst)
        elif item.is_file() and item.name != "__init__.py":
            shutil.copy2(item, dst)

    _init_git_repo(tmp_path)
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "synthetic consumer initial"],
        cwd=tmp_path,
        capture_output=True,
        check=True,
    )
    return tmp_path, tmp_path / "automil"


def _write_graph(adir: Path, nodes: dict, base_commit: str = ""):
    graph = {
        "schema_version": 1,
        "meta": {
            "best_node_id": None,
            "best_primary_value": 0.0,
            "total_executed": 0,
            "total_proposed": 0,
            "next_id": 1,
            "baseline_primary_value": 0.0,
            "scoring": {"exploration_weight": 0.005, "novelty_weight": 0.003},
        },
        "nodes": nodes,
        "technique_stats": {},
    }
    (adir / "graph.json").write_text(json.dumps(graph, indent=2))


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def _isolated_registry():
    from automil.registry._state import _clear_registry

    _clear_registry()
    yield
    _clear_registry()


def test_happy_path_pass(tmp_path, cli_runner, monkeypatch):
    proj, adir = _setup_synthetic_project(tmp_path)
    monkeypatch.chdir(proj)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=proj,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    _write_graph(
        adir,
        {
            "node_0001": {
                "id": "node_0001",
                "type": "executed",
                "status": "keep",
                "primary_value": 0.5,
                "base_commit": head,
                "created_at": "2026-05-02T10:00:00Z",
            }
        },
    )

    from automil.cli import main

    result = cli_runner.invoke(main, ["verify-repro", "node_0001"])
    assert result.exit_code == 0, result.output
    manifest = yaml.safe_load((adir / "repro_manifest.yaml").read_text())
    assert manifest["status"] == "pass"
    assert manifest["actual_primary_value"] == pytest.approx(0.5)


def test_manifest_schema(tmp_path, cli_runner, monkeypatch):
    proj, adir = _setup_synthetic_project(tmp_path)
    monkeypatch.chdir(proj)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=proj,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    _write_graph(
        adir,
        {
            "node_0001": {
                "id": "node_0001",
                "type": "executed",
                "status": "keep",
                "primary_value": 0.5,
                "base_commit": head,
                "created_at": "2026-05-02T10:00:00Z",
            }
        },
    )
    from automil.cli import main

    cli_runner.invoke(main, ["verify-repro", "node_0001"])
    m = yaml.safe_load((adir / "repro_manifest.yaml").read_text())
    for key in (
        "node_id",
        "expected_primary_value",
        "actual_primary_value",
        "tolerance",
        "status",
        "git_sha",
        "runtime_seconds",
        "generated_at",
    ):
        assert key in m, f"missing key {key}"


def test_tolerance_fail(tmp_path, cli_runner, monkeypatch):
    proj, adir = _setup_synthetic_project(tmp_path)
    monkeypatch.chdir(proj)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=proj,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    _write_graph(
        adir,
        {
            "node_0001": {
                "id": "node_0001",
                "type": "executed",
                "status": "keep",
                "primary_value": 0.7,  # actual will be 0.5
                "base_commit": head,
                "created_at": "2026-05-02T10:00:00Z",
            }
        },
    )
    from automil.cli import main

    result = cli_runner.invoke(main, ["verify-repro", "node_0001"])
    assert result.exit_code != 0
    m = yaml.safe_load((adir / "repro_manifest.yaml").read_text())
    assert m["status"] == "fail"


def test_tolerance_override_pass(tmp_path, cli_runner, monkeypatch):
    proj, adir = _setup_synthetic_project(tmp_path)
    monkeypatch.chdir(proj)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=proj,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    _write_graph(
        adir,
        {
            "node_0001": {
                "id": "node_0001",
                "type": "executed",
                "status": "keep",
                "primary_value": 0.7,
                "base_commit": head,
                "created_at": "2026-05-02T10:00:00Z",
            }
        },
    )
    from automil.cli import main

    result = cli_runner.invoke(
        main,
        ["verify-repro", "node_0001", "--tolerance", "0.5"],
    )
    assert result.exit_code == 0
    m = yaml.safe_load((adir / "repro_manifest.yaml").read_text())
    assert m["status"] == "pass"


def test_atomic_write_no_tmp(tmp_path, cli_runner, monkeypatch):
    proj, adir = _setup_synthetic_project(tmp_path)
    monkeypatch.chdir(proj)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=proj,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    _write_graph(
        adir,
        {
            "node_0001": {
                "id": "node_0001",
                "type": "executed",
                "status": "keep",
                "primary_value": 0.5,
                "base_commit": head,
                "created_at": "2026-05-02T10:00:00Z",
            }
        },
    )
    from automil.cli import main

    cli_runner.invoke(main, ["verify-repro", "node_0001"])
    leftover = list(adir.glob("repro_manifest*.tmp"))
    assert leftover == []


def test_missing_node_lists_available(tmp_path, cli_runner, monkeypatch):
    proj, adir = _setup_synthetic_project(tmp_path)
    monkeypatch.chdir(proj)
    _write_graph(
        adir,
        {"node_0001": {"id": "node_0001", "type": "executed", "status": "keep"}},
    )
    from automil.cli import main

    result = cli_runner.invoke(main, ["verify-repro", "node_9999"])
    assert result.exit_code != 0
    assert "available" in result.output.lower()


def test_missing_config_hard_fail(tmp_path, cli_runner, monkeypatch):
    # Setup git but not the synthetic_consumer config.
    _init_git_repo(tmp_path)
    (tmp_path / "README.md").write_text("# x\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "x"], cwd=tmp_path, capture_output=True, check=True
    )
    monkeypatch.chdir(tmp_path)
    from automil.cli import main

    result = cli_runner.invoke(main, ["verify-repro", "node_0001"])
    assert result.exit_code != 0


def test_help_quality(cli_runner):
    from automil.cli import main

    result = cli_runner.invoke(main, ["verify-repro", "--help"])
    assert "after" in result.output.lower() or "porting" in result.output.lower()
    assert "tolerance" in result.output.lower()


def test_check_recognises_repro_manifest(tmp_path, cli_runner, monkeypatch):
    """After a successful verify-repro, automil check no longer warns
    about missing repro_manifest.yaml."""
    proj, adir = _setup_synthetic_project(tmp_path)
    monkeypatch.chdir(proj)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=proj,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    _write_graph(
        adir,
        {
            "node_0001": {
                "id": "node_0001",
                "type": "executed",
                "status": "keep",
                "primary_value": 0.5,
                "base_commit": head,
                "created_at": "2026-05-02T10:00:00Z",
            }
        },
    )
    from automil.cli import main

    cli_runner.invoke(main, ["verify-repro", "node_0001"])
    # Now run check.
    check_result = cli_runner.invoke(main, ["check"])
    # The "missing" warning is gone.
    assert "repro_manifest.yaml not found" not in check_result.output
