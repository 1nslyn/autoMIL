"""A parentless ``submit --node node_0001 --override "--hparams {}"`` is how a dev cell
registers its root baseline (``benchmarks/scripts/dev_loop.py``): no proposal, no files,
the trainer's own recipe with its default hparams.

Pinned here, on a project configured the way a dev cell is (architecture-preserving,
wall-clock cap, policy files as the only editable surface):

* with a ``graph.json`` the submit registers the root as a running node that no parent
  and no file change accompanies, and queues it as a config-only candidate;
* without one the submit still queues the run but registers nothing, which is why the
  batch driver creates the graph before its first submit.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from automil.cli import main
from automil.graph import locked_update

ROOT_OVERRIDE = "--hparams {}"
DEV_CELL_CONFIG = {
    "registry": {
        "mode": "architecture-preserving",
        "protected": ["train.py"],
        "allowed_override_options": ["--hparams", "--policy-variant"],
        "allowed_variant_kinds": ["policy"],
    },
    "files": {"editable": ["automil/variants/_policies/*.py"]},
    "cap": {"mode": "wall_clock", "budget": "30d", "safety_buffer": "1h", "eval_budget": 200},
}


def _init_git_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for command in (["git", "init"], ["git", "config", "user.name", "test"],
                    ["git", "config", "user.email", "test@test.com"]):
        subprocess.run(command, cwd=path, capture_output=True, check=True)
    (path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, capture_output=True, check=True)


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    """A git repo whose ``automil/`` is configured like a dev cell; returns the automil dir."""
    _init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert CliRunner().invoke(main, ["init"]).exit_code == 0
    adir = tmp_path / "automil"
    config = {**yaml.safe_load((adir / "config.yaml").read_text()), **DEV_CELL_CONFIG}
    (adir / "config.yaml").write_text(yaml.safe_dump(config))
    return adir


def _submit_root(adir: Path):
    return CliRunner().invoke(main, [
        "submit", "--node", "node_0001", "--desc", "baseline: native baseline",
        "--techniques", "recipe:baseline", "--override", ROOT_OVERRIDE,
    ])


def _queued(adir: Path) -> dict:
    return json.loads((adir / "orchestrator" / "queue" / "node_0001.json").read_text())


def test_root_submit_registers_a_running_root_node_when_the_graph_exists(project):
    with locked_update(project / "graph.json"):
        pass  # what the batch driver does before its first submit
    result = _submit_root(project)
    assert result.exit_code == 0, result.output

    graph = json.loads((project / "graph.json").read_text())
    assert list(graph["nodes"]) == ["node_0001"]
    root = graph["nodes"]["node_0001"]
    assert (root["type"], root["status"], root["parent_id"]) == ("proposed", "running", None)
    assert root["techniques"] == ["recipe:baseline"]
    assert root["description"] == "baseline: native baseline"

    spec = _queued(project)
    assert spec["run_command_override"] == ROOT_OVERRIDE
    assert not spec["overlay_manifest"]  # no file travels with the root
    assert spec["admissibility"]["candidate_class"] == "config-only"


def test_root_submit_without_a_graph_queues_the_run_but_registers_no_node(project):
    result = _submit_root(project)
    assert result.exit_code == 0, result.output
    assert _queued(project)["run_command_override"] == ROOT_OVERRIDE
    assert not (project / "graph.json").exists()
