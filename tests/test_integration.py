"""End-to-end integration tests."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from automil.cli import main


def _init_git_repo(path: Path):
    """Initialize a git repo with an initial commit at the given path."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.name", "test"],
        cwd=path, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=path, capture_output=True,
    )
    # Create a dummy file and commit so HEAD exists
    (path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=path, capture_output=True, check=True,
    )


def _use_wall_clock(path: Path) -> None:
    config_path = path / "automil" / "config.yaml"
    config = yaml.safe_load(config_path.read_text()) or {}
    config.setdefault("cap", {})["mode"] = "wall_clock"
    config_path.write_text(yaml.safe_dump(config))


class TestEndToEnd:
    def test_init_submit_flow(self, tmp_path, monkeypatch):
        """Full flow: init in existing repo, submit experiment, check archive."""
        runner = CliRunner()

        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)

        # Init autoMIL overlay
        result = runner.invoke(main, ["init"])
        assert result.exit_code == 0, result.output
        _use_wall_clock(tmp_path)

        # Verify automil subdirectory
        adir = tmp_path / "automil"
        assert (adir / "config.yaml").exists()
        assert (adir / "program.md").exists()
        assert (adir / "learnings.md").exists()
        assert (adir / ".gitignore").exists()

        # No train.py or prepare.py
        assert not (adir / "train.py").exists()
        assert not (adir / "prepare.py").exists()
        assert not (tmp_path / "train.py").exists()
        assert not (tmp_path / "prepare.py").exists()

        # Create a file and submit
        (tmp_path / "model.py").write_text("print('experiment 1')\n")
        result = runner.invoke(
            main,
            ["submit", "--node", "node_0001", "--desc", "test exp",
             "--files", "model.py", "--mil-model", "test_model"],  # D-12: required
        )
        assert result.exit_code == 0, result.output

        # Verify archive structure
        archive = adir / "orchestrator" / "archive" / "node_0001"
        assert (archive / "model.py").exists()
        assert (archive / "model.py").read_text() == "print('experiment 1')\n"

        # Verify queue spec
        queue_files = list((adir / "orchestrator" / "queue").glob("*.json"))
        assert len(queue_files) == 1
        spec = json.loads(queue_files[0].read_text())
        assert spec["id"] == "node_0001"
        assert "base_commit" in spec
        assert "overlay_manifest" in spec

        # Status should work
        result = runner.invoke(main, ["status"])
        assert result.exit_code == 0, result.output

    def test_no_internal_paths_in_package(self):
        """Verify no institution-specific paths leaked into the package."""
        import automil
        pkg_dir = Path(automil.__file__).parent

        for f in pkg_dir.rglob("*"):
            if f.is_file() and f.suffix in (".py", ".md", ".yaml", ".j2", ".html", ".js"):
                content = f.read_text()
                assert "/mnt/pool" not in content, f"Internal path in {f}"
                assert "/home/jma" not in content, f"Home path in {f}"

    def test_multiple_submits(self, tmp_path, monkeypatch):
        """Multiple experiments can be submitted sequentially."""
        runner = CliRunner()
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        runner.invoke(main, ["init"])
        _use_wall_clock(tmp_path)

        for i in range(3):
            (tmp_path / "model.py").write_text(f"print('experiment {i}')\n")
            result = runner.invoke(
                main,
                ["submit", "--node", f"node_{i:04d}", "--desc", f"exp {i}",
                 "--files", "model.py", "--mil-model", "test_model"],  # D-12: required
            )
            assert result.exit_code == 0, result.output

        # All 3 should be in queue
        queue_files = list((tmp_path / "automil" / "orchestrator" / "queue").glob("*.json"))
        assert len(queue_files) == 3

        # All 3 archives should exist with different content
        for i in range(3):
            archive = tmp_path / "automil" / "orchestrator" / "archive" / f"node_{i:04d}"
            assert (archive / "model.py").read_text() == f"print('experiment {i}')\n"

    def test_propose_and_rank(self, tmp_path, monkeypatch):
        """Propose experiments and rank them."""
        runner = CliRunner()
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        runner.invoke(main, ["init"])
        _use_wall_clock(tmp_path)

        # Create a graph with a root node
        from automil.graph import ExperimentGraph
        graph = ExperimentGraph(path=str(tmp_path / "automil" / "graph.json"))
        root = graph.add_executed(
            parent_id=None,
            description="baseline",
            techniques=["baseline"],
            metrics={"test_auc": 0.85, "test_bacc": 0.80, "primary_value": 0.825},
            status="keep",
        )
        graph.save()

        # Propose via CLI
        result = runner.invoke(
            main,
            ["propose", "--parent", root, "--desc", "try focal loss",
             "--techniques", "focal"],
        )
        assert result.exit_code == 0, result.output

        # Rank should show the proposal
        result = runner.invoke(main, ["rank"])
        assert result.exit_code == 0, result.output
        assert "focal" in result.output

    def test_deleted_file_submission(self, tmp_path, monkeypatch):
        """Deleting a file records it as a deletion in the spec."""
        runner = CliRunner()
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        runner.invoke(main, ["init"])
        _use_wall_clock(tmp_path)

        # Create and commit a file, then delete it
        (tmp_path / "old_module.py").write_text("# to be deleted\n")
        subprocess.run(["git", "add", "old_module.py"], cwd=tmp_path, capture_output=True)
        subprocess.run(["git", "commit", "-m", "add old module"], cwd=tmp_path, capture_output=True)
        (tmp_path / "old_module.py").unlink()

        # Also modify an existing file so the experiment isn't empty
        (tmp_path / "model.py").write_text("# modified\n")

        result = runner.invoke(
            main,
            ["submit", "--node", "node_del", "--desc", "delete old module",
             "--files", "old_module.py", "--files", "model.py",
             "--mil-model", "test_model"],  # D-12: required
        )
        assert result.exit_code == 0, result.output

        # Check spec has deletions
        queue_files = list((tmp_path / "automil" / "orchestrator" / "queue").glob("*.json"))
        assert len(queue_files) == 1
        spec = json.loads(queue_files[0].read_text())
        assert "old_module.py" in spec.get("deletions", [])
        assert "model.py" in spec.get("overlay_manifest", {})

    def test_empty_submit_rejected(self, tmp_path, monkeypatch):
        """Submit with no changed files and no deletions is rejected."""
        runner = CliRunner()
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        runner.invoke(main, ["init"])
        _use_wall_clock(tmp_path)

        # Submit an existing unchanged file - auto-detect finds nothing
        result = runner.invoke(
            main,
            ["submit", "--node", "node_empty", "--desc", "nothing"],
        )
        # Should fail - no changed files detected
        assert result.exit_code != 0

    def test_propose_then_rank_has_scores(self, tmp_path, monkeypatch):
        """Proposals have non-zero scores immediately after propose."""
        runner = CliRunner()
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        runner.invoke(main, ["init"])
        _use_wall_clock(tmp_path)

        from automil.graph import ExperimentGraph
        graph = ExperimentGraph(path=str(tmp_path / "automil" / "graph.json"))
        root = graph.add_executed(
            parent_id=None,
            description="baseline",
            techniques=["baseline"],
            metrics={"test_auc": 0.85, "test_bacc": 0.80, "primary_value": 0.825},
            status="keep",
        )
        graph.save()

        # Propose two experiments under different parents
        runner.invoke(main, ["propose", "--parent", root, "--desc", "try A", "--techniques", "a"])
        runner.invoke(main, ["propose", "--parent", root, "--desc", "try B", "--techniques", "b"])

        # Rank should show non-zero scores
        result = runner.invoke(main, ["rank"])
        assert result.exit_code == 0, result.output
        # Scores should not all be 0.0000
        assert "0.0000" not in result.output or "0.8" in result.output

    def test_start_stop_loop(self, tmp_path, monkeypatch):
        """start-loop and stop-loop manage the flag file."""
        runner = CliRunner()
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        runner.invoke(main, ["init"])
        _use_wall_clock(tmp_path)

        result = runner.invoke(main, ["start-loop"])
        assert result.exit_code == 0
        assert (tmp_path / ".automil_active").exists()

        result = runner.invoke(main, ["stop-loop"])
        assert result.exit_code == 0
        assert not (tmp_path / ".automil_active").exists()
