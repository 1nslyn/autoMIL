"""Tests for the automil CLI."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from automil.cli import main


@pytest.fixture
def cli_runner():
    return CliRunner()


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


class TestInit:
    def test_creates_automil_subdir(self, cli_runner, tmp_path, monkeypatch):
        """automil init creates an automil/ subdirectory in an existing repo."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)

        result = cli_runner.invoke(main, ["init"])
        assert result.exit_code == 0, result.output

        adir = tmp_path / "automil"
        assert (adir / "config.yaml").exists()
        assert (adir / "program.md").exists()
        assert (adir / "plan.md").exists()
        assert (adir / "learnings.md").exists()
        assert (adir / ".gitignore").exists()
        assert (adir / "orchestrator" / "queue").is_dir()
        assert (adir / "orchestrator" / "archive").is_dir()
        assert (adir / "orchestrator" / "completed").is_dir()
        assert (adir / "orchestrator" / "running").is_dir()

    def test_no_train_py_or_prepare_py(self, cli_runner, tmp_path, monkeypatch):
        """automil init does not create train.py or prepare.py."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)

        cli_runner.invoke(main, ["init"])

        assert not (tmp_path / "automil" / "train.py").exists()
        assert not (tmp_path / "automil" / "prepare.py").exists()
        assert not (tmp_path / "train.py").exists()
        assert not (tmp_path / "prepare.py").exists()

    def test_requires_git_repo(self, cli_runner, tmp_path, monkeypatch):
        """automil init errors if not in a git repo."""
        tmp_path.mkdir(parents=True, exist_ok=True)
        monkeypatch.chdir(tmp_path)

        result = cli_runner.invoke(main, ["init"])
        assert result.exit_code != 0
        assert "git repository" in result.output

    def test_errors_if_already_initialized(self, cli_runner, tmp_path, monkeypatch):
        """automil init errors if automil/ already has config.yaml."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)

        cli_runner.invoke(main, ["init"])
        result = cli_runner.invoke(main, ["init"])
        assert result.exit_code != 0
        assert "already initialized" in result.output

    def test_gitignore_excludes_runtime(self, cli_runner, tmp_path, monkeypatch):
        """Runtime files (graph.json, results.tsv, orchestrator/) are gitignored."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)

        cli_runner.invoke(main, ["init"])

        gitignore = (tmp_path / "automil" / ".gitignore").read_text()
        assert "graph.json" in gitignore
        assert "results.tsv" in gitignore
        assert "orchestrator/" in gitignore

    def test_update_refreshes_stale_skill(self, cli_runner, tmp_path, monkeypatch):
        """`automil init --update` refreshes a stale framework-managed skill.

        Regression: skills used `if not dst.exists()` and ignored --update
        entirely, so updated agent guidance (e.g. the anti-HP-sweep directives)
        could never reach an already-initialized project.
        """
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        skill = tmp_path / ".claude" / "skills" / "automil" / "SKILL.md"
        assert skill.exists(), "init should install the automil skill"
        canonical = skill.read_text()
        assert len(canonical.splitlines()) > 80, "canonical skill should be the full version"

        # Simulate drift: an older/truncated skill already on disk.
        skill.write_text("# stale skill\n")

        result = cli_runner.invoke(main, ["init", "--update"])
        assert result.exit_code == 0, result.output
        assert skill.read_text() == canonical, (
            "init --update must refresh the stale skill to the canonical version"
        )

    def test_registers_activity_hooks(self, cli_runner, tmp_path, monkeypatch):
        """init installs on_tool.sh and registers the activity hooks (P2.1)."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        hook = tmp_path / ".claude" / "hooks" / "on_tool.sh"
        assert hook.exists(), "on_tool.sh should be installed"

        settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
        hooks = settings.get("hooks", {})
        assert "PostToolUse" in hooks
        post = hooks["PostToolUse"][0]
        assert "on_tool.sh" in str(post)
        assert post.get("matcher") == "*"
        assert post["hooks"][0].get("async") is True
        # UserPromptSubmit + SessionStart also wired
        assert any("on_tool.sh" in str(e) for e in hooks.get("UserPromptSubmit", []))
        assert any("on_tool.sh" in str(e) for e in hooks.get("SessionStart", []))
        # Existing Stop hook preserved
        assert any("on_stop.sh" in str(e) for e in hooks.get("Stop", []))

        # Idempotent: --update must not duplicate entries.
        cli_runner.invoke(main, ["init", "--update"])
        settings2 = json.loads((tmp_path / ".claude" / "settings.json").read_text())
        assert len(settings2["hooks"]["PostToolUse"]) == 1
        assert len(settings2["hooks"]["Stop"]) == 1


class TestCheck:
    def test_check_reports_placeholder_paths(self, cli_runner, tmp_path, monkeypatch):
        """automil check warns about placeholder data paths."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])
        result = cli_runner.invoke(main, ["check"])
        assert result.exit_code == 0
        assert "placeholder" in result.output.lower() or "ISSUES" in result.output

    def test_check_accepts_existing_relative_data_paths(self, cli_runner, tmp_path, monkeypatch):
        """Relative data paths should be resolved from the project root."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        features_dir = tmp_path / "data" / "features"
        splits_dir = tmp_path / "data" / "splits"
        mapping_csv = tmp_path / "data" / "mapping.csv"
        features_dir.mkdir(parents=True)
        splits_dir.mkdir(parents=True)
        mapping_csv.write_text("slide_id,label\n")

        config_path = tmp_path / "automil" / "config.yaml"
        config_text = config_path.read_text()
        config_text = config_text.replace("/path/to/features/", "data/features")
        config_text = config_text.replace("/path/to/splits/", "data/splits")
        config_text = config_text.replace("/path/to/mapping.csv", "data/mapping.csv")
        (tmp_path / "train.py").write_text("open('result.json', 'w').write('{}')\n")
        config_text = config_text.replace('script: "train.py"', 'script: "train.py"')
        config_path.write_text(config_text)

        result = cli_runner.invoke(main, ["check"])
        assert result.exit_code == 0
        assert "data.features_dir path does not exist" not in result.output
        assert "data.splits_dir path does not exist" not in result.output
        assert "data.mapping_csv path does not exist" not in result.output


class TestSubmit:
    def test_submit_captures_files(self, cli_runner, tmp_path, monkeypatch):
        """automil submit snapshots specified files to archive."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        # Create a file to submit
        (tmp_path / "model.py").write_text("print('modified')\n")

        # Submit
        result = cli_runner.invoke(
            main,
            ["submit", "--node", "node_0001", "--desc", "test", "--files", "model.py",
             "--mil-model", "test_model"],  # D-12: required
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        # Check archive
        archive = tmp_path / "automil" / "orchestrator" / "archive" / "node_0001"
        assert (archive / "model.py").exists()
        assert (archive / "model.py").read_text() == "print('modified')\n"

        # Check spec in queue
        queue_files = list((tmp_path / "automil" / "orchestrator" / "queue").glob("*.json"))
        assert len(queue_files) == 1
        spec = json.loads(queue_files[0].read_text())
        assert spec["id"] == "node_0001"
        assert "base_commit" in spec

    def test_submit_auto_detect_supports_directory_scopes(self, cli_runner, tmp_path, monkeypatch):
        """Directory scopes in files.editable should capture changed files beneath them."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        config_path = tmp_path / "automil" / "config.yaml"
        config_text = config_path.read_text().replace("editable: []", 'editable: ["models/"]')
        config_path.write_text(config_text)

        (tmp_path / "models").mkdir()
        (tmp_path / "models" / "custom.py").write_text("print('changed')\n")
        (tmp_path / "notes.txt").write_text("ignore me\n")

        result = cli_runner.invoke(
            main,
            ["submit", "--node", "node_scope", "--desc", "dir scope",
             "--mil-model", "test_model"],  # D-12: required
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        archive = tmp_path / "automil" / "orchestrator" / "archive" / "node_scope"
        assert (archive / "models" / "custom.py").exists()
        assert not (archive / "notes.txt").exists()

    def test_submit_warns_for_readonly_glob(self, cli_runner, tmp_path, monkeypatch):
        """Readonly glob patterns should warn when explicit files match them."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        config_path = tmp_path / "automil" / "config.yaml"
        config_text = config_path.read_text().replace("readonly: []", 'readonly: ["data/*.py"]')
        config_path.write_text(config_text)

        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "prepare.py").write_text("print('readonly')\n")

        result = cli_runner.invoke(
            main,
            ["submit", "--node", "node_ro", "--desc", "readonly", "--files", "data/prepare.py",
             "--mil-model", "test_model"],  # D-12: required
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "marked readonly" in result.output

    def test_submit_refuses_proposed_parent(self, cli_runner, tmp_path, monkeypatch):
        """Guard 1: submit must refuse when --parent is a pending proposal."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        # Build a graph containing one kept executed node and one pending
        # proposal hanging off it.
        from automil.graph import ExperimentGraph
        graph = ExperimentGraph(path=str(tmp_path / "automil" / "graph.json"))
        root = graph.add_executed(
            parent_id=None, description="baseline", techniques=[],
            metrics={"composite": 0.80, "test_auc": 0.82, "test_bacc": 0.78,
                     "val_auc": 0.82, "val_bacc": 0.78,
                     "vram_gb": 0.4, "elapsed_min": 60, "gpu": 0},
            status="keep",
        )
        pending = graph.add_proposed(
            parent_id=root, description="pending parent", techniques=["x"],
        )
        graph.save()

        (tmp_path / "model.py").write_text("print('child')\n")
        result = cli_runner.invoke(
            main,
            ["submit", "--node", "node_0099", "--desc", "child of pending",
             "--parent", pending, "--files", "model.py"],
        )
        assert result.exit_code != 0
        assert "not been executed" in result.output or "type=proposed" in result.output

    def test_submit_refuses_unknown_parent(self, cli_runner, tmp_path, monkeypatch):
        """Guard 1: submit must refuse an unknown --parent id."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        from automil.graph import ExperimentGraph
        graph = ExperimentGraph(path=str(tmp_path / "automil" / "graph.json"))
        graph.add_executed(
            parent_id=None, description="baseline", techniques=[],
            metrics={"composite": 0.80, "test_auc": 0.82, "test_bacc": 0.78,
                     "val_auc": 0.82, "val_bacc": 0.78,
                     "vram_gb": 0.4, "elapsed_min": 60, "gpu": 0},
            status="keep",
        )
        graph.save()

        (tmp_path / "model.py").write_text("print('child')\n")
        result = cli_runner.invoke(
            main,
            ["submit", "--node", "node_0050", "--desc", "orphan",
             "--parent", "node_9999", "--files", "model.py"],
        )
        assert result.exit_code != 0
        assert "does not exist" in result.output

    def test_submit_allows_kept_parent(self, cli_runner, tmp_path, monkeypatch):
        """Guard 1: submit must still allow a normal kept parent."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        from automil.graph import ExperimentGraph
        graph = ExperimentGraph(path=str(tmp_path / "automil" / "graph.json"))
        root = graph.add_executed(
            parent_id=None, description="baseline", techniques=[],
            metrics={"composite": 0.80, "test_auc": 0.82, "test_bacc": 0.78,
                     "val_auc": 0.82, "val_bacc": 0.78,
                     "vram_gb": 0.4, "elapsed_min": 60, "gpu": 0},
            status="keep",
        )
        graph.save()

        (tmp_path / "model.py").write_text("print('child')\n")
        result = cli_runner.invoke(
            main,
            ["submit", "--node", "node_0002", "--desc", "legit child",
             "--parent", root, "--files", "model.py",
             "--mil-model", "test_model"],  # D-12: required
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output


class TestPropose:
    def test_propose_refuses_exact_duplicate(self, cli_runner, tmp_path, monkeypatch):
        """Guard 3: propose must refuse an exact-description sibling duplicate."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        from automil.graph import ExperimentGraph
        graph = ExperimentGraph(path=str(tmp_path / "automil" / "graph.json"))
        root = graph.add_executed(
            parent_id=None, description="baseline", techniques=[],
            metrics={"composite": 0.80, "test_auc": 0.82, "test_bacc": 0.78,
                     "val_auc": 0.82, "val_bacc": 0.78,
                     "vram_gb": 0.4, "elapsed_min": 60, "gpu": 0},
            status="keep",
        )
        graph.save()

        # First propose succeeds.
        r1 = cli_runner.invoke(
            main,
            ["propose", "--parent", root, "--desc", "try dropout 0.4"],
            catch_exceptions=False,
        )
        assert r1.exit_code == 0

        # Second propose with same parent + same description refused.
        r2 = cli_runner.invoke(
            main,
            ["propose", "--parent", root, "--desc", "try dropout 0.4"],
        )
        assert r2.exit_code != 0
        assert "already exists" in r2.output

    def test_propose_allows_different_parent(self, cli_runner, tmp_path, monkeypatch):
        """Guard 3: same description under a different parent is allowed."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        from automil.graph import ExperimentGraph
        graph = ExperimentGraph(path=str(tmp_path / "automil" / "graph.json"))
        root_a = graph.add_executed(
            parent_id=None, description="A", techniques=[],
            metrics={"composite": 0.80, "test_auc": 0.82, "test_bacc": 0.78,
                     "val_auc": 0.82, "val_bacc": 0.78,
                     "vram_gb": 0.4, "elapsed_min": 60, "gpu": 0},
            status="keep",
        )
        root_b = graph.add_executed(
            parent_id=None, description="B", techniques=[],
            metrics={"composite": 0.81, "test_auc": 0.83, "test_bacc": 0.79,
                     "val_auc": 0.83, "val_bacc": 0.79,
                     "vram_gb": 0.4, "elapsed_min": 60, "gpu": 0},
            status="keep",
        )
        graph.save()

        r1 = cli_runner.invoke(
            main,
            ["propose", "--parent", root_a, "--desc", "shared desc"],
            catch_exceptions=False,
        )
        assert r1.exit_code == 0
        r2 = cli_runner.invoke(
            main,
            ["propose", "--parent", root_b, "--desc", "shared desc"],
            catch_exceptions=False,
        )
        assert r2.exit_code == 0


class TestRank:
    def test_rank_outputs_proposals(self, cli_runner, tmp_path, monkeypatch):
        """automil rank shows top proposals from graph.json."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        # Create a graph with proposals
        from automil.graph import ExperimentGraph
        graph = ExperimentGraph(path=str(tmp_path / "automil" / "graph.json"))
        root = graph.add_executed(
            parent_id=None,
            description="baseline",
            techniques=["baseline"],
            metrics={"test_auc": 0.85, "test_bacc": 0.80, "composite": 0.825},
            status="keep",
        )
        graph.add_proposed(
            parent_id=root,
            description="try focal loss",
            techniques=["focal"],
        )
        graph.save()

        result = cli_runner.invoke(main, ["rank"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "focal" in result.output


class TestSubmitPathValidation:
    """T-00-01: submit rejects absolute paths, .. traversal, and resolve-escapes."""

    def test_submit_rejects_absolute_path(self, cli_runner, tmp_path, monkeypatch):
        """submit must reject --files with an absolute path (T-00-01)."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        result = cli_runner.invoke(
            main,
            ["submit", "--node", "node_abs", "--desc", "absolute path attack",
             "--files", "/etc/passwd"],
        )
        assert result.exit_code != 0
        assert "Invalid path" in result.output or "absolute" in result.output.lower()

    def test_submit_rejects_dotdot_traversal(self, cli_runner, tmp_path, monkeypatch):
        """submit must reject --files containing .. path components (T-00-01)."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        result = cli_runner.invoke(
            main,
            ["submit", "--node", "node_dotdot", "--desc", "dotdot traversal attack",
             "--files", "../secret.txt"],
        )
        assert result.exit_code != 0
        assert "Invalid path" in result.output or ".." in result.output

    def test_submit_rejects_escape_via_resolve(self, cli_runner, tmp_path, monkeypatch):
        """submit must reject a symlink that resolves outside the git root (T-00-01)."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        # Create a symlink inside the repo that points outside the repo root
        outside_target = tmp_path.parent / "outside_secret.txt"
        outside_target.write_text("secret content\n")
        symlink = tmp_path / "escape_link.txt"
        symlink.symlink_to(outside_target)

        result = cli_runner.invoke(
            main,
            ["submit", "--node", "node_escape", "--desc", "symlink escape attack",
             "--files", "escape_link.txt"],
        )
        assert result.exit_code != 0
        assert "escapes" in result.output or "Path escapes" in result.output

    def test_submit_auto_detect_excludes_automil_dir(self, cli_runner, tmp_path, monkeypatch):
        """Auto-detect must not include automil/ or .claude/ files in the snapshot (T-00-03)."""
        _init_git_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        cli_runner.invoke(main, ["init"])

        # Create a real model file that should be captured
        (tmp_path / "model.py").write_text("print('model changed')\n")

        # Create automil/ and .claude/ files that must NOT be captured
        (tmp_path / "automil" / "graph.json").write_text('{"nodes": {}}\n')
        (tmp_path / ".claude").mkdir(exist_ok=True)
        (tmp_path / ".claude" / "foo.md").write_text("agent notes\n")

        result = cli_runner.invoke(
            main,
            ["submit", "--node", "node_autodetect", "--desc", "auto-detect exclusion test",
             "--mil-model", "test_model"],  # D-12: required
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output

        # Queue file must list model.py but NOT automil/graph.json or .claude/foo.md
        queue_files = list((tmp_path / "automil" / "orchestrator" / "queue").glob("*.json"))
        assert len(queue_files) == 1
        spec = json.loads(queue_files[0].read_text())
        manifest_keys = set(spec.get("overlay_manifest", {}).keys())

        assert "model.py" in manifest_keys, "model.py should be in the manifest"
        assert not any(k.startswith("automil/") for k in manifest_keys), (
            f"automil/ files leaked into manifest: {manifest_keys}"
        )
        assert not any(k.startswith(".claude/") for k in manifest_keys), (
            f".claude/ files leaked into manifest: {manifest_keys}"
        )


class TestCliHelp:
    """CLN-06: automil --help must list all 11 subcommands (regression sentinel for CLI split)."""

    def test_main_help_lists_all_11_subcommands(self, cli_runner):
        """automil --help output contains all 11 expected subcommands (CLN-06)."""
        expected_commands = {
            "check",
            "init",
            "orchestrator",
            "propose",
            "rank",
            "reconcile",
            "start-loop",
            "status",
            "stop-loop",
            "submit",
            "viz",
        }

        result = cli_runner.invoke(main, ["--help"])
        assert result.exit_code == 0, result.output

        output = result.output
        missing = [cmd for cmd in expected_commands if cmd not in output]
        assert not missing, (
            f"automil --help is missing subcommands: {missing}\n"
            f"Full output:\n{output}"
        )


# ---------------------------------------------------------------------------
# OPS-03 RED stub (Wave 0 — Nyquist compliance)
# ---------------------------------------------------------------------------


def test_submit_existing_pending_marks_running(cli_runner, tmp_path, monkeypatch):
    """Submit against existing type=proposed,status=pending node transitions it to running.

    OPS-03: submit.py locked_update block (~L497) is missing an else branch that calls
    graph.mark_running() for pre-existing pending nodes. The current code only calls
    mark_running for newly-created nodes inside the `if not graph.get_node(node)` branch.
    """
    # Create a minimal git repo + automil structure.
    _init_git_repo(tmp_path)
    adir = tmp_path / "automil"
    adir.mkdir(exist_ok=True)
    (adir / "config.yaml").write_text(
        "run:\n  script: train.py\n  mil_model: clam_sb\n"
    )
    (adir / "orchestrator" / "queue").mkdir(parents=True, exist_ok=True)

    # Dummy training script (submit preflight checks it exists).
    (tmp_path / "train.py").write_text("# dummy\n")

    node_id = "node_0030"

    # Pre-write graph with a pending node (type=proposed, status=pending).
    graph_data = {
        "schema_version": 1,
        "meta": {
            "best_composite": 0.0,
            "best_node_id": None,
            "total_executed": 0,
            "total_proposed": 1,
            "next_id": 10,
            "baseline_composite": 0.0,
            "scoring": {
                "exploration_weight": 0.005,
                "novelty_weight": 0.003,
            },
        },
        "nodes": {
            node_id: {
                "id": node_id,
                "parent_id": None,
                "type": "proposed",
                "status": "pending",
                "description": "pre-existing pending node",
                "techniques": [],
                "metadata": {},
            }
        },
        "technique_stats": {},
    }
    (adir / "graph.json").write_text(json.dumps(graph_data, indent=2))

    monkeypatch.chdir(tmp_path)

    result = cli_runner.invoke(
        main,
        [
            "submit",
            "--node", node_id,
            "--desc", "test existing pending",
            "--files", "train.py",
            "--mil-model", "clam_sb",
        ],
        catch_exceptions=False,
    )

    # After submit, graph node must be running (OPS-03 success criterion).
    graph = json.loads((adir / "graph.json").read_text())
    assert graph["nodes"][node_id]["status"] == "running", (
        f"expected status='running', got {graph['nodes'][node_id].get('status')!r}\n"
        f"submit output: {result.output}"
    )
