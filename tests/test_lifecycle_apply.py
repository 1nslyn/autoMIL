"""Coverage for `automil apply <node_id>` (CLI-01 / D-41)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner


def _init_git_repo(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, capture_output=True)
    (path / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "."], cwd=path, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=path, capture_output=True, check=True)


def _setup(tmp_path: Path) -> Path:
    """Setup a git repo + run automil init; returns the automil/ dir."""
    _init_git_repo(tmp_path)
    import os
    os.chdir(tmp_path)
    from automil.cli import main
    CliRunner().invoke(main, ["init"])
    adir = tmp_path / "automil"
    config_path = adir / "config.yaml"
    config = yaml.safe_load(config_path.read_text()) or {}
    config.setdefault("cap", {})["mode"] = "wall_clock"
    config_path.write_text(yaml.safe_dump(config))
    return adir


def _write_graph(adir: Path, nodes: dict):
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


# ---------------------------------------------------------------------------
# Test 1: apply happy path — model only
# ---------------------------------------------------------------------------

def test_apply_model_only(tmp_path, cli_runner, monkeypatch):
    """variant_spec with kind=model populates model.variant + model.parent."""
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_graph(adir, {
        "node_0001": {
            "id": "node_0001",
            "type": "executed",
            "status": "keep",
            "primary_value": 0.5,
            "variant_spec": {"kind": "model", "name": "v0001", "parent": "p"},
        }
    })
    from automil.cli import main
    result = cli_runner.invoke(main, ["apply", "node_0001"])
    assert result.exit_code == 0, result.output
    cfg = yaml.safe_load((adir / "config.yaml").read_text())
    assert cfg["model"]["variant"] == "v0001"
    assert cfg["model"]["parent"] == "p"


# ---------------------------------------------------------------------------
# Test 2: apply happy path — loss only
# ---------------------------------------------------------------------------

def test_apply_loss_only(tmp_path, cli_runner, monkeypatch):
    """variant_spec with kind=loss populates loss.variant."""
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_graph(adir, {
        "node_0001": {
            "id": "node_0001",
            "type": "executed",
            "status": "keep",
            "primary_value": 0.5,
            "variant_spec": {"kind": "loss", "name": "l0001", "parent": None},
        }
    })
    from automil.cli import main
    result = cli_runner.invoke(main, ["apply", "node_0001"])
    assert result.exit_code == 0, result.output
    cfg = yaml.safe_load((adir / "config.yaml").read_text())
    assert cfg["loss"]["variant"] == "l0001"


# ---------------------------------------------------------------------------
# Test 3: apply happy path — policy only
# ---------------------------------------------------------------------------

def test_apply_policy_only(tmp_path, cli_runner, monkeypatch):
    """variant_spec with kind=policy populates policy.variant."""
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_graph(adir, {
        "node_0001": {
            "id": "node_0001",
            "type": "executed",
            "status": "keep",
            "primary_value": 0.5,
            "variant_spec": {"kind": "policy", "name": "sam", "parent": None},
        }
    })
    from automil.cli import main
    result = cli_runner.invoke(main, ["apply", "node_0001"])
    assert result.exit_code == 0, result.output
    cfg = yaml.safe_load((adir / "config.yaml").read_text())
    assert cfg["policy"]["variant"] == "sam"


# ---------------------------------------------------------------------------
# Test 4: apply happy path — combined recipe (model + loss + policy)
# ---------------------------------------------------------------------------

def test_apply_combined_recipe(tmp_path, cli_runner, monkeypatch):
    """recipe field with all three kinds updates all three sections."""
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_graph(adir, {
        "node_0001": {
            "id": "node_0001",
            "type": "executed",
            "status": "keep",
            "primary_value": 0.5,
            "recipe": [
                {"kind": "model", "name": "v0001", "parent": "p"},
                {"kind": "loss", "name": "l0001"},
                {"kind": "policy", "name": "sam"},
            ],
        }
    })
    from automil.cli import main
    result = cli_runner.invoke(main, ["apply", "node_0001"])
    assert result.exit_code == 0, result.output
    cfg = yaml.safe_load((adir / "config.yaml").read_text())
    assert cfg["model"]["variant"] == "v0001"
    assert cfg["model"]["parent"] == "p"
    assert cfg["loss"]["variant"] == "l0001"
    assert cfg["policy"]["variant"] == "sam"


# ---------------------------------------------------------------------------
# Test 5: idempotent — running twice produces byte-identical config
# ---------------------------------------------------------------------------

def test_apply_idempotent(tmp_path, cli_runner, monkeypatch):
    """Running apply twice on same node produces byte-identical config.yaml."""
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_graph(adir, {
        "node_0001": {
            "id": "node_0001",
            "type": "executed",
            "status": "keep",
            "primary_value": 0.5,
            "variant_spec": {"kind": "model", "name": "v0001", "parent": "p"},
        }
    })
    from automil.cli import main
    cli_runner.invoke(main, ["apply", "node_0001"])
    first = (adir / "config.yaml").read_text()
    cli_runner.invoke(main, ["apply", "node_0001"])
    second = (adir / "config.yaml").read_text()
    assert first == second


# ---------------------------------------------------------------------------
# Test 6: single .bak rolling — NOT a stack
# ---------------------------------------------------------------------------

def test_apply_single_bak_rolling(tmp_path, cli_runner, monkeypatch):
    """Repeated apply runs leave only ONE .bak file, not .bak.0/.bak.1/etc."""
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_graph(adir, {
        "node_0001": {
            "id": "node_0001",
            "type": "executed",
            "status": "keep",
            "variant_spec": {"kind": "model", "name": "v0001", "parent": "p"},
        },
        "node_0002": {
            "id": "node_0002",
            "type": "executed",
            "status": "keep",
            "variant_spec": {"kind": "model", "name": "v0002", "parent": "p"},
        },
    })
    from automil.cli import main
    cli_runner.invoke(main, ["apply", "node_0001"])
    cli_runner.invoke(main, ["apply", "node_0002"])
    # Only one .bak file should exist (not .bak.0, .bak.1, etc.)
    baks = list(adir.glob("config.yaml.bak*"))
    assert len(baks) == 1


# ---------------------------------------------------------------------------
# Test 7: .bak contents are the PREVIOUS config — not the original
# ---------------------------------------------------------------------------

def test_apply_bak_contains_previous(tmp_path, cli_runner, monkeypatch):
    """After two applies, .bak contains the config produced by the first apply."""
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_graph(adir, {
        "node_0001": {
            "id": "node_0001",
            "type": "executed",
            "status": "keep",
            "variant_spec": {"kind": "model", "name": "v0001", "parent": "p"},
        },
        "node_0002": {
            "id": "node_0002",
            "type": "executed",
            "status": "keep",
            "variant_spec": {"kind": "model", "name": "v0002", "parent": "p"},
        },
    })
    from automil.cli import main
    cli_runner.invoke(main, ["apply", "node_0001"])
    cli_runner.invoke(main, ["apply", "node_0002"])
    # .bak now contains the v0001 config (the version BEFORE the second apply).
    bak_cfg = yaml.safe_load((adir / "config.yaml.bak").read_text())
    assert bak_cfg["model"]["variant"] == "v0001"


# ---------------------------------------------------------------------------
# Test 8: atomic write — no .tmp leftover after success
# ---------------------------------------------------------------------------

def test_apply_no_tmp_leftover(tmp_path, cli_runner, monkeypatch):
    """No config.yaml*.tmp files persist after a successful apply."""
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_graph(adir, {
        "node_0001": {
            "id": "node_0001",
            "type": "executed",
            "status": "keep",
            "variant_spec": {"kind": "model", "name": "v0001", "parent": "p"},
        },
    })
    from automil.cli import main
    cli_runner.invoke(main, ["apply", "node_0001"])
    assert list(adir.glob("config.yaml*.tmp")) == []


# ---------------------------------------------------------------------------
# Test 9: missing node — error message includes "available:" + known node IDs
# ---------------------------------------------------------------------------

def test_apply_missing_node_lists_available(tmp_path, cli_runner, monkeypatch):
    """`apply node_9999` exits non-zero + lists known node IDs."""
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_graph(adir, {
        "node_0001": {"id": "node_0001", "type": "executed", "status": "keep"},
        "node_0042": {"id": "node_0042", "type": "executed", "status": "keep"},
    })
    from automil.cli import main
    result = cli_runner.invoke(main, ["apply", "node_9999"])
    assert result.exit_code != 0
    assert "available" in result.output.lower()
    assert "node_0001" in result.output
    assert "node_0042" in result.output


# ---------------------------------------------------------------------------
# Test 10: malformed config — model section is not a mapping
# ---------------------------------------------------------------------------

def test_apply_malformed_section_rejected(tmp_path, cli_runner, monkeypatch):
    """`model:` as a string (not mapping) causes apply to fail with clear message."""
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_graph(adir, {
        "node_0001": {
            "id": "node_0001",
            "type": "executed",
            "status": "keep",
            "variant_spec": {"kind": "model", "name": "v0001", "parent": "p"},
        },
    })
    # Corrupt config.yaml: model is a string instead of a mapping.
    cfg = yaml.safe_load((adir / "config.yaml").read_text()) or {}
    cfg["model"] = "I am a string, not a mapping"
    (adir / "config.yaml").write_text(yaml.safe_dump(cfg))

    from automil.cli import main
    result = cli_runner.invoke(main, ["apply", "node_0001"])
    assert result.exit_code != 0
    assert "not a mapping" in result.output.lower() or "model" in result.output


# ---------------------------------------------------------------------------
# Test 11: config.yaml missing — clear suggestion to run init
# ---------------------------------------------------------------------------

def test_apply_config_missing(tmp_path, cli_runner, monkeypatch):
    """apply fails with `Run automil init first` when config.yaml is absent."""
    # Setup git but skip automil init.
    _init_git_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    # No automil/config.yaml.
    from automil.cli import main
    result = cli_runner.invoke(main, ["apply", "node_0001"])
    assert result.exit_code != 0
    assert "init" in result.output.lower() or "config.yaml" in result.output


# ---------------------------------------------------------------------------
# Test 12: no codebase mutation — registry-first invariant (D-41)
# ---------------------------------------------------------------------------

def test_apply_no_codebase_mutation(tmp_path, cli_runner, monkeypatch):
    """D-41: apply ONLY edits config.yaml; no other files are mutated."""
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_graph(adir, {
        "node_0001": {
            "id": "node_0001",
            "type": "executed",
            "status": "keep",
            "variant_spec": {"kind": "model", "name": "v0001", "parent": "p"},
        },
    })
    # Snapshot mtimes of every file EXCEPT config.yaml and config.yaml.bak.
    snapshot = {
        p: p.stat().st_mtime
        for p in tmp_path.rglob("*")
        if p.is_file() and "config.yaml" not in p.name
    }
    from automil.cli import main
    cli_runner.invoke(main, ["apply", "node_0001"])
    for p, mt in snapshot.items():
        if p.exists():
            assert p.stat().st_mtime == mt, f"unexpected mutation: {p}"


# ---------------------------------------------------------------------------
# Test 13: --help workflow text mentions config + variant/code
# ---------------------------------------------------------------------------

def test_apply_help_workflow_text(cli_runner):
    """apply --help mentions config and variant/code (registry-first invariant)."""
    from automil.cli import main
    result = cli_runner.invoke(main, ["apply", "--help"])
    assert result.exit_code == 0
    assert "config" in result.output.lower()
    assert "code" in result.output.lower() or "variant" in result.output.lower()


# ---------------------------------------------------------------------------
# Test 14: node missing variant_spec — graceful error with port-variant hint
# ---------------------------------------------------------------------------

def test_apply_node_without_variant_spec(tmp_path, cli_runner, monkeypatch):
    """Node with no variant_spec/recipe → exits non-zero + suggests port-variant."""
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_graph(adir, {
        "node_0001": {
            "id": "node_0001",
            "type": "executed",
            "status": "keep",
            "primary_value": 0.5,
        },  # no variant_spec, no recipe
    })
    from automil.cli import main
    result = cli_runner.invoke(main, ["apply", "node_0001"])
    assert result.exit_code != 0
    # Should suggest running port-variant.
    assert "port-variant" in result.output or "variant_spec" in result.output


# ---------------------------------------------------------------------------
# Test 15: A1 fix — applied_variant.json written to archive/<node_id>/
# MANDATORY per plan 10-02 done criteria.
# ---------------------------------------------------------------------------

def test_apply_writes_applied_variant_json(tmp_path, cli_runner, monkeypatch):
    """A1 fix (D-01): apply writes applied_variant.json to archive/<node_id>/.

    The file must exist AND its content must match the selection dict with
    correct key names: {"model": {"variant": ..., "parent": ...}, "loss": {...}, "policy": {...}}.

    This proves the write mechanism; the consumer-side read (iris train.py dispatch)
    is tested separately in test_apl01_iris_dispatch.py (plan 10-04).
    """
    import json as _json
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_graph(adir, {
        "node_0001": {
            "id": "node_0001",
            "type": "executed",
            "status": "keep",
            "primary_value": 0.85,
            "variant_spec": {"kind": "model", "name": "classifier_v0", "parent": "baseline"},
        }
    })
    from automil.cli import main
    result = cli_runner.invoke(main, ["apply", "node_0001"])
    assert result.exit_code == 0, result.output

    # The applied_variant.json must exist in the archive dir.
    archive_path = adir / "orchestrator" / "archive" / "node_0001" / "applied_variant.json"
    assert archive_path.exists(), (
        f"applied_variant.json not found at {archive_path}. "
        "A1 fix: apply must write the selection snapshot for orchestrator overlay."
    )

    payload = _json.loads(archive_path.read_text())
    assert payload["model"]["variant"] == "classifier_v0", (
        f"model.variant mismatch: {payload}"
    )
    assert payload["model"]["parent"] == "baseline", (
        f"model.parent mismatch: {payload}"
    )
    assert "loss" in payload, f"Missing 'loss' key in applied_variant.json: {payload}"
    assert "policy" in payload, f"Missing 'policy' key in applied_variant.json: {payload}"

    # CR-01 fix: active_variant.json must also be written at the automil/ root
    # so submit can propagate it into the NEW node's archive.
    active_path = adir / "active_variant.json"
    assert active_path.exists(), (
        f"active_variant.json not found at {active_path}. "
        "CR-01 fix: apply must write automil/active_variant.json for submit propagation."
    )
    active_payload = _json.loads(active_path.read_text())
    assert active_payload["model"]["variant"] == "classifier_v0", (
        f"active_variant.json model.variant mismatch: {active_payload}"
    )


# ---------------------------------------------------------------------------
# Test 16: A1 fix — AUTOMIL_VARIANT_MODEL injected into queue spec env
# MANDATORY per plan 10-02 done criteria.
# ---------------------------------------------------------------------------

def test_apply_injects_env_var_into_queue_spec(tmp_path, cli_runner, monkeypatch):
    """A1 fix: apply injects AUTOMIL_VARIANT_MODEL into an existing queue spec's env.

    Set up a node with a model variant AND a pre-existing queue spec at
    automil/orchestrator/queue/{node_id}.json. After apply(), read back the
    spec and assert spec["env"]["AUTOMIL_VARIANT_MODEL"] equals the variant name.
    """
    import json as _json
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_graph(adir, {
        "node_0001": {
            "id": "node_0001",
            "type": "executed",
            "status": "keep",
            "primary_value": 0.85,
            "variant_spec": {"kind": "model", "name": "classifier_v0", "parent": "baseline"},
        }
    })

    # Create a pre-existing queue spec (simulates a submitted experiment).
    queue_dir = adir / "orchestrator" / "queue"
    queue_dir.mkdir(parents=True, exist_ok=True)
    pre_spec = {
        "id": "node_0001",
        "base_commit": "abc123",
        "overlay_dir": "archive/node_0001",
        "priority": 0,
    }
    (queue_dir / "node_0001.json").write_text(_json.dumps(pre_spec, indent=2))

    from automil.cli import main
    result = cli_runner.invoke(main, ["apply", "node_0001"])
    assert result.exit_code == 0, result.output

    # Read back the queue spec and verify env injection.
    updated_spec = _json.loads((queue_dir / "node_0001.json").read_text())
    assert "env" in updated_spec, (
        f"Queue spec missing 'env' key after apply: {updated_spec}"
    )
    assert updated_spec["env"]["AUTOMIL_VARIANT_MODEL"] == "classifier_v0", (
        f"AUTOMIL_VARIANT_MODEL not injected correctly: {updated_spec['env']}"
    )


# ---------------------------------------------------------------------------
# Test 17: CR-01 real-flow trap-closer — applied_variant.json reaches the NEW
# node's archive via apply → submit, WITHOUT the test manually placing it.
#
# This is the test that proves CR-01 is actually fixed. The prior tests
# (15, 16) inject or assert on archive/<applied_node>/ — they never exercise
# the apply→submit→overlay path. This test does:
#
#   1. automil apply <node_0001>    → writes active_variant.json to automil/
#   2. automil submit <node_0002>   → submit reads active_variant.json and
#                                     copies it into archive/node_0002/ as
#                                     applied_variant.json
#
# The test then asserts that archive/node_0002/applied_variant.json exists
# with correct content. The file is NOT placed there by the test setup.
# ---------------------------------------------------------------------------

def test_apply_then_submit_propagates_to_new_node_archive(
    tmp_path, cli_runner, monkeypatch
):
    """CR-01 real-flow: apply <old_node> then submit <new_node> propagates
    applied_variant.json into archive/<new_node>/ WITHOUT the test injecting it.

    This is the trap-closer. The bug was: apply wrote to archive/<old_node>/
    but submit creates archive/<new_node>/ — the two never intersected, so
    applied_variant.json never reached the new worktree. The fix: apply writes
    active_variant.json at the automil/ root; submit copies it into the same
    project-relative automil path under archive/<new_node>/ before writing the
    queue spec. That location is what the worktree runtime actually reads.
    """
    import json as _json

    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)

    # Graph with two nodes: node_0001 (already executed — the "good" result
    # we apply), node_0002 (proposal — the next experiment to submit).
    _write_graph(adir, {
        "node_0001": {
            "id": "node_0001",
            "type": "executed",
            "status": "keep",
            "primary_value": 0.88,
            "variant_spec": {"kind": "model", "name": "classifier_v0", "parent": "baseline"},
        },
        "node_0002": {
            "id": "node_0002",
            "type": "proposed",
            "status": "pending",
            "primary_value": 0.0,
        },
    })

    # Step 1: apply the good node's variant. This should write active_variant.json.
    from automil.cli import main
    apply_result = cli_runner.invoke(main, ["apply", "node_0001"])
    assert apply_result.exit_code == 0, apply_result.output

    # Verify active_variant.json was written at the framework level.
    active_path = adir / "active_variant.json"
    assert active_path.exists(), (
        f"active_variant.json not found at {active_path} after apply. "
        "CR-01 fix: apply must write to automil/active_variant.json."
    )

    # Step 2: submit the NEXT node (node_0002). Submit should copy
    # active_variant.json → archive/node_0002/automil/applied_variant.json.
    # We need a changed file to snapshot — create a dummy one.
    (tmp_path / "model.py").write_text("# next experiment\n")
    # Mark it as a changed tracked file (git add then unstage so it's dirty)
    import subprocess
    subprocess.run(["git", "add", "model.py"], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "add model"],
        cwd=tmp_path, capture_output=True, check=True,
    )
    # Now make a change so the file is detected as modified
    (tmp_path / "model.py").write_text("# next experiment v2\n")

    submit_result = cli_runner.invoke(
        main,
        [
            "submit",
            "--node", "node_0002",
            "--desc", "next experiment after apply",
            "--files", "model.py",
            "--parent", "node_0001",
            "--mil-model", "classifier_v0",
        ],
    )
    assert submit_result.exit_code == 0, submit_result.output

    # CRITICAL assertion: applied_variant.json must be in the NEW node's archive.
    # This file was NOT placed there by the test setup — it must arrive via
    # submit's propagation of active_variant.json.
    new_archive = adir / "orchestrator" / "archive" / "node_0002"
    applied_in_new = new_archive / "automil" / "applied_variant.json"
    assert applied_in_new.exists(), (
        f"CR-01 REGRESSION: applied_variant.json NOT found at {applied_in_new}. "
        "submit must preserve the project-relative automil path in the archive. "
        "Without this, apply_overlay never carries the selection into the worktree."
    )

    # Verify the content is correct (from the node_0001 apply).
    payload = _json.loads(applied_in_new.read_text())
    assert payload["model"]["variant"] == "classifier_v0", (
        f"CR-01: applied_variant.json in new node has wrong variant: {payload}"
    )
    assert payload["model"]["parent"] == "baseline", (
        f"CR-01: applied_variant.json in new node has wrong parent: {payload}"
    )

    # Also verify the overlay_dir in the queue spec points at the new node's archive,
    # confirming apply_overlay will find applied_variant.json there at run time.
    queue_spec_path = adir / "orchestrator" / "queue" / "node_0002.json"
    assert queue_spec_path.exists(), "Queue spec for node_0002 not written"
    spec = _json.loads(queue_spec_path.read_text())
    assert spec["overlay_dir"] == "archive/node_0002", (
        f"overlay_dir should point at new node's archive, got: {spec['overlay_dir']}"
    )
