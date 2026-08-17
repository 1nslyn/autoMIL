"""Coverage for `automil port-variant` (CLI-05 / D-43, D-44)."""
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
    subprocess.run(["git", "commit", "-m", "i"], cwd=path, capture_output=True, check=True)


def _setup(tmp_path: Path) -> Path:
    _init_git_repo(tmp_path)
    from automil.cli import main
    import os
    os.chdir(tmp_path)
    CliRunner().invoke(main, ["init"])
    return tmp_path / "automil"


def _write_archive_spec(adir: Path, node_id: str, *, overlay: dict, parent: str | None,
                        base_commit: str = "abc1234", primary_value: float = 0.5,
                        techniques: list | None = None):
    archive = adir / "orchestrator" / "archive" / node_id
    archive.mkdir(parents=True, exist_ok=True)
    spec = {
        "id": node_id,
        "base_commit": base_commit,
        "overlay_manifest": overlay,
        "deletions": [],
        "primary_value": primary_value,
        "graph_metadata": {"parent_id": parent, "techniques": techniques or []},
    }
    (archive / "spec.json").write_text(json.dumps(spec, indent=2))


def _write_graph_with_node(adir: Path, node_id: str, *, primary_value: float = 0.5):
    graph = {
        "schema_version": 1,
        "meta": {"best_node_id": node_id, "best_primary_value": primary_value,
                 "total_executed": 1, "total_proposed": 0,
                 "next_id": 2, "baseline_primary_value": 0.0,
                 "scoring": {"exploration_weight": 0.005, "novelty_weight": 0.003}},
        "nodes": {
            node_id: {"id": node_id, "type": "executed", "status": "keep",
                      "primary_value": primary_value, "base_commit": "abc1234",
                      "created_at": "2026-05-02T10:00:00Z"}
        },
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


def test_happy_port_model_auto(tmp_path, cli_runner, monkeypatch):
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_archive_spec(adir, "node_0176",
                        overlay={"models/model_clam.py": "sha256:..."},
                        parent="clam_mb")
    _write_graph_with_node(adir, "node_0176")

    from automil.cli import main
    result = cli_runner.invoke(main, ["port-variant", "node_0176"])
    assert result.exit_code == 0, result.output
    var = adir / "variants" / "clam_mb" / "clam_mb_v0176.py"
    man = adir / "variants" / "clam_mb" / "clam_mb_v0176.json"
    assert var.exists()
    assert man.exists()


def test_auto_name_format(tmp_path, cli_runner, monkeypatch):
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_archive_spec(adir, "node_0042",
                        overlay={"models/model_clam.py": "sha256:..."},
                        parent="clam_mb")
    _write_graph_with_node(adir, "node_0042")
    from automil.cli import main
    result = cli_runner.invoke(main, ["port-variant", "node_0042"])
    assert result.exit_code == 0, result.output
    assert (adir / "variants" / "clam_mb" / "clam_mb_v0042.py").exists()


def test_auto_kind_loss(tmp_path, cli_runner, monkeypatch):
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_archive_spec(adir, "node_0050",
                        overlay={"utils/core_utils.py": "sha256:..."},
                        parent=None)
    _write_graph_with_node(adir, "node_0050")
    from automil.cli import main
    result = cli_runner.invoke(main, ["port-variant", "node_0050"])
    assert result.exit_code == 0, result.output
    # loss variants live under _losses/ with name = "loss_v<short>".
    assert (adir / "variants" / "_losses" / "loss_v0050.py").exists()


def test_auto_kind_policy(tmp_path, cli_runner, monkeypatch):
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_archive_spec(adir, "node_0060",
                        overlay={"training/optimizer.py": "sha256:..."},
                        parent=None)
    _write_graph_with_node(adir, "node_0060")
    from automil.cli import main
    result = cli_runner.invoke(main, ["port-variant", "node_0060"])
    assert result.exit_code == 0, result.output
    assert (adir / "variants" / "_policies" / "policy_v0060.py").exists()


def test_ambiguous_kind_requires_flag(tmp_path, cli_runner, monkeypatch):
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_archive_spec(adir, "node_0070", overlay={
        "models/model_clam.py": "sha256:...",
        "training/optimizer.py": "sha256:...",
    }, parent="clam_mb")
    _write_graph_with_node(adir, "node_0070")
    from automil.cli import main
    result = cli_runner.invoke(main, ["port-variant", "node_0070"])
    assert result.exit_code != 0
    assert "--kind" in result.output


def test_kind_override_resolves_ambiguity(tmp_path, cli_runner, monkeypatch):
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_archive_spec(adir, "node_0080", overlay={
        "models/model_clam.py": "sha256:...",
        "training/optimizer.py": "sha256:...",
    }, parent="clam_mb")
    _write_graph_with_node(adir, "node_0080")
    from automil.cli import main
    result = cli_runner.invoke(main, ["port-variant", "node_0080", "--kind", "model"])
    assert result.exit_code == 0, result.output
    assert (adir / "variants" / "clam_mb" / "clam_mb_v0080.py").exists()


def test_name_override(tmp_path, cli_runner, monkeypatch):
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_archive_spec(adir, "node_0090",
                        overlay={"models/model_clam.py": "sha256:..."},
                        parent="clam_mb")
    _write_graph_with_node(adir, "node_0090")
    from automil.cli import main
    result = cli_runner.invoke(main, ["port-variant", "node_0090", "--name", "my_custom_name"])
    assert result.exit_code == 0, result.output
    assert (adir / "variants" / "clam_mb" / "my_custom_name.py").exists()


def test_parent_override(tmp_path, cli_runner, monkeypatch):
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_archive_spec(adir, "node_0100",
                        overlay={"models/model_clam.py": "sha256:..."},
                        parent=None)
    _write_graph_with_node(adir, "node_0100")
    from automil.cli import main
    result = cli_runner.invoke(main, ["port-variant", "node_0100", "--parent", "ab_mil"])
    assert result.exit_code == 0, result.output
    assert (adir / "variants" / "ab_mil" / "ab_mil_v0100.py").exists()


def test_idempotent_same_node_id(tmp_path, cli_runner, monkeypatch):
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_archive_spec(adir, "node_0110",
                        overlay={"models/model_clam.py": "sha256:..."},
                        parent="clam_mb")
    _write_graph_with_node(adir, "node_0110")
    from automil.cli import main
    result1 = cli_runner.invoke(main, ["port-variant", "node_0110"])
    assert result1.exit_code == 0, result1.output
    var_path = adir / "variants" / "clam_mb" / "clam_mb_v0110.py"
    first_mtime = var_path.stat().st_mtime
    # Re-port; should be no-op.
    result = cli_runner.invoke(main, ["port-variant", "node_0110"])
    assert result.exit_code == 0
    assert "already ported" in result.output.lower() or "no-op" in result.output.lower()
    assert var_path.stat().st_mtime == first_mtime


def test_mismatched_node_id_same_name_hard_fail(tmp_path, cli_runner, monkeypatch):
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_archive_spec(adir, "node_0120", overlay={"models/model_clam.py": "sha:..."}, parent="clam_mb")
    _write_archive_spec(adir, "node_0121", overlay={"models/model_clam.py": "sha:..."}, parent="clam_mb")
    _write_graph_with_node(adir, "node_0120")
    # Patch graph to ALSO include node_0121.
    graph = json.loads((adir / "graph.json").read_text())
    graph["nodes"]["node_0121"] = {"id": "node_0121", "type": "executed", "status": "keep",
                                   "primary_value": 0.5, "base_commit": "abc1234",
                                   "created_at": "2026-05-02T10:00:00Z"}
    (adir / "graph.json").write_text(json.dumps(graph, indent=2))

    from automil.cli import main
    result1 = cli_runner.invoke(main, ["port-variant", "node_0120", "--name", "shared_name"])
    assert result1.exit_code == 0, result1.output
    result = cli_runner.invoke(main, ["port-variant", "node_0121", "--name", "shared_name"])
    assert result.exit_code != 0
    assert "node_id" in result.output
    assert "node_0120" in result.output  # the existing one
    assert "node_0121" in result.output  # the new one


def test_manifest_schema(tmp_path, cli_runner, monkeypatch):
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_archive_spec(adir, "node_0130",
                        overlay={"models/model_clam.py": "sha:..."},
                        parent="clam_mb")
    _write_graph_with_node(adir, "node_0130")
    from automil.cli import main
    result = cli_runner.invoke(main, ["port-variant", "node_0130"])
    assert result.exit_code == 0, result.output
    manifest_path = adir / "variants" / "clam_mb" / "clam_mb_v0130.json"
    manifest = json.loads(manifest_path.read_text())
    for key in ("spec", "source_node", "source_overlay_files", "ported_at", "tool_version"):
        assert key in manifest


def test_module_body_has_register_and_docstring(tmp_path, cli_runner, monkeypatch):
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_archive_spec(adir, "node_0140",
                        overlay={"models/model_clam.py": "sha:..."},
                        parent="clam_mb")
    _write_graph_with_node(adir, "node_0140")
    from automil.cli import main
    result = cli_runner.invoke(main, ["port-variant", "node_0140"])
    assert result.exit_code == 0, result.output
    body = (adir / "variants" / "clam_mb" / "clam_mb_v0140.py").read_text()
    assert "@register" in body
    assert "ModelVariant" in body
    assert "Parent: clam_mb" in body
    assert "Node ID: node_0140" in body


def test_calls_refresh_registry(tmp_path, cli_runner, monkeypatch):
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_archive_spec(adir, "node_0150",
                        overlay={"models/model_clam.py": "sha:..."},
                        parent="clam_mb")
    _write_graph_with_node(adir, "node_0150")
    from automil.cli import main
    result = cli_runner.invoke(main, ["port-variant", "node_0150"])
    assert result.exit_code == 0, result.output
    init_text = (adir / "variants" / "clam_mb" / "__init__.py").read_text()
    assert "from . import clam_mb_v0150" in init_text


def test_missing_spec_json_hard_fail(tmp_path, cli_runner, monkeypatch):
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_graph_with_node(adir, "node_0160")
    from automil.cli import main
    result = cli_runner.invoke(main, ["port-variant", "node_0160"])
    assert result.exit_code != 0
    assert "spec.json" in result.output


def test_empty_overlay_manifest_hard_fail(tmp_path, cli_runner, monkeypatch):
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_archive_spec(adir, "node_0170", overlay={}, parent="clam_mb")
    _write_graph_with_node(adir, "node_0170")
    from automil.cli import main
    result = cli_runner.invoke(main, ["port-variant", "node_0170"])
    assert result.exit_code != 0
    assert "overlay" in result.output.lower() or "empty" in result.output.lower()


def test_model_without_parent_hard_fail(tmp_path, cli_runner, monkeypatch):
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    # Overlay path has no parent signal AND graph_metadata.parent_id is None.
    _write_archive_spec(adir, "node_0180",
                        overlay={"models/some_random_path.py": "sha:..."},
                        parent=None)
    _write_graph_with_node(adir, "node_0180")
    from automil.cli import main
    result = cli_runner.invoke(main, ["port-variant", "node_0180", "--kind", "model"])
    assert result.exit_code != 0
    assert "--parent" in result.output


def test_help_quality(cli_runner):
    from automil.cli import main
    result = cli_runner.invoke(main, ["port-variant", "--help"])
    assert result.exit_code == 0
    assert "--kind" in result.output
    assert "--name" in result.output
    # workflow text:
    assert "manifest" in result.output.lower() or "register" in result.output.lower()


def test_variant_spec_written_to_graph_json(tmp_path, cli_runner, monkeypatch):
    """BLOCKER-02 fix: port-variant MUST populate graph.json node[variant_spec]
    so the downstream `automil apply <node_id>` operator workflow works.
    Without this, apply hard-fails with "no recorded variant_spec; run
    `automil port-variant` first" even when port-variant just succeeded.
    """
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_archive_spec(adir, "node_0190",
                        overlay={"models/model_clam.py": "sha:..."},
                        parent="clam_mb")
    _write_graph_with_node(adir, "node_0190")
    from automil.cli import main
    result = cli_runner.invoke(main, ["port-variant", "node_0190"])
    assert result.exit_code == 0, result.output

    graph = json.loads((adir / "graph.json").read_text())
    node = graph["nodes"]["node_0190"]
    assert "variant_spec" in node, (
        "port-variant did not write variant_spec into graph.json — "
        "this breaks the apply <node_id> integration. See BLOCKER-02 fix in 01-11."
    )
    spec = node["variant_spec"]
    assert spec["kind"] == "model"
    assert spec["name"] == "clam_mb_v0190"
    assert spec["parent"] == "clam_mb"


def test_variant_spec_for_loss_kind(tmp_path, cli_runner, monkeypatch):
    """variant_spec.parent is None for loss kind — apply must read this correctly."""
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    _write_archive_spec(adir, "node_0195",
                        overlay={"utils/core_utils.py": "sha:..."},
                        parent=None)
    _write_graph_with_node(adir, "node_0195")
    from automil.cli import main
    result = cli_runner.invoke(main, ["port-variant", "node_0195"])
    assert result.exit_code == 0, result.output

    graph = json.loads((adir / "graph.json").read_text())
    spec = graph["nodes"]["node_0195"]["variant_spec"]
    assert spec["kind"] == "loss"
    assert spec["parent"] is None


def test_apply_after_port_variant_no_mock(tmp_path, cli_runner, monkeypatch):
    """Test 19 — lifecycle integration: invoke port-variant FIRST (writes variant
    module + populates node['variant_spec'] in graph.json), THEN invoke apply.

    apply MUST succeed using the real graph.json that port-variant wrote —
    NO manual injection of variant_spec into the test's graph.json at any point.

    This test lives in 01-11 (not 01-09) because it requires BOTH port-variant
    (implemented here) AND apply (Plan 01-09, which is in this plan's depends_on
    chain). Same-wave plans cannot depend on each other — placing this in 01-09
    would have violated the Wave 5 parallel-execution contract.
    """
    adir = _setup(tmp_path)
    monkeypatch.chdir(tmp_path)

    # 1. Set up archive/spec.json so port-variant can derive a variant.
    _write_archive_spec(adir, "node_0501",
                        overlay={"models/model_clam.py": "sha256:abc"},
                        parent="clam_mb")

    # 2. graph.json with the node but NO variant_spec yet — realistic state
    #    right after `automil submit` but before port-variant.
    _write_graph_with_node(adir, "node_0501")

    from automil.cli import main

    # 3. port-variant: writes variant module + manifest + variant_spec to graph.json.
    port_result = cli_runner.invoke(main, ["port-variant", "node_0501"])
    assert port_result.exit_code == 0, (
        f"port-variant must succeed for lifecycle integration test. "
        f"Output: {port_result.output}"
    )

    # 4. Sanity: graph.json now has variant_spec.
    graph = json.loads((adir / "graph.json").read_text())
    assert "variant_spec" in graph["nodes"]["node_0501"], (
        "port-variant did not write variant_spec into graph.json. "
        "The downstream apply will hard-fail. See BLOCKER-02 fix in 01-11."
    )

    # 5. apply reads node['variant_spec'] from the REAL graph.json — must succeed.
    apply_result = cli_runner.invoke(main, ["apply", "node_0501"])
    assert apply_result.exit_code == 0, (
        f"apply must succeed after port-variant has run. "
        f"Output: {apply_result.output}"
    )

    # 6. Verify config.yaml was updated with the variant selection.
    cfg = yaml.safe_load((adir / "config.yaml").read_text())
    assert cfg["model"]["variant"] == "clam_mb_v0501"
    assert cfg["model"]["parent"] == "clam_mb"
