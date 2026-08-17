"""Tests for automil promote top-level CLI command (plan 05-09 / GTE-04 / D-141, D-145, D-151).

6 behaviour tests covering:
  T-7  pass_path              : positive deltas -> status='registered', output contains PASS/registered
  T-8  fail_path              : negative deltas -> exit 0, status='keep', output contains fail/keep
  T-9  calibrate_mode         : --calibrate -> status stays 'candidate', output mentions calibrate/dry-run
  T-10 non_candidate_exits    : keep node (not nominated) -> non-zero exit with 'nominate first' hint
  T-11 unknown_node_exits     : unknown id -> non-zero exit + 'not found'
  T-12 no_manifest_exits      : candidate's parent has no manifest -> non-zero exit + 'manifest' hint

Strategy (option b from plan 05-09):
  - monkeypatch automil.cli.promote._resolve_backend to return a NullBackend
  - monkeypatch automil.gate.promote.evaluate_candidate to return controlled deltas
  This avoids all filesystem/daemon setup and deterministically drives gate outcomes.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

from automil.gate.manifest import GateManifest, write_manifest


# ---------------------------------------------------------------------------
# Shared project fixture
# ---------------------------------------------------------------------------

_HELD_OUT_CELLS = [
    ("cell_aa0000000001", "ccrcc", "uni_v2", "high_grade"),
    ("cell_bb0000000002", "clwd", "hibou_l", "subtype"),
    ("cell_cc0000000003", "ccrcc", "ctranspath", "high_grade"),
    ("cell_dd0000000004", "clwd", "uni_v2", "subtype"),
    ("cell_ee0000000005", "ccrcc", "uni_v2", "subtype"),
]
# 5 cells with p_threshold=0.2: Bonferroni-corrected alpha=0.04 > Wilcoxon min-p (0.031 for n=5)
# Matches test_promote.py's proven setup for a statistically achievable pass path.
_MANIFEST_K = 5
_MANIFEST_P_THRESHOLD = 0.2

_PARENT_COMPOSITE = 0.80
_CANDIDATE_COMPOSITE = 0.85


@pytest.fixture
def project_with_candidate(tmp_path):
    """Minimal automil project with a candidate node and a gate manifest.

    graph structure: node_0001 (keep, parent) -> node_0002 (candidate)
    manifest: registered for node_0001 with 2 held-out cells, K=2
    """
    adir = tmp_path / "automil"
    adir.mkdir()
    (adir / "config.yaml").write_text("# fixture\n")

    # Graph with parent + candidate
    graph_data = {
        "meta": {"total_executed": 2, "next_id": 3},
        "nodes": {
            "node_0001": {
                "id": "node_0001",
                "parent_id": None,
                "type": "executed",
                "status": "keep",
                "primary_value": _PARENT_COMPOSITE,
                "description": "parent node",
                "overlay_dir": str(tmp_path / "archive" / "node_0001"),
            },
            "node_0002": {
                "id": "node_0002",
                "parent_id": "node_0001",
                "type": "executed",
                "status": "candidate",
                "primary_value": _CANDIDATE_COMPOSITE,
                "description": "candidate node",
                "overlay_dir": str(tmp_path / "archive" / "node_0002"),
            },
        },
    }
    (adir / "graph.json").write_text(json.dumps(graph_data))

    # Gate manifest for parent node_0001
    manifests_dir = adir / "gate"
    manifests_dir.mkdir()
    manifest = GateManifest(
        parent_id="node_0001",
        created_at="2026-05-05T00:00:00Z",
        git_committed_at_sha="abc1234",
        held_out_cells=tuple(tuple(c) for c in _HELD_OUT_CELLS),  # type: ignore[arg-type]
        K=_MANIFEST_K,
        p_threshold=_MANIFEST_P_THRESHOLD,
        bootstrap_reps=100,  # small for speed in tests
        win_definition="delta_primary_value > 0 AND p < p_threshold",
        schema_version="gate-v1",
    )
    write_manifest(manifest, manifests_dir)

    # Archive dir for gate eval outputs
    archive_dir = adir / "archive"
    archive_dir.mkdir()

    return tmp_path


# ---------------------------------------------------------------------------
# Monkeypatch helpers
# ---------------------------------------------------------------------------

class _NullBackend:
    """Minimal backend stub — never called when evaluate_candidate is patched."""
    pass


def _get_promote_module():
    """Return the automil.gate.promote MODULE (not the function re-exported by __init__).

    Must go through sys.modules because the __init__ re-export shadows the module name.
    """
    if "automil.gate.promote" not in sys.modules:
        importlib.import_module("automil.gate.promote")
    return sys.modules["automil.gate.promote"]


def _make_fake_evaluate(deltas: list[float], skipped: list[str] | None = None):
    """Return a fake evaluate_candidate that yields deterministic per-cell deltas."""
    def fake_evaluate(candidate_node_id, manifest, backend, graph, **kwargs):
        per_cell = [
            {
                "cell_id": f"cell_{i:04d}",
                "dataset": "ccrcc",
                "encoder": "uni_v2",
                "task": "high_grade",
                "child_node_id": f"child_{i:04d}",
                "candidate_primary_value": _PARENT_COMPOSITE + d,
                "parent_primary_value": _PARENT_COMPOSITE,
                "delta": d,
                "status": "completed",
            }
            for i, d in enumerate(deltas)
        ]
        return per_cell, (skipped or [])
    return fake_evaluate


def _run_promote(project_dir, args: list[str], monkeypatch, deltas: list[float] | None = None):
    """Invoke `automil promote ...` with cwd set to project_dir.

    Always patches _resolve_backend (avoids real daemon).
    If deltas is provided, patches evaluate_candidate too.
    """
    from click.testing import CliRunner
    from automil.cli import main

    old_cwd = os.getcwd()
    try:
        os.chdir(str(project_dir))
        monkeypatch.setattr("automil.cli.promote._resolve_backend", lambda _name, _adir: _NullBackend())
        if deltas is not None:
            monkeypatch.setattr(_get_promote_module(), "evaluate_candidate", _make_fake_evaluate(deltas))
        result = CliRunner().invoke(main, ["promote"] + args, catch_exceptions=False)
    finally:
        os.chdir(old_cwd)
    return result


# ---------------------------------------------------------------------------
# T-7: Pass path — positive deltas -> status='registered'
# ---------------------------------------------------------------------------

def test_promote_cli_pass_path(project_with_candidate, monkeypatch):
    """Positive deltas: exit 0; status='registered'; output mentions pass/registered."""
    project = project_with_candidate

    # 5 strongly-positive deltas; Wilcoxon p~=0.031, Bonferroni-corrected alpha=0.04
    # (p_threshold=0.2 / K=5). Proven in test_promote.py to produce a reliable pass.
    result = _run_promote(project, ["node_0002"], monkeypatch=monkeypatch,
                          deltas=[0.05, 0.04, 0.06, 0.03, 0.05])

    assert result.exit_code == 0, (
        f"Expected exit 0 on pass path; got {result.exit_code}. Output: {result.output}"
    )
    graph_data = json.loads((project / "automil" / "graph.json").read_text())
    status = graph_data["nodes"]["node_0002"]["status"]
    assert status == "registered", (
        f"Expected status='registered' after gate pass; got {status!r}"
    )
    out_lower = result.output.lower()
    assert "registered" in out_lower or "pass" in out_lower, (
        f"Expected 'registered' or 'PASS' in output; got: {result.output}"
    )


# ---------------------------------------------------------------------------
# T-8: Fail path — negative deltas -> exit 0, status='keep'
# ---------------------------------------------------------------------------

def test_promote_cli_fail_path(project_with_candidate, monkeypatch):
    """Negative deltas: exit 0 (gate fail is not CLI failure); status='keep'."""
    project = project_with_candidate

    # 5 strongly-negative deltas — gate clearly fails
    result = _run_promote(project, ["node_0002"], monkeypatch=monkeypatch,
                          deltas=[-0.05, -0.04, -0.06, -0.03, -0.05])

    assert result.exit_code == 0, (
        f"Expected exit 0 even on gate fail; got {result.exit_code}. Output: {result.output}"
    )
    graph_data = json.loads((project / "automil" / "graph.json").read_text())
    status = graph_data["nodes"]["node_0002"]["status"]
    assert status == "keep", (
        f"Expected status='keep' after gate fail; got {status!r}"
    )
    out_lower = result.output.lower()
    assert "fail" in out_lower or "keep" in out_lower, (
        f"Expected 'fail' or 'keep' in output on fail path; got: {result.output}"
    )


# ---------------------------------------------------------------------------
# T-9: Calibrate mode — status stays 'candidate', output hints dry-run
# ---------------------------------------------------------------------------

def test_promote_cli_calibrate_mode(project_with_candidate, monkeypatch):
    """--calibrate: status unchanged (stays 'candidate'); output has calibrate/dry-run hint."""
    project = project_with_candidate

    # 5 positive deltas so the would-PASS branch is exercised
    result = _run_promote(project, ["node_0002", "--calibrate"], monkeypatch=monkeypatch,
                          deltas=[0.05, 0.04, 0.06, 0.03, 0.05])

    assert result.exit_code == 0, (
        f"Expected exit 0 with --calibrate; got {result.exit_code}. Output: {result.output}"
    )
    graph_data = json.loads((project / "automil" / "graph.json").read_text())
    status = graph_data["nodes"]["node_0002"]["status"]
    assert status == "candidate", (
        f"--calibrate must not change status; expected 'candidate', got {status!r}"
    )
    out_lower = result.output.lower()
    assert "calibrate" in out_lower or "dry" in out_lower or "unchanged" in out_lower, (
        f"Expected calibrate/dry-run indicator in output; got: {result.output}"
    )
    # Parent gate_log must NOT be written
    gate_log = project / "automil" / "gate" / "node_0001.gate_log.jsonl"
    assert not gate_log.exists(), (
        f"--calibrate must not write parent gate_log; found {gate_log}"
    )


# ---------------------------------------------------------------------------
# T-10: Non-candidate (keep) node exits non-zero with 'nominate first' hint
# ---------------------------------------------------------------------------

def test_promote_cli_non_candidate_exits_nonzero(project_with_candidate, monkeypatch):
    """status=keep node exits non-zero; output contains 'candidate' + 'nominate first'."""
    project = project_with_candidate

    # Revert candidate back to keep (forgot to nominate)
    graph_path = project / "automil" / "graph.json"
    graph_data = json.loads(graph_path.read_text())
    graph_data["nodes"]["node_0002"]["status"] = "keep"
    graph_path.write_text(json.dumps(graph_data))

    # No deltas needed — error fires before evaluate_candidate is called
    result = _run_promote(project, ["node_0002"], monkeypatch=monkeypatch)

    assert result.exit_code != 0, (
        f"Expected non-zero exit for non-candidate node; got {result.exit_code}. "
        f"Output: {result.output}"
    )
    out_lower = result.output.lower()
    assert "candidate" in out_lower, (
        f"Expected 'candidate' in error message; got: {result.output}"
    )
    assert "nominate" in out_lower, (
        f"Expected 'nominate' hint in error message; got: {result.output}"
    )


# ---------------------------------------------------------------------------
# T-11: Unknown node exits non-zero + 'not found'
# ---------------------------------------------------------------------------

def test_promote_cli_unknown_node_exits_nonzero(project_with_candidate, monkeypatch):
    """Unknown candidate_id exits non-zero; output contains 'not found'."""
    project = project_with_candidate

    result = _run_promote(project, ["node_9999"], monkeypatch=monkeypatch)

    assert result.exit_code != 0, (
        f"Expected non-zero exit for unknown node; got {result.exit_code}. "
        f"Output: {result.output}"
    )
    assert "not found" in result.output.lower(), (
        f"Expected 'not found' in output; got: {result.output}"
    )


# ---------------------------------------------------------------------------
# T-12: No manifest exits non-zero with 'manifest' + parent_id hint
# ---------------------------------------------------------------------------

def test_promote_cli_no_manifest_exits_nonzero(project_with_candidate, monkeypatch):
    """Candidate's parent has no manifest; non-zero exit; output contains 'manifest' + parent_id."""
    project = project_with_candidate

    # Remove the manifest for node_0001
    manifest_path = project / "automil" / "gate" / "node_0001.gate_manifest.json"
    manifest_path.unlink()

    result = _run_promote(project, ["node_0002"], monkeypatch=monkeypatch)

    assert result.exit_code != 0, (
        f"Expected non-zero exit when manifest absent; got {result.exit_code}. "
        f"Output: {result.output}"
    )
    out_lower = result.output.lower()
    assert "manifest" in out_lower, (
        f"Expected 'manifest' in error message; got: {result.output}"
    )
    assert "node_0001" in result.output, (
        f"Expected parent_id 'node_0001' in error message; got: {result.output}"
    )
