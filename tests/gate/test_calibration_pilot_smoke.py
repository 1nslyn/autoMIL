"""Smoke test for the calibration pilot mechanism (D-151).

This test verifies that `automil promote --calibrate` works end-to-end on
a synthetic graph. The ACTUAL pilot (Leo running node_0176 against fresh
cells) is a checkpoint task documented in docs/calibration-pilot.md.

Coverage:
  Test 1 (test_calibrate_pilot_synthetic_graph_smoke):
        5 held-out cells, mixed composites, --calibrate path:
        - exit_code == 0
        - graph node_0002 status STAYS "candidate"
        - archive/<node_0002>/gate_evaluation.jsonl exists with per-cell + decision
        - parent gate_log (node_0001.gate_log.jsonl) does NOT exist
        - output mentions "calibrate" or "dry-run"

  Test 2 (test_calibration_doc_scaffold_exists):
        docs/calibration-pilot.md scaffold exists and references
        "node_0176", "Recommended K", and the delta/wins column headers.

BCK-04 clean; framework purity D-148.
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
# Held-out cell fixture: 3 CCRCC + 2 CLWD = 5 total
# ---------------------------------------------------------------------------

_HELD_OUT_CELLS = [
    ("cell_ccrcc0000001", "ccrcc", "uni_v2", "high_grade"),
    ("cell_ccrcc0000002", "ccrcc", "ctranspath", "high_grade"),
    ("cell_ccrcc0000003", "ccrcc", "hibou_l", "high_grade"),
    ("cell_clwd0000001", "clwd", "uni_v2", "subtype"),
    ("cell_clwd0000002", "clwd", "hibou_l", "subtype"),
]

# 5 cells, p_threshold=0.2: Bonferroni-corrected alpha = 0.2/5 = 0.04
# Wilcoxon minimum achievable p for n=5 is 0.031 <= 0.04, so this is reachable.
_MANIFEST_K = 5
_MANIFEST_P_THRESHOLD = 0.2
_PARENT_COMPOSITE = 0.80

# Per-cell composites from the spec (D-151 calibration pilot example):
# [0.82, 0.83, 0.78, 0.84, 0.79]
# Deltas vs parent=0.80: [+0.02, +0.03, -0.02, +0.04, -0.01]
# Wins (delta > 0): 3/5 — typical near-threshold mixed signal
_PILOT_COMPOSITES = [0.82, 0.83, 0.78, 0.84, 0.79]
_PILOT_DELTAS = [c - _PARENT_COMPOSITE for c in _PILOT_COMPOSITES]


# ---------------------------------------------------------------------------
# Project fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def project_with_candidate(tmp_path):
    """Minimal automil project with a search-cell parent + candidate node + gate manifest.

    Graph:
      node_0001  (status=keep, composite=0.80, parent=None)  ← search cell parent
      node_0002  (status=candidate, composite=0.85, parent=node_0001)

    Manifest: node_0001 with 5 held-out cells (3 CCRCC + 2 CLWD), K=5, p=0.2.
    """
    adir = tmp_path / "automil"
    adir.mkdir()
    (adir / "config.yaml").write_text("# fixture\n")

    graph_data = {
        "meta": {"total_executed": 2, "next_id": 3},
        "nodes": {
            "node_0001": {
                "id": "node_0001",
                "parent_id": None,
                "type": "executed",
                "status": "keep",
                "composite": _PARENT_COMPOSITE,
                "description": "search-cell parent (known-good basis)",
                "overlay_dir": str(tmp_path / "archive" / "node_0001"),
            },
            "node_0002": {
                "id": "node_0002",
                "parent_id": "node_0001",
                "type": "executed",
                "status": "candidate",
                "composite": 0.85,
                "description": "candidate applying node_0176-equivalent changes",
                "overlay_dir": str(tmp_path / "archive" / "node_0002"),
            },
        },
    }
    (adir / "graph.json").write_text(json.dumps(graph_data))

    manifests_dir = adir / "gate"
    manifests_dir.mkdir()
    manifest = GateManifest(
        parent_id="node_0001",
        created_at="2026-05-05T00:00:00Z",
        git_committed_at_sha="abc1234",
        held_out_cells=tuple(tuple(c) for c in _HELD_OUT_CELLS),  # type: ignore[arg-type]
        K=_MANIFEST_K,
        p_threshold=_MANIFEST_P_THRESHOLD,
        bootstrap_reps=100,  # small for test speed
        win_definition="delta_composite > 0 AND p < p_threshold",
        schema_version="gate-v1",
    )
    write_manifest(manifest, manifests_dir)

    archive_dir = adir / "archive"
    archive_dir.mkdir()

    return tmp_path


# ---------------------------------------------------------------------------
# Monkeypatch helpers (mirrors test_cli_promote.py approach)
# ---------------------------------------------------------------------------

class _NullBackend:
    """Minimal backend stub — never called when evaluate_candidate is patched."""
    pass


def _get_gate_promote_module():
    """Return automil.gate.promote MODULE (not the re-exported function).

    sys.modules path necessary because __init__.py re-exports the function,
    shadowing the module name in normal import resolution.
    """
    if "automil.gate.promote" not in sys.modules:
        importlib.import_module("automil.gate.promote")
    return sys.modules["automil.gate.promote"]


def _make_fake_evaluate(deltas: list[float]):
    """Return a fake evaluate_candidate returning the given per-cell deltas + no skipped."""
    def fake_evaluate(candidate_node_id, manifest, backend, graph, **kwargs):
        per_cell = [
            {
                "cell_id": _HELD_OUT_CELLS[i][0],
                "dataset": _HELD_OUT_CELLS[i][1],
                "encoder": _HELD_OUT_CELLS[i][2],
                "task": _HELD_OUT_CELLS[i][3],
                "child_node_id": f"child_{i:04d}",
                "candidate_composite": _PARENT_COMPOSITE + d,
                "parent_composite": _PARENT_COMPOSITE,
                "delta": d,
                "status": "completed",
            }
            for i, d in enumerate(deltas)
        ]
        return per_cell, []  # no skipped cells
    return fake_evaluate


def _run_promote(project_dir, args: list[str], monkeypatch, deltas: list[float] | None = None):
    """Invoke `automil promote ...` with cwd=project_dir.

    Always patches _resolve_backend. If deltas is given, patches evaluate_candidate.
    """
    from click.testing import CliRunner
    from automil.cli import main

    old_cwd = os.getcwd()
    try:
        os.chdir(str(project_dir))
        monkeypatch.setattr(
            "automil.cli.promote._resolve_backend",
            lambda _name, _adir: _NullBackend(),
        )
        if deltas is not None:
            monkeypatch.setattr(
                _get_gate_promote_module(),
                "evaluate_candidate",
                _make_fake_evaluate(deltas),
            )
        result = CliRunner().invoke(main, ["promote"] + args, catch_exceptions=False)
    finally:
        os.chdir(old_cwd)
    return result


# ---------------------------------------------------------------------------
# Test 1: end-to-end --calibrate smoke test on a synthetic 5-cell graph
# ---------------------------------------------------------------------------

def test_calibrate_pilot_synthetic_graph_smoke(project_with_candidate, monkeypatch):
    """D-151 calibration pilot smoke test.

    Scenario: 3 CCRCC + 2 CLWD held-out cells; mixed deltas (+0.02, +0.03, -0.02,
    +0.04, -0.01); 3/5 wins — typical near-threshold case.

    Assertions:
    1. exit_code == 0
    2. node_0002 status STAYS "candidate" (no status mutation in calibrate mode)
    3. archive/node_0002/gate_evaluation.jsonl exists; first record has per_cell_results
       with all 5 cells; second record is decision summary with result field
    4. automil/gate/node_0001.gate_log.jsonl does NOT exist (calibrate bypasses parent log)
    5. CLI output mentions "calibrate" or "dry"
    """
    project = project_with_candidate
    adir = project / "automil"

    result = _run_promote(
        project,
        ["node_0002", "--calibrate"],
        monkeypatch=monkeypatch,
        deltas=_PILOT_DELTAS,
    )

    # 1. Exit code 0
    assert result.exit_code == 0, (
        f"Expected exit 0 with --calibrate; got {result.exit_code}.\nOutput: {result.output}"
    )

    # 2. Status unchanged
    graph_data = json.loads((adir / "graph.json").read_text())
    status = graph_data["nodes"]["node_0002"]["status"]
    assert status == "candidate", (
        f"--calibrate must not mutate status; expected 'candidate', got {status!r}"
    )

    # 3. Archive log with per-cell + decision records
    log_path = adir / "archive" / "node_0002" / "gate_evaluation.jsonl"
    assert log_path.exists(), (
        f"archive gate_evaluation.jsonl must be written even in --calibrate mode; "
        f"expected at {log_path}"
    )
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) >= 2, (
        f"Expected at least 2 JSONL lines (per_cell_results + decision); got {len(lines)}"
    )
    first_record = json.loads(lines[0])
    assert "per_cell_results" in first_record, (
        f"First JSONL record must contain 'per_cell_results'; got keys: {list(first_record.keys())}"
    )
    per_cell_list = first_record["per_cell_results"]
    assert len(per_cell_list) == 5, (
        f"Expected 5 per-cell results (3 CCRCC + 2 CLWD); got {len(per_cell_list)}"
    )
    # Verify all 5 cell IDs are present
    found_cell_ids = {r["cell_id"] for r in per_cell_list}
    expected_cell_ids = {c[0] for c in _HELD_OUT_CELLS}
    assert found_cell_ids == expected_cell_ids, (
        f"Per-cell results must cover all 5 held-out cells.\n"
        f"Found: {found_cell_ids}\nExpected: {expected_cell_ids}"
    )
    decision_record = json.loads(lines[-1])
    assert "result" in decision_record, (
        f"Decision record must contain 'result'; got keys: {list(decision_record.keys())}"
    )
    assert decision_record.get("event") == "decision" or "result" in decision_record, (
        "Last JSONL record must be the gate decision summary"
    )

    # 4. Parent gate_log must NOT exist
    parent_log = adir / "gate" / "node_0001.gate_log.jsonl"
    assert not parent_log.exists(), (
        f"--calibrate must NOT write parent gate_log; found {parent_log}"
    )

    # 5. Output mentions calibrate or dry-run
    out_lower = result.output.lower()
    assert "calibrate" in out_lower or "dry" in out_lower, (
        f"Expected 'calibrate' or 'dry' in CLI output; got: {result.output!r}"
    )


# ---------------------------------------------------------------------------
# Test 2: calibration scaffold document structure check
# ---------------------------------------------------------------------------

def test_calibration_doc_scaffold_exists():
    """Calibration scaffold doc at docs/calibration-pilot.md must exist and
    reference node_0176, Recommended K, and the delta/wins column headers.

    This test guards regression: if the scaffold is accidentally deleted or
    restructured, the pilot recipe is lost.
    """
    # Resolve from repo root (tests run from repo root by pyproject.toml config)
    scaffold = Path("docs/calibration-pilot.md")
    assert scaffold.exists(), (
        "Calibration scaffold doc missing — it carries the operator recipe "
        "for the deferred pilot (Leo fills in real per-cell numbers). "
        f"Expected path: {scaffold.resolve()}"
    )
    text = scaffold.read_text()
    assert "node_0176" in text, (
        "Scaffold must reference the known-good change (node_0176); "
        "check docs/calibration-pilot.md"
    )
    assert "Recommended K" in text, (
        "Scaffold must include a 'Recommended K' section for Leo to fill; "
        "check docs/calibration-pilot.md"
    )
    text_lower = text.lower()
    assert "delta" in text_lower, (
        "Scaffold must define a delta-matrix table (column 'delta'); "
        "check docs/calibration-pilot.md"
    )
    assert "wins" in text_lower, (
        "Scaffold must define a delta-matrix table (column 'wins'); "
        "check docs/calibration-pilot.md"
    )
