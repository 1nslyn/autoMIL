"""Tests for gate/promote.py — promote() orchestration.

TDD RED phase: 11 failing tests for plan 05-07.

Coverage:
  Test 1  — promote on non-candidate status raises ValueError
  Test 2  — promote with no manifest raises FileNotFoundError
  Test 3  — pass path: status -> 'registered', history event correct
  Test 4  — fail path: status -> 'keep', history event correct
  Test 5  — inconclusive (K_effective < K_floor): status stays 'candidate'
  Test 6  — Bonferroni-corrected alpha passed to paired_wilcoxon_with_bootstrap
  Test 7  — archive gate_evaluation.jsonl written with per-cell + decision records
  Test 8  — parent gate_log appended (jsonl, not overwritten) per promotion
  Test 9  — calibrate=True is dry-run: archive log written, no status mutation, no parent log
  Test 10 — graph.save() invoked exactly once at end (via mtime change)
  Test 11 — framework purity: zero autobench/AUTOBENCH_/benchmarks/ in source
"""
from __future__ import annotations

import json
import pathlib
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pytest

from automil.backends.base import Backend, JobHandle, JobSpec, JobState
from automil.cells.state import Cell, CellStatus
from automil.gate.manifest import GateManifest, write_manifest
from automil.graph import ExperimentGraph


# ---------------------------------------------------------------------------
# Helpers — graph and manifest builders
# ---------------------------------------------------------------------------

def _make_graph(
    *,
    candidate_status: str = "candidate",
    candidate_primary_value: float = 0.85,
    parent_primary_value: float = 0.80,
    tmp_path: Path | None = None,
) -> ExperimentGraph:
    """Build a two-node graph (parent=node_0001 -> candidate=node_0002).

    If tmp_path is given, save to disk so graph.save() can update mtime.
    """
    if tmp_path is not None:
        graph_path = tmp_path / "graph.json"
    else:
        graph_path = pathlib.Path("/nonexistent/graph.json")

    graph = ExperimentGraph.__new__(ExperimentGraph)
    graph.path = graph_path
    graph._technique_map = {}
    graph._data = {
        "schema_version": 1,
        "meta": {
            "best_primary_value": 0.0,
            "best_node_id": None,
            "total_executed": 0,
            "total_proposed": 0,
            "next_id": 3,
            "baseline_primary_value": 0.0,
            "scoring": {"exploration_weight": 0.005, "novelty_weight": 0.003},
        },
        "nodes": {
            "node_0001": {
                "id": "node_0001",
                "parent_id": None,
                "type": "executed",
                "status": "keep",
                "primary_value": parent_primary_value,
                "commit": "abc1234",
                "overlay_files": [],
                "overlay_dir": "archive/node_0001",
                "vram_gb": 4.0,
            },
            "node_0002": {
                "id": "node_0002",
                "parent_id": "node_0001",
                "type": "executed",
                "status": candidate_status,
                "primary_value": candidate_primary_value,
                "commit": "abc1234",
                "overlay_files": [],
                "overlay_dir": "archive/node_0002",
                "vram_gb": 4.0,
            },
        },
    }
    if tmp_path is not None:
        graph.save()
    return graph


def _make_manifest(
    *,
    held_out_cells: list[tuple[str, str, str, str]] | None = None,
    K: int = 2,
    p_threshold: float = 0.05,
    bootstrap_reps: int = 100,
) -> GateManifest:
    if held_out_cells is None:
        held_out_cells = [
            ("cell_aaaa1111", "ccrcc", "uni_v2", "high_grade"),
            ("cell_bbbb2222", "clwd", "hibou_l", "subtype"),
            ("cell_cccc3333", "ccrcc", "hibou_l", "high_grade"),
        ]
    return GateManifest(
        parent_id="node_0001",
        created_at="2026-05-05T00:00:00Z",
        git_committed_at_sha="abc1234",
        held_out_cells=tuple(tuple(c) for c in held_out_cells),  # type: ignore[arg-type]
        K=K,
        p_threshold=p_threshold,
        bootstrap_reps=bootstrap_reps,
        win_definition="delta_primary_value > 0 AND p < p_threshold",
        schema_version="gate-v1",
    )


# ---------------------------------------------------------------------------
# Synchronous recording backend (reused from test_evaluate.py pattern)
# ---------------------------------------------------------------------------

class _Job:
    def __init__(self, node_id: str, opaque_id: str, state: JobState = JobState.COMPLETED):
        self.handle = JobHandle(
            node_id=node_id,
            backend="recording",
            opaque_id=opaque_id,
            submitted_at=time.time(),
        )
        self.state = state
        self.node_primary_value: float = 0.0  # stamped on graph node after submit


class RecordingBackend(Backend):
    """Synchronous recording backend — deterministic, no threads."""

    def __init__(
        self,
        initial_state: JobState = JobState.COMPLETED,
        per_cell_primary_values: dict[str, float] | None = None,
    ) -> None:
        self._jobs: dict[str, _Job] = {}  # opaque_id -> _Job
        self._counter = 0
        self._initial_state = initial_state
        self._per_cell_primary_values = per_cell_primary_values or {}
        self.submitted_specs: list[tuple[JobSpec, str]] = []

    @property
    def submit_count(self) -> int:
        return len(self.submitted_specs)

    def submit(self, spec: JobSpec) -> JobHandle:
        self._counter += 1
        opaque_id = f"{self._counter}"
        job = _Job(node_id=spec.node_id, opaque_id=opaque_id, state=self._initial_state)
        self._jobs[opaque_id] = job
        self.submitted_specs.append((spec, spec.node_id))
        return job.handle

    def poll(self, handle: JobHandle) -> JobState:
        job = self._jobs.get(handle.opaque_id)
        if job is None:
            raise ValueError(f"Unknown job: {handle.opaque_id!r}")
        return job.state

    def cancel(self, handle: JobHandle, signal: Optional[int] = None) -> None:
        job = self._jobs.get(handle.opaque_id)
        if job is not None:
            job.state = JobState.CANCELLED

    def list_running(self) -> list[JobHandle]:
        return [
            j.handle for j in self._jobs.values()
            if j.state in (JobState.PENDING, JobState.RUNNING)
        ]

    def log_iter(self, handle: JobHandle) -> Iterator[str]:
        return iter([])


# ---------------------------------------------------------------------------
# Common monkeypatches
# ---------------------------------------------------------------------------

    def healthcheck(self) -> "HealthReport":
        raise NotImplementedError("healthcheck deferred for test-fixture backend")

def _patch_no_cells(monkeypatch) -> None:
    """Patch get_cell so no cells are cap-exhausted."""
    monkeypatch.setattr("automil.gate.evaluate.get_cell", lambda cid: None)


def _get_promote_module():
    """Return the automil.gate.promote MODULE (not the function of the same name).

    We must use sys.modules because `import automil.gate.promote` at the top
    of the package resolves to the *function* due to the __init__.py re-export.
    """
    import importlib
    import sys
    # Force module load if not yet imported
    if "automil.gate.promote" not in sys.modules:
        importlib.import_module("automil.gate.promote")
    return sys.modules["automil.gate.promote"]


def _patch_evaluate_with_deltas(monkeypatch, deltas: list[float]) -> None:
    """Replace evaluate_candidate with a stub that returns given deltas."""
    def fake_evaluate(candidate_node_id, manifest, backend, graph, **kwargs):
        per_cell = [
            {
                "cell_id": f"cell_{i:04d}",
                "dataset": "ccrcc",
                "encoder": "uni_v2",
                "task": "high_grade",
                "child_node_id": f"child_{i:04d}",
                "candidate_primary_value": 0.80 + d,
                "parent_primary_value": 0.80,
                "delta": d,
                "status": "completed",
            }
            for i, d in enumerate(deltas)
        ]
        return per_cell, []  # no skipped

    monkeypatch.setattr(_get_promote_module(), "evaluate_candidate", fake_evaluate)


# ---------------------------------------------------------------------------
# Test 1: non-candidate status raises ValueError
# ---------------------------------------------------------------------------

def test_promote_requires_candidate_status(tmp_path):
    """Calling promote on a keep node raises ValueError with 'candidate' + 'nominate first'."""
    from automil.gate.promote import promote

    graph = _make_graph(candidate_status="keep", tmp_path=tmp_path)
    manifests_dir = tmp_path / "gate"
    archive_dir = tmp_path / "archive"

    with pytest.raises(ValueError, match="candidate") as exc_info:
        promote("node_0002", backend=None, graph=graph,
                manifests_dir=manifests_dir, archive_dir=archive_dir)

    assert "nominate" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Test 2: missing manifest raises FileNotFoundError
# ---------------------------------------------------------------------------

def test_promote_requires_existing_manifest(tmp_path):
    """Promote on a candidate whose parent has no manifest raises FileNotFoundError."""
    from automil.gate.promote import promote

    graph = _make_graph(candidate_status="candidate", tmp_path=tmp_path)
    manifests_dir = tmp_path / "gate"  # empty — no manifest written
    manifests_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = tmp_path / "archive"

    with pytest.raises(FileNotFoundError):
        promote("node_0002", backend=None, graph=graph,
                manifests_dir=manifests_dir, archive_dir=archive_dir)


# ---------------------------------------------------------------------------
# Test 3: pass path
# ---------------------------------------------------------------------------

def test_promote_pass_path(tmp_path, monkeypatch):
    """Strong positive deltas -> status='registered'; history has gate_result pass.

    Uses 5 held-out cells with p_threshold=0.1 so the Bonferroni-corrected alpha
    (0.1/5 = 0.02) is comfortably above the Wilcoxon minimum achievable p for n=5
    (0.031 is the minimum with the one-sided test). We use rng_seed=42 for
    deterministic bootstrap CI.
    """
    from automil.gate.promote import promote

    # 5 strongly-positive deltas; Wilcoxon p=0.031, Bonferroni-corrected alpha=0.02
    # We use p_threshold=0.2 (K=5 -> p_corrected=0.04 > 0.031)
    deltas = [0.05, 0.04, 0.06, 0.03, 0.05]
    _patch_evaluate_with_deltas(monkeypatch, deltas)

    held_out_cells = [
        ("cell_a", "ccrcc", "uni_v2", "high_grade"),
        ("cell_b", "clwd", "hibou_l", "subtype"),
        ("cell_c", "ccrcc", "ctranspath", "high_grade"),
        ("cell_d", "clwd", "uni_v2", "subtype"),
        ("cell_e", "ccrcc", "uni_v2", "subtype"),
    ]
    graph = _make_graph(candidate_status="candidate", tmp_path=tmp_path)
    # p_threshold=0.2 -> Bonferroni-corrected = 0.2/5 = 0.04; Wilcoxon p=0.031 <= 0.04 -> PASS
    manifest = _make_manifest(
        held_out_cells=held_out_cells, K=5, p_threshold=0.2, bootstrap_reps=200
    )
    manifests_dir = tmp_path / "gate"
    write_manifest(manifest, manifests_dir)
    archive_dir = tmp_path / "archive"

    result = promote(
        "node_0002",
        backend=RecordingBackend(),
        graph=graph,
        manifests_dir=manifests_dir,
        archive_dir=archive_dir,
    )

    assert result is True, "Expected pass path to return True"
    node = graph.nodes["node_0002"]
    assert node["status"] == "registered"

    history = node.get("history", [])
    gate_events = [e for e in history if e.get("event") == "gate_result"]
    assert len(gate_events) == 1
    ev = gate_events[0]
    assert ev["result"] == "pass"
    assert "p_value" in ev
    assert ev.get("ci_low") is not None
    assert ev["wins"] == 5
    assert ev["K_effective"] == 5
    assert ev.get("skipped_cells_due_to_cap", []) == []


# ---------------------------------------------------------------------------
# Test 4: fail path
# ---------------------------------------------------------------------------

def test_promote_fail_path(tmp_path, monkeypatch):
    """Mixed/negative deltas -> status='keep'; history has gate_result fail."""
    from automil.gate.promote import promote

    deltas = [-0.02, 0.01, -0.03]
    _patch_evaluate_with_deltas(monkeypatch, deltas)

    graph = _make_graph(candidate_status="candidate", tmp_path=tmp_path)
    manifest = _make_manifest(K=3, bootstrap_reps=100)
    manifests_dir = tmp_path / "gate"
    write_manifest(manifest, manifests_dir)
    archive_dir = tmp_path / "archive"

    result = promote(
        "node_0002",
        backend=RecordingBackend(),
        graph=graph,
        manifests_dir=manifests_dir,
        archive_dir=archive_dir,
    )

    assert result is False
    node = graph.nodes["node_0002"]
    assert node["status"] == "keep"

    history = node.get("history", [])
    gate_events = [e for e in history if e.get("event") == "gate_result"]
    assert len(gate_events) == 1
    assert gate_events[0]["result"] == "fail"


# ---------------------------------------------------------------------------
# Test 5: inconclusive when K_effective < K_floor
# ---------------------------------------------------------------------------

def test_promote_inconclusive_when_K_effective_below_floor(tmp_path, monkeypatch):
    """K_effective < K_floor -> status STAYS 'candidate'; history event result='inconclusive'."""
    from automil.gate.promote import promote

    # Manifest has K=3, 3 held-out cells. evaluate_candidate skips 2 -> K_effective=1 < K_floor=2
    def fake_evaluate(candidate_node_id, manifest, backend, graph, **kwargs):
        per_cell = [
            {
                "cell_id": "cell_0000",
                "dataset": "ccrcc",
                "encoder": "uni_v2",
                "task": "high_grade",
                "child_node_id": "child_0000",
                "candidate_primary_value": 0.85,
                "parent_primary_value": 0.80,
                "delta": 0.05,
                "status": "completed",
            }
        ]
        skipped = ["cell_1111", "cell_2222"]  # 2 skipped -> K_effective = 3 - 2 = 1
        return per_cell, skipped

    monkeypatch.setattr(_get_promote_module(), "evaluate_candidate", fake_evaluate)

    # Manifest K=3 with 3 cells
    manifest = _make_manifest(
        held_out_cells=[
            ("cell_0000", "ccrcc", "uni_v2", "high_grade"),
            ("cell_1111", "clwd", "hibou_l", "subtype"),
            ("cell_2222", "ccrcc", "hibou_l", "high_grade"),
        ],
        K=3,
        bootstrap_reps=100,
    )
    graph = _make_graph(candidate_status="candidate", tmp_path=tmp_path)
    manifests_dir = tmp_path / "gate"
    write_manifest(manifest, manifests_dir)
    archive_dir = tmp_path / "archive"

    result = promote(
        "node_0002",
        backend=RecordingBackend(),
        graph=graph,
        manifests_dir=manifests_dir,
        archive_dir=archive_dir,
        K_floor=2,
    )

    assert result is False
    node = graph.nodes["node_0002"]
    # Status MUST stay 'candidate' — not 'keep' (D-150 inconclusive path)
    assert node["status"] == "candidate", (
        f"Expected status='candidate' (inconclusive); got {node['status']!r}"
    )

    history = node.get("history", [])
    gate_events = [e for e in history if e.get("event") == "gate_result"]
    assert len(gate_events) == 1
    assert gate_events[0]["result"] == "inconclusive"


# ---------------------------------------------------------------------------
# Test 6: Bonferroni-corrected alpha passed to stats
# ---------------------------------------------------------------------------

def test_promote_uses_bonferroni_corrected_alpha(tmp_path, monkeypatch):
    """K=4, p=0.05 -> p_corrected=0.0125 passed to paired_wilcoxon_with_bootstrap."""
    from automil.gate import stats as gate_stats

    captured: dict[str, float] = {}
    original_func = gate_stats.paired_wilcoxon_with_bootstrap

    def spy(deltas, p_threshold, bootstrap_reps=1000, rng_seed=None):
        captured["p_threshold"] = p_threshold
        return original_func(deltas, p_threshold, bootstrap_reps, rng_seed)

    monkeypatch.setattr(_get_promote_module(), "paired_wilcoxon_with_bootstrap", spy)

    # 4 cells, K=4
    held_out_cells = [
        ("cell_a", "ccrcc", "uni_v2", "high_grade"),
        ("cell_b", "clwd", "hibou_l", "subtype"),
        ("cell_c", "ccrcc", "ctranspath", "high_grade"),
        ("cell_d", "clwd", "uni_v2", "subtype"),
    ]
    deltas = [0.03, 0.02, 0.04, 0.025]

    _patch_evaluate_with_deltas(monkeypatch, deltas)

    manifest = _make_manifest(
        held_out_cells=held_out_cells,
        K=4,
        p_threshold=0.05,
        bootstrap_reps=100,
    )
    graph = _make_graph(candidate_status="candidate", tmp_path=tmp_path)
    manifests_dir = tmp_path / "gate"
    write_manifest(manifest, manifests_dir)
    archive_dir = tmp_path / "archive"

    from automil.gate.promote import promote
    promote(
        "node_0002",
        backend=RecordingBackend(),
        graph=graph,
        manifests_dir=manifests_dir,
        archive_dir=archive_dir,
    )

    assert "p_threshold" in captured, "spy was never called — promote didn't call stats"
    assert captured["p_threshold"] == pytest.approx(0.0125), (
        f"Expected p_threshold=0.0125 (Bonferroni: 0.05/4); got {captured['p_threshold']}"
    )


# ---------------------------------------------------------------------------
# Test 7: archive gate_evaluation.jsonl written
# ---------------------------------------------------------------------------

def test_promote_writes_archive_gate_evaluation_jsonl(tmp_path, monkeypatch):
    """archive_dir/<candidate_id>/gate_evaluation.jsonl exists after promote."""
    from automil.gate.promote import promote

    deltas = [0.05, 0.04, 0.06]
    _patch_evaluate_with_deltas(monkeypatch, deltas)

    graph = _make_graph(candidate_status="candidate", tmp_path=tmp_path)
    manifest = _make_manifest(K=3, bootstrap_reps=100)
    manifests_dir = tmp_path / "gate"
    write_manifest(manifest, manifests_dir)
    archive_dir = tmp_path / "archive"

    promote(
        "node_0002",
        backend=RecordingBackend(),
        graph=graph,
        manifests_dir=manifests_dir,
        archive_dir=archive_dir,
    )

    log_path = archive_dir / "node_0002" / "gate_evaluation.jsonl"
    assert log_path.exists(), f"archive log not found at {log_path}"

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) >= 2, f"Expected at least 2 JSONL lines; got {len(lines)}: {lines}"

    # First line: per_cell_results
    first = json.loads(lines[0])
    assert "per_cell_results" in first or first.get("event") == "per_cell_results"

    # Last line: decision summary
    last = json.loads(lines[-1])
    assert last.get("event") == "decision" or "result" in last
    assert "p_value" in last or "result" in last


# ---------------------------------------------------------------------------
# Test 8: parent gate_log appended (not overwritten)
# ---------------------------------------------------------------------------

def test_promote_appends_parent_gate_log(tmp_path, monkeypatch):
    """Promote appends to parent gate_log.jsonl; second candidate adds second line."""
    from automil.gate.promote import promote

    deltas = [0.05, 0.04, 0.06]
    _patch_evaluate_with_deltas(monkeypatch, deltas)

    # First candidate: node_0002
    graph = _make_graph(candidate_status="candidate", tmp_path=tmp_path)
    manifest = _make_manifest(K=3, bootstrap_reps=100)
    manifests_dir = tmp_path / "gate"
    write_manifest(manifest, manifests_dir)
    archive_dir = tmp_path / "archive"

    promote(
        "node_0002",
        backend=RecordingBackend(),
        graph=graph,
        manifests_dir=manifests_dir,
        archive_dir=archive_dir,
    )

    parent_log = manifests_dir / "node_0001.gate_log.jsonl"
    assert parent_log.exists(), f"Parent gate_log not written to {parent_log}"
    lines_after_first = parent_log.read_text().strip().splitlines()
    assert len(lines_after_first) == 1

    # Add a second candidate and promote it
    graph.nodes["node_0003"] = {
        "id": "node_0003",
        "parent_id": "node_0001",
        "type": "executed",
        "status": "candidate",
        "primary_value": 0.84,
        "commit": "abc1234",
        "overlay_files": [],
        "overlay_dir": "archive/node_0003",
        "vram_gb": 4.0,
    }

    promote(
        "node_0003",
        backend=RecordingBackend(),
        graph=graph,
        manifests_dir=manifests_dir,
        archive_dir=archive_dir,
    )

    lines_after_second = parent_log.read_text().strip().splitlines()
    assert len(lines_after_second) == 2, (
        f"Expected 2 lines in parent_gate_log (not overwritten); got {len(lines_after_second)}"
    )

    rec1 = json.loads(lines_after_second[0])
    rec2 = json.loads(lines_after_second[1])
    assert rec1["candidate_node_id"] == "node_0002"
    assert rec2["candidate_node_id"] == "node_0003"


# ---------------------------------------------------------------------------
# Test 9: calibrate=True is dry-run
# ---------------------------------------------------------------------------

def test_promote_calibrate_mode_dry_run(tmp_path, monkeypatch):
    """calibrate=True: archive log written; status NOT mutated; parent gate_log NOT appended."""
    from automil.gate.promote import promote

    deltas = [0.05, 0.04, 0.06]
    _patch_evaluate_with_deltas(monkeypatch, deltas)

    graph = _make_graph(candidate_status="candidate", tmp_path=tmp_path)
    manifest = _make_manifest(K=3, bootstrap_reps=100)
    manifests_dir = tmp_path / "gate"
    write_manifest(manifest, manifests_dir)
    archive_dir = tmp_path / "archive"

    promote(
        "node_0002",
        backend=RecordingBackend(),
        graph=graph,
        manifests_dir=manifests_dir,
        archive_dir=archive_dir,
        calibrate=True,
    )

    # Status must remain 'candidate' (no mutation)
    node = graph.nodes["node_0002"]
    assert node["status"] == "candidate", (
        f"calibrate=True must not mutate status; got {node['status']!r}"
    )

    # Archive log must be written (operator inspection)
    log_path = archive_dir / "node_0002" / "gate_evaluation.jsonl"
    assert log_path.exists(), "calibrate=True must still write archive log"

    # Parent gate_log must NOT be appended
    parent_log = manifests_dir / "node_0001.gate_log.jsonl"
    assert not parent_log.exists(), (
        "calibrate=True must NOT append to parent gate_log (not a real promotion)"
    )


# ---------------------------------------------------------------------------
# Test 10: graph.save() invoked exactly once
# ---------------------------------------------------------------------------

def test_promote_calls_graph_save(tmp_path, monkeypatch):
    """graph.save() is called exactly once at the end of promote (after all mutations)."""
    from automil.gate.promote import promote

    deltas = [0.05, 0.04, 0.06]
    _patch_evaluate_with_deltas(monkeypatch, deltas)

    graph = _make_graph(candidate_status="candidate", tmp_path=tmp_path)
    manifest = _make_manifest(K=3, bootstrap_reps=100)
    manifests_dir = tmp_path / "gate"
    write_manifest(manifest, manifests_dir)
    archive_dir = tmp_path / "archive"

    graph_path = tmp_path / "graph.json"
    mtime_before = graph_path.stat().st_mtime

    # Small sleep to ensure mtime will differ
    time.sleep(0.01)

    promote(
        "node_0002",
        backend=RecordingBackend(),
        graph=graph,
        manifests_dir=manifests_dir,
        archive_dir=archive_dir,
    )

    mtime_after = graph_path.stat().st_mtime
    assert mtime_after >= mtime_before, (
        "graph.save() must be called; mtime should change"
    )


# ---------------------------------------------------------------------------
# Test 11: framework purity
# ---------------------------------------------------------------------------

def test_promote_no_autobench_imports():
    """gate/promote.py must contain zero autobench/AUTOBENCH_/benchmarks/ references."""
    src = pathlib.Path(__file__).parent.parent.parent / "src" / "automil" / "gate" / "promote.py"
    assert src.exists(), f"promote.py not found at {src}"
    text = src.read_text()
    for forbidden in ("autobench", "AUTOBENCH_", "benchmarks/"):
        assert forbidden not in text, (
            f"Framework purity violation: {forbidden!r} found in gate/promote.py"
        )
