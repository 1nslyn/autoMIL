"""Tests for the D-143 Two-stage gate composition: Stage A (nominate) + Stage B (promote).

TDD RED phase: 4 failing tests for plan 05-07 Task 2.

The two stages use DISJOINT data:
  Stage A: Pareto on search cells (keep/discard, enforced upstream by nominate)
  Stage B: held-out cells (paired Wilcoxon, enforced by promote)

Coverage:
  Test 1 — full keep -> nominate -> promote (pass) -> registered trail in history
  Test 2 — Stage A blocks promote without prior nominate (ValueError)
  Test 3 — Stage B can revert to keep: keep -> nominate -> promote (fail) -> keep
  Test 4 — Stages use disjoint data: search-cell primary_value vs held-out deltas
"""
from __future__ import annotations

import pathlib
import sys
import time
from typing import Iterator, Optional

import numpy as np
import pytest

from automil.backends.base import Backend, JobHandle, JobSpec, JobState
from automil.gate.manifest import GateManifest, write_manifest
from automil.gate.nominate import nominate
from automil.graph import ExperimentGraph


# ---------------------------------------------------------------------------
# Helpers — shared graph / manifest builders
# ---------------------------------------------------------------------------

def _make_graph(tmp_path) -> ExperimentGraph:
    """Build parent (node_0001, keep) -> candidate (node_0002, keep) graph on disk."""
    graph_path = tmp_path / "graph.json"
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
                "primary_value": 0.80,
                "commit": "abc1234",
                "overlay_files": [],
                "overlay_dir": "archive/node_0001",
                "vram_gb": 4.0,
                "history": [],
            },
            "node_0002": {
                "id": "node_0002",
                "parent_id": "node_0001",
                "type": "executed",
                "status": "keep",
                # This primary_value (0.85) comes from the search cell — Stage A result.
                # It must NOT appear in the Stage B deltas.
                "primary_value": 0.85,
                "commit": "abc1234",
                "overlay_files": [],
                "overlay_dir": "archive/node_0002",
                "vram_gb": 4.0,
                "history": [],
            },
        },
    }
    graph.save()
    return graph


def _make_manifest(tmp_path) -> tuple[GateManifest, pathlib.Path]:
    """Build a 2-cell held-out manifest for node_0001."""
    held_out_cells = [
        ("cell_heldout_a", "clwd", "hibou_l", "subtype"),
        ("cell_heldout_b", "ccrcc", "ctranspath", "high_grade"),
    ]
    manifest = GateManifest(
        parent_id="node_0001",
        created_at="2026-05-05T00:00:00Z",
        git_committed_at_sha="abc1234",
        held_out_cells=tuple(tuple(c) for c in held_out_cells),  # type: ignore[arg-type]
        K=2,
        # Use p_threshold=0.5 -> Bonferroni-corrected alpha=0.25; Wilcoxon min p=0.25 for n=2
        # so with positive deltas this will pass
        p_threshold=0.5,
        bootstrap_reps=100,
        win_definition="delta_primary_value > 0 AND p < p_threshold",
        schema_version="gate-v1",
    )
    manifests_dir = tmp_path / "gate"
    write_manifest(manifest, manifests_dir)
    return manifest, manifests_dir


def _get_promote_module():
    """Return the promote module object (not the function) via sys.modules."""
    import importlib
    if "automil.gate.promote" not in sys.modules:
        importlib.import_module("automil.gate.promote")
    return sys.modules["automil.gate.promote"]


# ---------------------------------------------------------------------------
# Synchronous recording backend — pure pass-through (all jobs COMPLETED)
# ---------------------------------------------------------------------------

class _Job:
    def __init__(self, node_id: str, opaque_id: str):
        self.handle = JobHandle(
            node_id=node_id,
            backend="recording",
            opaque_id=opaque_id,
            submitted_at=time.time(),
        )
        self.state = JobState.COMPLETED


class RecordingBackend(Backend):
    def __init__(self) -> None:
        self._jobs: dict[str, _Job] = {}
        self._counter = 0
        self.submitted_specs: list[JobSpec] = []

    def submit(self, spec: JobSpec) -> JobHandle:
        self._counter += 1
        job = _Job(node_id=spec.node_id, opaque_id=str(self._counter))
        self._jobs[str(self._counter)] = job
        self.submitted_specs.append(spec)
        return job.handle

    def poll(self, handle: JobHandle) -> JobState:
        return self._jobs[handle.opaque_id].state

    def cancel(self, handle: JobHandle, signal: Optional[int] = None) -> None:
        pass

    def list_running(self) -> list[JobHandle]:
        return []

    def log_iter(self, handle: JobHandle) -> Iterator[str]:
        return iter([])


# ---------------------------------------------------------------------------
# Test 1: full keep -> nominate -> promote (pass) -> registered trail
# ---------------------------------------------------------------------------

    def healthcheck(self) -> "HealthReport":
        raise NotImplementedError("healthcheck deferred for test-fixture backend")

def test_two_stage_gate_keep_then_candidate_then_registered(tmp_path, monkeypatch):
    """Full D-143 two-stage trail: keep -> nominated -> registered.

    History must contain exactly these gate events in order:
      1. event="nominated"  (Stage A: nominate)
      2. event="gate_result", result="pass"  (Stage B: promote)
    """
    from automil.gate.promote import promote

    graph = _make_graph(tmp_path)
    manifest, manifests_dir = _make_manifest(tmp_path)
    archive_dir = tmp_path / "archive"
    monkeypatch.setattr("automil.gate.evaluate.get_cell", lambda cid: None)

    # Pre-stamp child nodes primary_value so polling gives positive deltas
    def fake_evaluate(candidate_node_id, manifest, backend, graph, **kwargs):
        # Both held-out cells show positive delta vs parent (parent=0.80)
        per_cell = [
            {
                "cell_id": "cell_heldout_a",
                "dataset": "clwd",
                "encoder": "hibou_l",
                "task": "subtype",
                "child_node_id": "child_0001",
                "candidate_primary_value": 0.88,
                "parent_primary_value": 0.80,
                "delta": 0.08,
                "status": "completed",
            },
            {
                "cell_id": "cell_heldout_b",
                "dataset": "ccrcc",
                "encoder": "ctranspath",
                "task": "high_grade",
                "child_node_id": "child_0002",
                "candidate_primary_value": 0.87,
                "parent_primary_value": 0.80,
                "delta": 0.07,
                "status": "completed",
            },
        ]
        return per_cell, []

    monkeypatch.setattr(_get_promote_module(), "evaluate_candidate", fake_evaluate)

    # --- Stage A: nominate ---
    assert graph.nodes["node_0002"]["status"] == "keep"
    nominate("node_0002", graph)
    graph.save()
    assert graph.nodes["node_0002"]["status"] == "candidate"

    # --- Stage B: promote ---
    result = promote(
        "node_0002",
        backend=RecordingBackend(),
        graph=graph,
        manifests_dir=manifests_dir,
        archive_dir=archive_dir,
    )
    assert result is True
    assert graph.nodes["node_0002"]["status"] == "registered"

    # Verify full history trail
    history = graph.nodes["node_0002"].get("history", [])
    event_names = [e.get("event") for e in history]

    assert "nominated" in event_names, f"'nominated' not in history: {event_names}"
    assert "gate_result" in event_names, f"'gate_result' not in history: {event_names}"

    # Sequence check: nominated comes before gate_result
    nominated_idx = next(i for i, e in enumerate(history) if e.get("event") == "nominated")
    gate_idx = next(i for i, e in enumerate(history) if e.get("event") == "gate_result")
    assert nominated_idx < gate_idx, (
        f"'nominated' (idx={nominated_idx}) must precede 'gate_result' (idx={gate_idx})"
    )

    # Gate result must be 'pass'
    gate_event = history[gate_idx]
    assert gate_event["result"] == "pass", f"Expected result='pass': {gate_event}"


# ---------------------------------------------------------------------------
# Test 2: Stage A blocks promote without prior nominate
# ---------------------------------------------------------------------------

def test_stage_a_blocks_promote_without_nominate(tmp_path):
    """Calling promote directly on a keep node raises ValueError — Stage A enforced."""
    from automil.gate.promote import promote

    graph = _make_graph(tmp_path)
    manifest, manifests_dir = _make_manifest(tmp_path)
    archive_dir = tmp_path / "archive"

    # node_0002 has status='keep' — nominate NOT called
    assert graph.nodes["node_0002"]["status"] == "keep"

    with pytest.raises(ValueError) as exc_info:
        promote(
            "node_0002",
            backend=RecordingBackend(),
            graph=graph,
            manifests_dir=manifests_dir,
            archive_dir=archive_dir,
        )

    msg = str(exc_info.value)
    assert "candidate" in msg, f"Expected 'candidate' in error: {msg!r}"
    assert "nominate" in msg.lower(), f"Expected 'nominate' hint in error: {msg!r}"


# ---------------------------------------------------------------------------
# Test 3: Stage B can revert to keep (fail path)
# ---------------------------------------------------------------------------

def test_stage_b_can_revert_to_keep(tmp_path, monkeypatch):
    """keep -> nominate -> promote (fail) -> back to keep; history has both events."""
    from automil.gate.promote import promote

    graph = _make_graph(tmp_path)
    manifest, manifests_dir = _make_manifest(tmp_path)
    archive_dir = tmp_path / "archive"

    # Mixed/negative deltas — gate must fail
    def fake_evaluate_fail(candidate_node_id, manifest, backend, graph, **kwargs):
        per_cell = [
            {
                "cell_id": "cell_heldout_a",
                "dataset": "clwd",
                "encoder": "hibou_l",
                "task": "subtype",
                "child_node_id": "child_0001",
                "candidate_primary_value": 0.78,
                "parent_primary_value": 0.80,
                "delta": -0.02,
                "status": "completed",
            },
            {
                "cell_id": "cell_heldout_b",
                "dataset": "ccrcc",
                "encoder": "ctranspath",
                "task": "high_grade",
                "child_node_id": "child_0002",
                "candidate_primary_value": 0.79,
                "parent_primary_value": 0.80,
                "delta": -0.01,
                "status": "completed",
            },
        ]
        return per_cell, []

    monkeypatch.setattr(_get_promote_module(), "evaluate_candidate", fake_evaluate_fail)

    # Stage A
    nominate("node_0002", graph)
    graph.save()
    assert graph.nodes["node_0002"]["status"] == "candidate"

    # Stage B (fail)
    result = promote(
        "node_0002",
        backend=RecordingBackend(),
        graph=graph,
        manifests_dir=manifests_dir,
        archive_dir=archive_dir,
    )
    assert result is False
    assert graph.nodes["node_0002"]["status"] == "keep", (
        "Failed gate must revert status to 'keep'"
    )

    # History must have nominated event THEN gate_result=fail
    history = graph.nodes["node_0002"].get("history", [])
    nominated_events = [e for e in history if e.get("event") == "nominated"]
    gate_events = [e for e in history if e.get("event") == "gate_result"]

    assert len(nominated_events) == 1, f"Expected 1 nominated event: {history}"
    assert len(gate_events) == 1, f"Expected 1 gate_result event: {history}"
    assert gate_events[0]["result"] == "fail"

    # Sequence: nominated before gate_result
    h_events = [(i, e.get("event")) for i, e in enumerate(history)]
    nom_idx = next(i for i, ev in h_events if ev == "nominated")
    gate_idx = next(i for i, ev in h_events if ev == "gate_result")
    assert nom_idx < gate_idx


# ---------------------------------------------------------------------------
# Test 4: stages use disjoint data (D-143 defence)
# ---------------------------------------------------------------------------

def test_stages_use_disjoint_data(tmp_path, monkeypatch):
    """Stage A primary_value (search cell) must not appear in Stage B deltas.

    The candidate's primary_value (0.85) comes from the search cell (Stage A).
    Stage B deltas must come purely from held-out eval results — not from
    node["primary_value"] being re-used as a delta.

    We capture what deltas were passed to paired_wilcoxon_with_bootstrap and
    verify that 0.85 (or 0.85-0.80=0.05 derived from search primary_value) does
    NOT appear as one of the captured deltas if our fake_evaluate provides
    different values.
    """
    from automil.gate.promote import promote

    graph = _make_graph(tmp_path)
    manifest, manifests_dir = _make_manifest(tmp_path)
    archive_dir = tmp_path / "archive"

    # node_0002["primary_value"] = 0.85 — the search-cell primary_value (Stage A)
    # held-out cell deltas must be 0.07 and 0.09 (different from search delta 0.05)
    held_out_deltas = [0.07, 0.09]

    def fake_evaluate(candidate_node_id, manifest, backend, graph, **kwargs):
        per_cell = [
            {
                "cell_id": "cell_heldout_a",
                "dataset": "clwd",
                "encoder": "hibou_l",
                "task": "subtype",
                "child_node_id": "child_0001",
                "candidate_primary_value": 0.80 + held_out_deltas[0],
                "parent_primary_value": 0.80,
                "delta": held_out_deltas[0],
                "status": "completed",
            },
            {
                "cell_id": "cell_heldout_b",
                "dataset": "ccrcc",
                "encoder": "ctranspath",
                "task": "high_grade",
                "child_node_id": "child_0002",
                "candidate_primary_value": 0.80 + held_out_deltas[1],
                "parent_primary_value": 0.80,
                "delta": held_out_deltas[1],
                "status": "completed",
            },
        ]
        return per_cell, []

    monkeypatch.setattr(_get_promote_module(), "evaluate_candidate", fake_evaluate)

    # Spy on paired_wilcoxon_with_bootstrap to capture what deltas were passed
    captured_deltas: list[np.ndarray] = []
    from automil.gate import stats as gate_stats
    original_stats = gate_stats.paired_wilcoxon_with_bootstrap

    def capturing_stats(deltas, p_threshold, bootstrap_reps=1000, rng_seed=None):
        captured_deltas.append(deltas.copy())
        return original_stats(deltas, p_threshold, bootstrap_reps, rng_seed)

    monkeypatch.setattr(_get_promote_module(), "paired_wilcoxon_with_bootstrap", capturing_stats)

    # Stage A
    nominate("node_0002", graph)
    graph.save()

    # Stage B
    promote(
        "node_0002",
        backend=RecordingBackend(),
        graph=graph,
        manifests_dir=manifests_dir,
        archive_dir=archive_dir,
    )

    assert len(captured_deltas) == 1, (
        f"Expected 1 call to paired_wilcoxon_with_bootstrap; got {len(captured_deltas)}"
    )
    actual_deltas = captured_deltas[0]

    # The Stage A search-cell delta (0.85 - 0.80 = 0.05) must NOT appear in Stage B deltas
    search_cell_delta = 0.85 - 0.80  # = 0.05
    assert search_cell_delta not in actual_deltas.tolist(), (
        f"Stage A search-cell delta ({search_cell_delta}) leaked into Stage B deltas: {actual_deltas}"
    )

    # Stage B deltas must match the held-out eval results
    assert set(actual_deltas.tolist()) == set(held_out_deltas), (
        f"Stage B deltas {actual_deltas.tolist()} don't match held-out deltas {held_out_deltas}"
    )
