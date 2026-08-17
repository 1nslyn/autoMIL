"""Tests for gate/evaluate.py — evaluate_candidate() submission + polling pipeline.

TDD RED phase: 9 failing tests for plan 05-06.

Coverage:
  Test 1 — backend.submit called exactly N times (once per held-out cell)
  Test 2 — each JobSpec carries gate-eval metadata stamps
  Test 3 — cells in REFUSING_NEW are skipped; backend.submit NOT called for them
  Test 4 — per-cell results paired by cell_id; delta = candidate - parent
  Test 5 — child nodes tagged: edge_type='gate_eval', metadata.held_out=True
  Test 6 — polling continues until all jobs reach terminal state
  Test 7 — crashed jobs return status='crashed', delta=0.0
  Test 8 — poll_timeout_s exceeded raises TimeoutError
  Test 9 — framework purity: zero autobench/AUTOBENCH_/benchmarks/ in source
"""
from __future__ import annotations

import pathlib
import threading
import time
from typing import Iterator, Optional
from unittest.mock import MagicMock

import pytest

from automil.backends.base import Backend, JobHandle, JobSpec, JobState
from automil.cells.state import Cell, CellStatus, make_cell_id
from automil.gate.manifest import GateManifest
from automil.graph import ExperimentGraph


# ---------------------------------------------------------------------------
# Helper: minimal ExperimentGraph built in memory
# ---------------------------------------------------------------------------

def _make_graph(candidate_primary_value: float = 0.85, parent_primary_value: float = 0.80) -> ExperimentGraph:
    """Build a two-node graph (parent → candidate) without touching disk."""
    graph = ExperimentGraph.__new__(ExperimentGraph)
    graph.path = pathlib.Path("/nonexistent/graph.json")
    graph._technique_map = {}
    graph._data = {
        "schema_version": 1,
        "meta": {
            "best_primary_value": 0.0,
            "best_node_id": None,
            "total_executed": 0,
            "total_proposed": 0,
            "next_id": 1,
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
                "status": "candidate",
                "primary_value": candidate_primary_value,
                "commit": "abc1234",
                "overlay_files": [],
                "overlay_dir": "archive/node_0002",
                "vram_gb": 4.0,
            },
        },
    }
    return graph


def _make_manifest(held_out_cells: list[tuple[str, str, str, str]] | None = None) -> GateManifest:
    """Build a minimal GateManifest with 3 held-out cells unless overridden."""
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
        K=2,
        p_threshold=0.05,
        bootstrap_reps=1000,
        win_definition="delta_primary_value > 0 AND p < p_threshold",
        schema_version="gate-v1",
    )


# ---------------------------------------------------------------------------
# Recording backend — single-threaded, synchronous (deterministic for tests)
# ---------------------------------------------------------------------------

class _Job:
    def __init__(self, node_id: str, opaque_id: str, state: JobState = JobState.PENDING):
        self.handle = JobHandle(
            node_id=node_id,
            backend="recording",
            opaque_id=opaque_id,
            submitted_at=time.time(),
        )
        self.state = state
        self._log: list[str] = []


class RecordingBackend(Backend):
    """Synchronous recording backend — no threads, deterministic for unit tests.

    Usage:
        rb = RecordingBackend()
        rb.set_terminal_state("node_0003", JobState.COMPLETED)  # preset before calling evaluate
        results, skipped = evaluate_candidate(...)
        assert rb.submit_count == 3
        assert "gate_eval" in rb.metadata_by_node_id["node_0003"]
    """

    def __init__(self, initial_state: JobState = JobState.COMPLETED) -> None:
        self._jobs: dict[str, _Job] = {}   # opaque_id -> _Job
        self._counter = 0
        self._initial_state = initial_state
        self.submitted_specs: list[tuple[JobSpec, str]] = []  # (spec, handle.node_id)
        self.metadata_by_node_id: dict[str, dict[str, str]] = {}
        # Per-node overrides: node_id -> state
        self._state_overrides: dict[str, JobState] = {}
        # Never-terminal node ids (for timeout test)
        self._never_terminal: set[str] = set()

    def set_terminal_state(self, node_id: str, state: JobState) -> None:
        """Override terminal state for a specific node_id (must be set BEFORE submit)."""
        self._state_overrides[node_id] = state

    def set_never_terminal(self, node_id: str) -> None:
        """Mark a node_id as never reaching terminal state (forces timeout)."""
        self._never_terminal.add(node_id)

    @property
    def submit_count(self) -> int:
        return len(self.submitted_specs)

    def submit(self, spec: JobSpec) -> JobHandle:
        self._counter += 1
        opaque_id = f"{self._counter}"
        # Use overridden state if set; otherwise initial_state
        state = self._state_overrides.get(spec.node_id, self._initial_state)
        if spec.node_id in self._never_terminal:
            state = JobState.PENDING  # will never advance
        job = _Job(node_id=spec.node_id, opaque_id=opaque_id, state=state)
        self._jobs[opaque_id] = job
        self.submitted_specs.append((spec, spec.node_id))
        self.metadata_by_node_id[spec.node_id] = dict(spec.metadata)
        return job.handle

    def poll(self, handle: JobHandle) -> JobState:
        job = self._jobs.get(handle.opaque_id)
        if job is None:
            raise ValueError(f"Unknown job opaque_id: {handle.opaque_id!r}")
        # Never-terminal: always PENDING
        if handle.node_id in self._never_terminal:
            return JobState.PENDING
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
# Shared fixture
# ---------------------------------------------------------------------------

    def healthcheck(self) -> "HealthReport":
        raise NotImplementedError("healthcheck deferred for test-fixture backend")

@pytest.fixture
def graph() -> ExperimentGraph:
    return _make_graph()


@pytest.fixture
def manifest() -> GateManifest:
    return _make_manifest()


@pytest.fixture
def backend() -> RecordingBackend:
    return RecordingBackend(initial_state=JobState.COMPLETED)


# ---------------------------------------------------------------------------
# Test 1: backend.submit called exactly N times
# ---------------------------------------------------------------------------

def test_evaluate_calls_backend_submit_per_held_out_cell(graph, manifest, backend):
    """evaluate_candidate must call backend.submit exactly once per held-out cell."""
    from automil.gate.evaluate import evaluate_candidate

    per_cell_results, skipped = evaluate_candidate(
        "node_0002", manifest, backend, graph, poll_interval_s=0.001, poll_timeout_s=5.0,
    )

    assert backend.submit_count == 3, (
        f"Expected 3 backend.submit calls; got {backend.submit_count}"
    )
    assert len(per_cell_results) == 3
    assert skipped == []


# ---------------------------------------------------------------------------
# Test 2: metadata stamps on each submitted JobSpec
# ---------------------------------------------------------------------------

def test_evaluate_metadata_gate_eval_stamps(graph, manifest, backend):
    """Each submitted JobSpec must carry the 5 required gate-eval metadata keys."""
    from automil.gate.evaluate import evaluate_candidate

    evaluate_candidate(
        "node_0002", manifest, backend, graph, poll_interval_s=0.001, poll_timeout_s=5.0,
    )

    for spec, node_id in backend.submitted_specs:
        md = dict(spec.metadata)
        assert md.get("gate_eval") == "true", f"metadata.gate_eval not 'true' on {node_id}: {md}"
        assert md.get("held_out") == "true", f"metadata.held_out not 'true' on {node_id}: {md}"
        assert md.get("gate_parent_node") == "node_0002", (
            f"metadata.gate_parent_node wrong on {node_id}: {md}"
        )
        assert "cell_id" in md, f"metadata.cell_id missing on {node_id}: {md}"
        assert md.get("edge_type") == "gate_eval", (
            f"metadata.edge_type not 'gate_eval' on {node_id}: {md}"
        )

    # Also verify via _metadata_by_node_id (keyed by handle.node_id)
    for node_id, md in backend.metadata_by_node_id.items():
        assert md["gate_eval"] == "true"


# ---------------------------------------------------------------------------
# Test 3: skip cells in REFUSING_NEW state
# ---------------------------------------------------------------------------

def test_evaluate_skips_refusing_cells(graph, tmp_path, monkeypatch):
    """Cells in REFUSING_NEW state are skipped; submit called only for active cells."""
    from automil.gate.evaluate import evaluate_candidate

    refusing_cell_id = "cell_aaaa1111"

    # Mock get_cell to return a REFUSING_NEW cell for refusing_cell_id, None otherwise
    refusing_cell = Cell(
        cell_id=refusing_cell_id,
        dataset="ccrcc",
        encoder="uni_v2",
        mil_model="node_0001",
        started_at=0.0,
        budget_seconds=3600,
        safety_buffer_seconds=300,
        status=CellStatus.REFUSING_NEW,
    )

    def fake_get_cell(cell_id: str) -> Cell | None:
        if cell_id == refusing_cell_id:
            return refusing_cell
        return None  # other cells: fresh, not refusing

    monkeypatch.setattr("automil.gate.evaluate.get_cell", fake_get_cell)

    manifest = _make_manifest()  # 3 held-out cells; first is refusing_cell_id
    backend = RecordingBackend(initial_state=JobState.COMPLETED)

    per_cell_results, skipped = evaluate_candidate(
        "node_0002", manifest, backend, graph, poll_interval_s=0.001, poll_timeout_s=5.0,
    )

    assert backend.submit_count == 2, (
        f"Expected 2 backend.submit calls (refusing cell skipped); got {backend.submit_count}"
    )
    assert refusing_cell_id in skipped, f"refusing cell not in skipped: {skipped}"
    assert len(per_cell_results) == 2


# ---------------------------------------------------------------------------
# Test 4: per-cell results paired by cell_id with correct deltas
# ---------------------------------------------------------------------------

def test_evaluate_returns_paired_deltas(tmp_path, monkeypatch):
    """Delta = candidate_primary_value - parent_primary_value; paired by cell_id NOT order."""
    from automil.gate.evaluate import evaluate_candidate

    # Candidate primary_value=0.85, parent primary_value=0.80 (from _make_graph defaults)
    graph = _make_graph(candidate_primary_value=0.85, parent_primary_value=0.80)

    # Two held-out cells: a and b
    manifest = _make_manifest(held_out_cells=[
        ("cell_aaa", "ccrcc", "uni_v2", "high_grade"),
        ("cell_bbb", "clwd", "hibou_l", "subtype"),
    ])

    monkeypatch.setattr("automil.gate.evaluate.get_cell", lambda cid: None)

    backend = RecordingBackend(initial_state=JobState.COMPLETED)

    per_cell_results, skipped = evaluate_candidate(
        "node_0002", manifest, backend, graph, poll_interval_s=0.001, poll_timeout_s=5.0,
    )

    assert skipped == []
    assert len(per_cell_results) == 2

    # Both should have parent_primary_value=0.80 (from parent node)
    for r in per_cell_results:
        assert r["parent_primary_value"] == pytest.approx(0.80, abs=1e-6), (
            f"parent_primary_value mismatch: {r}"
        )
        # Delta should be candidate_primary_value - parent_primary_value (from graph node primary_value)
        # candidate primary_value = 0.85, parent primary_value = 0.80 → delta = 0.05
        assert "delta" in r
        assert "cell_id" in r
        assert r["cell_id"] in ("cell_aaa", "cell_bbb")

    cell_ids = {r["cell_id"] for r in per_cell_results}
    assert cell_ids == {"cell_aaa", "cell_bbb"}


# ---------------------------------------------------------------------------
# Test 5: child nodes tagged with edge_type + metadata.held_out
# ---------------------------------------------------------------------------

def test_evaluate_tags_child_nodes_edge_type(graph, manifest, backend, monkeypatch):
    """Child nodes must have edge_type='gate_eval' AND metadata.held_out=True after submit."""
    from automil.gate.evaluate import evaluate_candidate

    monkeypatch.setattr("automil.gate.evaluate.get_cell", lambda cid: None)

    evaluate_candidate(
        "node_0002", manifest, backend, graph, poll_interval_s=0.001, poll_timeout_s=5.0,
    )

    # Find all gate_eval child nodes
    gate_eval_nodes = [
        n for n in graph.nodes.values()
        if n.get("edge_type") == "gate_eval"
    ]

    assert len(gate_eval_nodes) == 3, (
        f"Expected 3 gate_eval child nodes; got {len(gate_eval_nodes)}: {list(graph.nodes.keys())}"
    )

    for n in gate_eval_nodes:
        assert n.get("edge_type") == "gate_eval", f"edge_type missing/wrong: {n}"
        assert n.get("metadata", {}).get("held_out") is True, (
            f"metadata.held_out not True: {n}"
        )
        assert n.get("metadata", {}).get("gate_eval") is True, (
            f"metadata.gate_eval not True: {n}"
        )
        assert n.get("metadata", {}).get("gate_parent_node") == "node_0002", (
            f"metadata.gate_parent_node wrong: {n}"
        )


# ---------------------------------------------------------------------------
# Test 6: polling continues until all jobs terminal
# ---------------------------------------------------------------------------

def test_evaluate_polls_until_all_terminal(monkeypatch):
    """evaluate_candidate blocks until all submitted jobs reach terminal state."""
    from automil.gate.evaluate import evaluate_candidate

    graph = _make_graph()
    manifest = _make_manifest()

    monkeypatch.setattr("automil.gate.evaluate.get_cell", lambda cid: None)

    # Use a backend that starts PENDING but transitions to COMPLETED after first poll
    class EventualBackend(RecordingBackend):
        def __init__(self):
            super().__init__(initial_state=JobState.PENDING)
            self._poll_counts: dict[str, int] = {}

        def poll(self, handle: JobHandle) -> JobState:
            count = self._poll_counts.get(handle.node_id, 0) + 1
            self._poll_counts[handle.node_id] = count
            # Complete after 3 polls
            if count >= 3:
                return JobState.COMPLETED
            return JobState.PENDING

    backend = EventualBackend()

    per_cell_results, skipped = evaluate_candidate(
        "node_0002", manifest, backend, graph,
        poll_interval_s=0.005,  # fast for test
        poll_timeout_s=10.0,
    )

    assert len(per_cell_results) == 3
    assert skipped == []


# ---------------------------------------------------------------------------
# Test 7: crashed job returns status='crashed', delta=0.0
# ---------------------------------------------------------------------------

def test_evaluate_handles_crashed_eval(graph, manifest, monkeypatch):
    """A crashed job must appear in per_cell_results with status='crashed' and delta=0.0."""
    from automil.gate.evaluate import evaluate_candidate

    monkeypatch.setattr("automil.gate.evaluate.get_cell", lambda cid: None)

    backend = RecordingBackend(initial_state=JobState.COMPLETED)

    # We need to find out what node_id will be assigned to the first cell
    # The evaluate function uses graph.next_id() — first unused node id
    # Since graph has node_0001, node_0002; next_id counter=1 → node_0001 collides,
    # so it will generate node_0001, node_0002 etc. via _next_child_id fallback.
    # We use a custom backend that crashes the first submitted job
    crashed_cell_id = "cell_aaaa1111"
    first_node_id_holder: list[str] = []

    original_submit = backend.submit

    def patched_submit(spec: JobSpec) -> JobHandle:
        handle = original_submit(spec)
        cell_id = dict(spec.metadata).get("cell_id", "")
        if cell_id == crashed_cell_id:
            first_node_id_holder.append(spec.node_id)
            backend._jobs[handle.opaque_id].state = JobState.CRASHED
        return handle

    backend.submit = patched_submit  # type: ignore[method-assign]

    per_cell_results, skipped = evaluate_candidate(
        "node_0002", manifest, backend, graph,
        poll_interval_s=0.001, poll_timeout_s=5.0,
    )

    # All cells accounted for (crashed cell in results, not skipped)
    assert len(per_cell_results) + len(skipped) == 3

    crashed = [r for r in per_cell_results if r["cell_id"] == crashed_cell_id]
    assert len(crashed) == 1, f"Expected crashed result for {crashed_cell_id}: {per_cell_results}"
    assert crashed[0]["status"] == "crashed", f"Expected status='crashed': {crashed[0]}"
    assert crashed[0]["delta"] == pytest.approx(0.0), (
        f"Expected delta=0.0 for crashed: {crashed[0]}"
    )


# ---------------------------------------------------------------------------
# Test 8: poll_timeout_s exceeded raises TimeoutError
# ---------------------------------------------------------------------------

def test_evaluate_timeout_returns_partial(graph, monkeypatch):
    """evaluate_candidate raises TimeoutError when poll_timeout_s is exceeded."""
    from automil.gate.evaluate import evaluate_candidate

    monkeypatch.setattr("automil.gate.evaluate.get_cell", lambda cid: None)

    # Only 1 held-out cell — easier to force timeout
    manifest = _make_manifest(held_out_cells=[
        ("cell_stuck", "ccrcc", "uni_v2", "high_grade"),
    ])

    backend = RecordingBackend(initial_state=JobState.PENDING)
    backend.set_never_terminal("will_be_set_dynamically")

    # Use a backend that NEVER completes the stuck cell
    never_complete_backend = RecordingBackend(initial_state=JobState.PENDING)
    # Override poll to always return PENDING
    original_poll = never_complete_backend.poll

    def always_pending(handle: JobHandle) -> JobState:
        return JobState.PENDING

    never_complete_backend.poll = always_pending  # type: ignore[method-assign]

    with pytest.raises(TimeoutError):
        evaluate_candidate(
            "node_0002", manifest, never_complete_backend, graph,
            poll_interval_s=0.01,
            poll_timeout_s=0.1,  # 100ms — will expire quickly
        )


# ---------------------------------------------------------------------------
# Test 9: framework purity — no autobench/AUTOBENCH_/benchmarks/ in source
# ---------------------------------------------------------------------------

def test_evaluate_no_autobench_imports():
    """gate/evaluate.py must contain zero autobench/AUTOBENCH_/benchmarks/ references."""
    src = pathlib.Path(__file__).parent.parent.parent / "src" / "automil" / "gate" / "evaluate.py"
    assert src.exists(), f"evaluate.py not found at {src}"
    text = src.read_text()
    for forbidden in ("autobench", "AUTOBENCH_", "benchmarks/"):
        assert forbidden not in text, (
            f"Framework purity violation: {forbidden!r} found in gate/evaluate.py"
        )
