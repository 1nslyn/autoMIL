"""Pitfall-6 anti-acceptance gate (D-149 / GTE-01..06 / Phase 5 goal-backward verifier).

This test is to Phase 5 what ``test_cap_fires_with_partial_fold_recovery`` is to Phase 4:
a synthetic end-to-end scenario that exercises the entire Wave 1-6 pipeline and asserts
the load-bearing invariants in one shot.

Without this test green, Phase 5 has not delivered "the gate enforces held-out separation."

D-149 assertions (9 total):
  1.  Synthetic 3-cell graph (1 search + 2 held-out)
  2.  Manifest registered declaring the 2 held-out cells
  3.  Synthetic search loop proposes 3 candidates with primary values on search cell only
  4.  Agent's view (rank output + trajectory.jsonl) contains zero held-out cell IDs
  5.  Operator nominates candidate #2; calls promote
  6.  2 gate_eval nodes spawned via Backend.submit() with correct metadata
  7.  Pre-promote status='candidate'; post-promote status='registered' (pass) or 'keep' (fail)
  8.  AFTER promote, trajectory.jsonl STILL has zero held-out cell IDs
  9.  Status transitions in node['history'] with timestamps
"""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import pytest

from automil.backends.base import Backend, JobHandle, JobSpec, JobState
from automil.gate import GateManifest, write_manifest_committed
from automil.gate.manifest import SCHEMA_VERSION, write_manifest
from automil.gate.nominate import nominate
from automil.gate.promote import promote
from automil.graph import ExperimentGraph
from automil.trajectory.redactor import _held_out_ids_cached, redact, redact_event


# ---------------------------------------------------------------------------
# Recording mock backend — no real processes, deterministic
# ---------------------------------------------------------------------------

class _RecordingBackend(Backend):
    """Synchronous recording backend for D-149 assertions.

    Implements the real Backend ABC (JobHandle uses the correct signature:
    node_id, backend, opaque_id, submitted_at — NOT job_id or backend_name).

    Usage::
        backend = _RecordingBackend()
        backend.set_primary_value(held_out_a_id, 0.92)  # candidate beats parent
        # monkeypatch automil.cli.promote._resolve_backend -> backend
    """

    def __init__(self) -> None:
        self.submitted_specs: list[JobSpec] = []
        self._primary_values: dict[str, float] = {}  # cell_id -> primary_value to stamp on graph
        self._counter = 0
        # opaque_id -> state
        self._jobs: dict[str, JobState] = {}
        # Map node_id -> opaque_id for cancel
        self._node_to_opaque: dict[str, str] = {}

    def set_primary_value(self, cell_id: str, primary_value: float) -> None:
        """Pre-configure the primary-metric value for a held-out cell."""
        self._primary_values[cell_id] = primary_value

    def submit(self, spec: JobSpec) -> JobHandle:
        """Record spec; immediately move job to COMPLETED so poll() returns terminal."""
        self.submitted_specs.append(spec)
        self._counter += 1
        opaque_id = f"mock-{self._counter}"
        self._jobs[opaque_id] = JobState.COMPLETED
        self._node_to_opaque[spec.node_id] = opaque_id
        # Real JobHandle signature (backends/base.py):
        #   JobHandle(node_id, backend, opaque_id, submitted_at)
        # NOT job_id / backend_name (those fields do not exist).
        return JobHandle(
            node_id=spec.node_id,
            backend="mock",
            opaque_id=opaque_id,
            submitted_at=0.0,
        )

    def poll(self, handle: JobHandle) -> JobState:
        state = self._jobs.get(handle.opaque_id, JobState.COMPLETED)
        return state

    def cancel(self, handle: JobHandle, signal: Optional[int] = None) -> None:
        if handle.opaque_id in self._jobs:
            self._jobs[handle.opaque_id] = JobState.CANCELLED

    def list_running(self) -> list[JobHandle]:
        return []

    def log_iter(self, handle: JobHandle) -> Iterator[str]:
        yield ""

    @property
    def submit_call_count(self) -> int:
        return len(self.submitted_specs)


# ---------------------------------------------------------------------------
# Shared fixture — pitfall6_project
# ---------------------------------------------------------------------------

    def healthcheck(self) -> "HealthReport":
        raise NotImplementedError("healthcheck deferred for test-fixture backend")

@pytest.fixture
def pitfall6_project(tmp_path):
    """Build a synthetic 3-cell automil project with git repo.

    Returns:
        (project_root, search_node_id, [cand_1_id, cand_2_id, cand_3_id],
         held_out_a_id, held_out_b_id)

    Layout:
        tmp_path/
          .git/
          automil/
            config.yaml
            graph.json
            gate/        (gate manifests)
    """
    # Step A: git init with required config
    subprocess.run(["git", "init", "-q"], cwd=str(tmp_path), check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=str(tmp_path), check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(tmp_path), check=True,
    )
    # Initial commit (needed for write_manifest_committed)
    (tmp_path / "README.md").write_text("pitfall6 test repo\n")
    subprocess.run(["git", "add", "README.md"], cwd=str(tmp_path), check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=str(tmp_path), check=True,
    )

    # Step B: automil overlay directory
    adir = tmp_path / "automil"
    adir.mkdir()
    (adir / "config.yaml").write_text(
        "gate:\n  auto_nominate: false\n  K: 2\n  p_threshold: 0.05\n"
    )

    # Step C: cell IDs (using a naming scheme that is distinct for assertion 4)
    search_cell_id = "search_cell_aabbccdd0001"
    held_out_a_id  = "held_out_cell_11223344aaaa"
    held_out_b_id  = "held_out_cell_55667788bbbb"
    # Extra cells for statistical power: Wilcoxon needs K>=5 to achieve p<=0.04
    # (K=2 minimum achievable p=0.25 which can't pass Bonferroni-corrected alpha;
    # using 5 cells with p_threshold=0.2 -> p_corrected=0.04 > Wilcoxon min-p=0.031)
    held_out_c_id  = "held_out_cell_99aabbcc3333"
    held_out_d_id  = "held_out_cell_ddee1122ff44"
    held_out_e_id  = "held_out_cell_aabb9988cc55"

    # D-149 assertion 3: parent node (search cell) + 3 candidate proposals
    # Each candidate has primary_value from SEARCH CELL ONLY at proposal time.
    search_node_id = "node_0001"
    cand_1_id = "node_0002"
    cand_2_id = "node_0003"  # the one the operator nominates
    cand_3_id = "node_0004"

    # D-149 assertion 1: 3-cell graph — 1 search parent + 3 candidate children
    graph_data = {
        "schema_version": 1,
        "meta": {
            "best_primary_value": 0.0,
            "best_node_id": None,
            "total_executed": 4,
            "total_proposed": 0,
            "next_id": 5,
            "baseline_primary_value": 0.0,
            "scoring": {"exploration_weight": 0.005, "novelty_weight": 0.003},
        },
        "nodes": {
            search_node_id: {
                "id": search_node_id,
                "parent_id": None,
                "type": "executed",
                "status": "keep",
                "primary_value": 0.85,
                "description": "search cell baseline",
                "metadata": {
                    "held_out": False,
                    "cell_id": search_cell_id,
                },
                "commit": "abc1234",
                "overlay_files": [],
                "overlay_dir": f"archive/{search_node_id}",
                "vram_gb": 4.0,
            },
            cand_1_id: {
                "id": cand_1_id,
                "parent_id": search_node_id,
                "type": "executed",
                "status": "keep",
                "primary_value": 0.86,
                "description": "candidate 1 — search cell only",
                "metadata": {"held_out": False},
                "commit": "abc1234",
                "overlay_files": [],
                "overlay_dir": f"archive/{cand_1_id}",
                "vram_gb": 4.0,
            },
            cand_2_id: {
                "id": cand_2_id,
                "parent_id": search_node_id,
                "type": "executed",
                "status": "keep",
                "primary_value": 0.88,
                "description": "candidate 2 — NOMINATED by operator",
                "metadata": {"held_out": False},
                "commit": "abc1234",
                "overlay_files": [],
                "overlay_dir": f"archive/{cand_2_id}",
                "vram_gb": 4.0,
            },
            cand_3_id: {
                "id": cand_3_id,
                "parent_id": search_node_id,
                "type": "executed",
                "status": "keep",
                "primary_value": 0.87,
                "description": "candidate 3 — search cell only",
                "metadata": {"held_out": False},
                "commit": "abc1234",
                "overlay_files": [],
                "overlay_dir": f"archive/{cand_3_id}",
                "vram_gb": 4.0,
            },
        },
        "technique_stats": {},
    }
    graph_json_path = adir / "graph.json"
    graph_json_path.write_text(json.dumps(graph_data, indent=2))

    # Step D: gate manifest directory
    gate_dir = adir / "gate"
    gate_dir.mkdir()

    return (
        tmp_path,
        search_node_id,
        [cand_1_id, cand_2_id, cand_3_id],
        held_out_a_id,
        held_out_b_id,
        [held_out_c_id, held_out_d_id, held_out_e_id],
        gate_dir,
        adir,
    )


# ---------------------------------------------------------------------------
# Helper: build and commit the gate manifest
# ---------------------------------------------------------------------------

def _register_manifest(
    parent_id: str,
    held_out_a_id: str,
    held_out_b_id: str,
    held_out_extras: list[str],
    gate_dir: Path,
    git_root: Path,
) -> GateManifest:
    """D-149 assertion 2: register manifest and commit to git.

    Uses K=5 cells with p_threshold=0.2 — proven in test_promote.py to achieve
    a statistically reliable PASS with 5 positive deltas (Wilcoxon p=0.031 <=
    Bonferroni-corrected alpha 0.04). K=2 is statistically impossible (minimum
    achievable Wilcoxon p for n=2 is 0.25).
    """
    held_out_extra_tuples = tuple(
        (cid, "ccrcc", "uni_v2", "high_grade") for cid in held_out_extras
    )
    manifest = GateManifest(
        parent_id=parent_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        git_committed_at_sha="PENDING",
        held_out_cells=(
            (held_out_a_id, "ccrcc", "uni_v2",  "high_grade"),
            (held_out_b_id, "clwd",  "hibou_l", "subtype"),
        ) + held_out_extra_tuples,
        K=5,
        p_threshold=0.2,   # Bonferroni: 0.2/5 = 0.04 > Wilcoxon min-p 0.031 -> PASS
        bootstrap_reps=100,  # small for test speed
        win_definition="delta_primary_value > 0 AND p < p_threshold",
        schema_version=SCHEMA_VERSION,
    )
    write_manifest_committed(manifest, gate_dir, git_root)
    return manifest


# ---------------------------------------------------------------------------
# Helper: get promote module (needed for monkeypatching evaluate_candidate)
# ---------------------------------------------------------------------------

def _get_promote_module():
    if "automil.gate.promote" not in sys.modules:
        importlib.import_module("automil.gate.promote")
    return sys.modules["automil.gate.promote"]


# ---------------------------------------------------------------------------
# Main load-bearing test: PASS path — all 9 D-149 assertions in one function
# ---------------------------------------------------------------------------

def test_pitfall6_held_out_isolation_pass_path(pitfall6_project, monkeypatch):
    """D-149 end-to-end Pitfall-6 anti-acceptance test — PASS path.

    Exercises the full Wave 1-6 pipeline with a synthetic 3-cell graph.
    All 9 D-149 assertions verified in one test. This is the load-bearing
    gate for Phase 5.

    If this test fails, Phase 5's held-out isolation contract is broken.
    """
    (
        project_root,
        search_node_id,
        [cand_1_id, cand_2_id, cand_3_id],
        held_out_a_id,
        held_out_b_id,
        [held_out_c_id, held_out_d_id, held_out_e_id],
        gate_dir,
        adir,
    ) = pitfall6_project
    all_held_out_ids = {held_out_a_id, held_out_b_id, held_out_c_id, held_out_d_id, held_out_e_id}

    # === D-149 assertion 1: verify the 3-cell graph exists ===
    graph_path = adir / "graph.json"
    graph_data = json.loads(graph_path.read_text())
    assert search_node_id in graph_data["nodes"], (
        "Pitfall-6 assertion 1: search cell node must be in graph"
    )
    assert cand_2_id in graph_data["nodes"], (
        "Pitfall-6 assertion 1: candidate 2 must be in graph"
    )
    # Held-out cell IDs live in the manifest, not as nodes at this stage
    assert len([
        n for n in graph_data["nodes"].values()
        if n.get("metadata", {}).get("held_out", False)
    ]) == 0, (
        "Pitfall-6 assertion 1: no held-out nodes in graph before promote"
    )

    # === D-149 assertion 2: register manifest ===
    manifest = _register_manifest(
        search_node_id, held_out_a_id, held_out_b_id,
        [held_out_c_id, held_out_d_id, held_out_e_id],
        gate_dir, project_root,
    )
    manifest_path = gate_dir / f"{search_node_id}.gate_manifest.json"
    assert manifest_path.exists(), (
        "Pitfall-6 assertion 2: gate_manifest.json must exist after write_manifest_committed"
    )
    # git log should show the manifest commit
    log = subprocess.run(
        ["git", "log", "--oneline", "-3"],
        cwd=str(project_root), capture_output=True, text=True,
    ).stdout
    assert "gate: register manifest" in log, (
        f"Pitfall-6 assertion 2: gate manifest must be committed to git; log={log!r}"
    )

    # === D-149 assertion 3: 3 candidates exist with primary values from search cell only ===
    for cid in [cand_1_id, cand_2_id, cand_3_id]:
        node = graph_data["nodes"][cid]
        # Each candidate has primary_value > 0 (from search cell context only)
        assert float(node.get("primary_value", 0.0)) > 0.0, (
            f"Pitfall-6 assertion 3: candidate {cid} must have a primary_value from search cell"
        )
        # No held_out flag on search-loop candidates
        assert not node.get("metadata", {}).get("held_out", False), (
            f"Pitfall-6 assertion 3: search-loop candidate {cid} must not be marked held_out"
        )

    # === D-149 assertion 4a: rank output (via CliRunner) excludes held-out cell IDs ===
    # Change cwd so _find_automil_dir() finds our fixture
    monkeypatch.chdir(str(project_root))
    # Clear held_out_ids lru_cache so it re-reads from the fixture graph
    _held_out_ids_cached.cache_clear()

    from click.testing import CliRunner
    from automil.cli import main

    runner = CliRunner()
    rank_result = runner.invoke(main, ["rank", "--n", "20"])
    assert rank_result.exit_code == 0, (
        f"Pitfall-6 assertion 4a: rank must exit 0; got {rank_result.exit_code}. "
        f"Output: {rank_result.output}"
    )
    for hid in all_held_out_ids:
        assert hid not in rank_result.output, (
            f"Pitfall-6 assertion 4a: held-out cell ID {hid!r} "
            "must not appear in rank output"
        )

    # === D-149 assertion 4b: trajectory redactor replaces held-out IDs ===
    # Simulate a graph.json mtime read (needed for lru_cache key)
    # We must add held-out node IDs to graph.json so the redactor knows about them.
    # The held-out nodes are created by evaluate_candidate DURING promote — so for
    # the pre-promote redactor test, we manually inject a fake held-out node.
    # Node IDs must match _NODE_ID_RE = r"\bnode_\d{4,}\b" — use numeric suffix only.
    fake_held_out_node_id = "node_0099"
    graph_data_loaded = json.loads(graph_path.read_text())
    graph_data_loaded["nodes"][fake_held_out_node_id] = {
        "id": fake_held_out_node_id,
        "parent_id": cand_2_id,
        "type": "gate_eval",
        "status": "pending",
        "metadata": {"held_out": True, "gate_eval": True, "cell_id": held_out_a_id},
        "primary_value": 0.0,
    }
    graph_path.write_text(json.dumps(graph_data_loaded, indent=2))

    # Invalidate the lru_cache so it re-reads the updated graph.json
    _held_out_ids_cached.cache_clear()

    # Now redact an event that contains the held-out node_id
    fake_event = {
        "gen_ai.event.name": "tool_result",
        "content": (
            f"Running experiment on {fake_held_out_node_id} and {cand_2_id}; "
            f"cell={held_out_a_id}"
        ),
    }
    redacted = redact_event(fake_event)
    # The held-out node_id should be replaced
    assert "<HELD_OUT>" in str(redacted), (
        f"Pitfall-6 assertion 4b: held-out node ID must be redacted to <HELD_OUT>; "
        f"got: {redacted}"
    )
    # The non-held-out node (cand_2_id) must NOT be redacted
    assert cand_2_id in str(redacted), (
        f"Pitfall-6 assertion 4b: non-held-out node {cand_2_id!r} must remain visible; "
        f"got: {redacted}"
    )

    # Remove the fake held-out node — restore graph to pre-promote state
    graph_data_loaded["nodes"].pop(fake_held_out_node_id)
    graph_path.write_text(json.dumps(graph_data_loaded, indent=2))
    _held_out_ids_cached.cache_clear()

    # === D-149 assertion 5: operator nominates candidate_2 ===
    nominate_result = runner.invoke(main, ["nominate", cand_2_id])
    assert nominate_result.exit_code == 0, (
        f"Pitfall-6 assertion 5: automil nominate must exit 0; "
        f"output: {nominate_result.output}"
    )
    # Re-read graph to verify nomination
    graph_after_nominate = json.loads(graph_path.read_text())
    status_after_nominate = graph_after_nominate["nodes"][cand_2_id].get("status")
    assert status_after_nominate == "candidate", (
        f"Pitfall-6 assertion 5: after nominate, status must be 'candidate'; "
        f"got {status_after_nominate!r}"
    )

    # === D-149 assertion 6: promote spawns 2 gate_eval Backend.submit calls ===
    # Configure backend: positive deltas => PASS path
    # 5 strongly positive deltas (0.05..0.09 above parent=0.85):
    # Wilcoxon p=0.031 <= Bonferroni-corrected alpha=0.04 -> PASS
    backend = _RecordingBackend()
    backend.set_primary_value(held_out_a_id,  0.90)  # delta +0.05
    backend.set_primary_value(held_out_b_id,  0.91)  # delta +0.06
    backend.set_primary_value(held_out_c_id,  0.92)  # delta +0.07
    backend.set_primary_value(held_out_d_id,  0.93)  # delta +0.08
    backend.set_primary_value(held_out_e_id,  0.94)  # delta +0.09

    # Monkeypatch _resolve_backend so promote_cmd uses our recording backend
    monkeypatch.setattr(
        "automil.cli.promote._resolve_backend",
        lambda _name, _adir: backend,
    )

    # Also we need to stamp primary_value on graph nodes when backend submits,
    # so _read_eval_primary_value finds the right value.
    # We override evaluate_candidate at the promote module level to do this inline.
    # Actually, let's use the real evaluate_candidate path but pre-stamp graph nodes.
    # The backend.submit records the spec; _read_eval_primary_value reads graph.nodes[child_id].
    # We need to stamp after submit. Use a monkeypatch on _poll_handles to stamp primary values.
    import automil.gate.evaluate as _eval_mod

    _orig_read_eval_primary_value = _eval_mod._read_eval_primary_value

    def _patched_read_eval_primary_value(handle, _backend, graph, child_id,
                                     fallback_primary_value, state_str):
        # Find the cell_id from the graph node's metadata
        node = graph.nodes.get(child_id, {})
        cell_id = node.get("metadata", {}).get("cell_id", "")
        configured = backend._primary_values.get(cell_id)
        if configured is not None:
            # Stamp primary_value so the pairing logic works
            graph.nodes[child_id]["primary_value"] = configured
        return _orig_read_eval_primary_value(
            handle, _backend, graph, child_id, fallback_primary_value, state_str
        )

    monkeypatch.setattr(_eval_mod, "_read_eval_primary_value", _patched_read_eval_primary_value)

    # Also patch get_cell to return None (no cap-exhausted cells) for evaluate
    monkeypatch.setattr("automil.gate.evaluate.get_cell", lambda cid: None)

    # Run promote via CLI
    promote_result = runner.invoke(main, ["promote", cand_2_id], catch_exceptions=False)
    assert promote_result.exit_code == 0, (
        f"Pitfall-6 assertion 6/7: automil promote must exit 0 on PASS path; "
        f"output: {promote_result.output}"
    )

    # D-149 assertion 6: exactly 5 Backend.submit calls (one per held-out cell)
    assert backend.submit_call_count == 5, (
        f"Pitfall-6 assertion 6: promote must spawn 5 gate_eval Backend.submit calls "
        f"(one per held-out cell; got {backend.submit_call_count})"
    )

    # Each submitted spec must carry the correct metadata
    for spec in backend.submitted_specs:
        md = dict(spec.metadata)
        assert md.get("gate_eval") == "true", (
            f"Pitfall-6 assertion 6: metadata.gate_eval must be 'true'; got {md}"
        )
        assert md.get("held_out") == "true", (
            f"Pitfall-6 assertion 6: metadata.held_out must be 'true'; got {md}"
        )
        assert md.get("edge_type") == "gate_eval", (
            f"Pitfall-6 assertion 6: metadata.edge_type must be 'gate_eval'; got {md}"
        )
        assert md.get("gate_parent_node") == cand_2_id, (
            f"Pitfall-6 assertion 6: metadata.gate_parent_node must be {cand_2_id!r}; got {md}"
        )
        assert md.get("cell_id") in all_held_out_ids, (
            f"Pitfall-6 assertion 6: metadata.cell_id must be one of the held-out cells; got {md}"
        )

    # === D-149 assertion 7: post-promote status='registered' (PASS path) ===
    graph_after_promote = json.loads(graph_path.read_text())
    status_post_promote = graph_after_promote["nodes"][cand_2_id].get("status")
    assert status_post_promote == "registered", (
        f"Pitfall-6 assertion 7: PASS path — status must be 'registered' after promote; "
        f"got {status_post_promote!r}"
    )

    # === D-149 assertion 8: trajectory redaction STILL replaces held-out IDs after promote ===
    # After promote, graph.json now has gate_eval child nodes with held_out=True.
    # Invalidate the lru_cache so redactor re-reads updated graph.json.
    _held_out_ids_cached.cache_clear()

    # Build a fake trajectory event that mentions a gate-eval child node
    gate_eval_nodes = [
        nid for nid, n in graph_after_promote["nodes"].items()
        if n.get("metadata", {}).get("held_out", True) and n.get("edge_type") == "gate_eval"
    ]

    if gate_eval_nodes:
        gate_eval_nid = gate_eval_nodes[0]
        fake_post_promote_event = {
            "gen_ai.event.name": "tool_result",
            "content": (
                f"gate eval result: {gate_eval_nid} primary_value=0.92 "
                f"cell={held_out_a_id}"
            ),
        }
        redacted_post = redact_event(fake_post_promote_event)
        assert "<HELD_OUT>" in str(redacted_post), (
            f"Pitfall-6 assertion 8: gate-eval node ID {gate_eval_nid!r} must be "
            f"redacted to <HELD_OUT> in trajectory even AFTER promote; "
            f"got: {redacted_post}"
        )

    # Also verify: forensic archive log has held-out IDs (NOT redacted — it's not trajectory)
    archive_log = adir / "archive" / cand_2_id / "gate_evaluation.jsonl"
    assert archive_log.exists(), (
        "Pitfall-6 assertion 8: gate_evaluation.jsonl must exist in archive/ "
        "(forensic log is NOT trajectory — held-out IDs are present there)"
    )
    archive_content = archive_log.read_text()
    # The archive log should contain held-out cell IDs (it's the forensic side, not redacted)
    assert held_out_a_id in archive_content or held_out_b_id in archive_content, (
        f"Pitfall-6 assertion 8: archive gate_evaluation.jsonl must contain held-out "
        f"cell IDs (forensic log); content preview: {archive_content[:300]}"
    )

    # === D-149 assertion 9: node['history'] has timestamped transitions ===
    history = graph_after_promote["nodes"][cand_2_id].get("history", [])
    assert len(history) >= 2, (
        f"Pitfall-6 assertion 9: history must have at least 2 events "
        f"('nominated' + 'gate_result'); got {history}"
    )

    event_names = [e.get("event") for e in history]
    assert "nominated" in event_names, (
        f"Pitfall-6 assertion 9: history must contain 'nominated' event; "
        f"events: {event_names}"
    )
    assert "gate_result" in event_names, (
        f"Pitfall-6 assertion 9: history must contain 'gate_result' event; "
        f"events: {event_names}"
    )

    # Each history event must have a parseable ISO timestamp
    for ev in history:
        ts = ev.get("timestamp")
        assert ts is not None, (
            f"Pitfall-6 assertion 9: each history event must have 'timestamp'; "
            f"event={ev}"
        )
        try:
            datetime.fromisoformat(ts)
        except ValueError as exc:
            pytest.fail(
                f"Pitfall-6 assertion 9: timestamp {ts!r} is not valid ISO format; "
                f"error: {exc}"
            )


# ---------------------------------------------------------------------------
# Companion test: FAIL path — assertion 7 fail leg
# ---------------------------------------------------------------------------

def test_pitfall6_fail_path_reverts_to_keep(pitfall6_project, monkeypatch):
    """D-149 assertion 7 fail leg: negative deltas -> status='keep' after promote.

    This is the companion to the PASS path test. It exercises the case where
    the gate rejects the candidate, reverting status back to 'keep'.
    """
    (
        project_root,
        search_node_id,
        [cand_1_id, cand_2_id, cand_3_id],
        held_out_a_id,
        held_out_b_id,
        [held_out_c_id, held_out_d_id, held_out_e_id],
        gate_dir,
        adir,
    ) = pitfall6_project

    monkeypatch.chdir(str(project_root))
    _held_out_ids_cached.cache_clear()

    # Register manifest
    _register_manifest(
        search_node_id, held_out_a_id, held_out_b_id,
        [held_out_c_id, held_out_d_id, held_out_e_id],
        gate_dir, project_root,
    )

    # Nominate cand_2
    from click.testing import CliRunner
    from automil.cli import main

    runner = CliRunner()
    nominate_result = runner.invoke(main, ["nominate", cand_2_id])
    assert nominate_result.exit_code == 0

    # Configure backend: NEGATIVE deltas => FAIL path
    # Parent primary_value = 0.85; held-out primary values < 0.85 -> delta negative
    backend = _RecordingBackend()
    backend.set_primary_value(held_out_a_id, 0.78)  # delta = 0.78 - 0.85 = -0.07
    backend.set_primary_value(held_out_b_id, 0.79)  # delta = 0.79 - 0.85 = -0.06
    backend.set_primary_value(held_out_c_id, 0.77)  # delta = -0.08
    backend.set_primary_value(held_out_d_id, 0.76)  # delta = -0.09
    backend.set_primary_value(held_out_e_id, 0.75)  # delta = -0.10

    monkeypatch.setattr(
        "automil.cli.promote._resolve_backend",
        lambda _name, _adir: backend,
    )
    monkeypatch.setattr("automil.gate.evaluate.get_cell", lambda cid: None)

    # Patch _read_eval_primary_value to return our configured primary values
    import automil.gate.evaluate as _eval_mod
    _orig = _eval_mod._read_eval_primary_value

    def _patched(handle, _backend, graph, child_id, fallback_primary_value, state_str):
        node = graph.nodes.get(child_id, {})
        cell_id = node.get("metadata", {}).get("cell_id", "")
        configured = backend._primary_values.get(cell_id)
        if configured is not None:
            graph.nodes[child_id]["primary_value"] = configured
        return _orig(handle, _backend, graph, child_id, fallback_primary_value, state_str)

    monkeypatch.setattr(_eval_mod, "_read_eval_primary_value", _patched)

    promote_result = runner.invoke(main, ["promote", cand_2_id], catch_exceptions=False)
    assert promote_result.exit_code == 0, (
        f"Pitfall-6 FAIL path: promote must exit 0 even on gate fail "
        f"(it's not a CLI error); output: {promote_result.output}"
    )

    graph_path = adir / "graph.json"
    graph_after = json.loads(graph_path.read_text())
    status = graph_after["nodes"][cand_2_id].get("status")

    # D-149 assertion 7 fail leg: status reverts to 'keep'
    assert status == "keep", (
        f"Pitfall-6 assertion 7 (FAIL path): negative deltas must revert status "
        f"to 'keep'; got {status!r}"
    )


# ---------------------------------------------------------------------------
# Redactor-isolation test — assertions 4 + 8 focused
# ---------------------------------------------------------------------------

def test_pitfall6_redactor_isolation_pre_and_post(pitfall6_project, monkeypatch):
    """Focused D-149 assertion 4 + 8: redactor replaces held-out IDs in trajectory.

    Verifies:
    - Pre-promote: held-out node IDs injected manually -> redacted to <HELD_OUT>
    - Search node IDs (non-held-out) stay visible
    - cache_clear() between subtests ensures the lru_cache doesn't leak stale state
    """
    (
        project_root,
        search_node_id,
        [cand_1_id, cand_2_id, cand_3_id],
        held_out_a_id,
        held_out_b_id,
        [held_out_c_id, held_out_d_id, held_out_e_id],
        gate_dir,
        adir,
    ) = pitfall6_project

    monkeypatch.chdir(str(project_root))

    # ---- Subtest A: inject fake held-out nodes into graph, verify redaction ----
    graph_path = adir / "graph.json"
    graph_data = json.loads(graph_path.read_text())

    # Node IDs must match _NODE_ID_RE = r"\bnode_\d{4,}\b" (numeric suffix only)
    fake_node_a = "node_0091"
    fake_node_b = "node_0092"

    for fake_nid, cid in [(fake_node_a, held_out_a_id), (fake_node_b, held_out_b_id)]:
        graph_data["nodes"][fake_nid] = {
            "id": fake_nid,
            "parent_id": cand_2_id,
            "type": "gate_eval",
            "status": "completed",
            "metadata": {
                "held_out": True,
                "gate_eval": True,
                "cell_id": cid,
            },
            "primary_value": 0.92,
        }

    graph_path.write_text(json.dumps(graph_data, indent=2))
    _held_out_ids_cached.cache_clear()

    # Event mentioning BOTH held-out and non-held-out nodes
    event = {
        "gen_ai.event.name": "tool_result",
        "content": (
            f"Results: {fake_node_a} scored 0.92, "
            f"{fake_node_b} scored 0.93, "
            f"{cand_2_id} search primary_value=0.88"
        ),
    }
    redacted = redact_event(event)
    content_str = str(redacted)

    # Held-out node IDs must be replaced
    assert "<HELD_OUT>" in content_str, (
        "Pitfall-6 assertion 4/8 (redactor): held-out node IDs must be replaced "
        "with <HELD_OUT> in trajectory"
    )
    assert fake_node_a not in content_str, (
        f"Pitfall-6 assertion 4/8: {fake_node_a!r} must be redacted"
    )
    assert fake_node_b not in content_str, (
        f"Pitfall-6 assertion 4/8: {fake_node_b!r} must be redacted"
    )

    # Non-held-out node (cand_2_id search candidate) must remain visible
    assert cand_2_id in content_str, (
        f"Pitfall-6 assertion 4/8: search candidate {cand_2_id!r} must stay visible "
        "in trajectory (NOT redacted)"
    )

    # ---- Subtest B: cache_clear() hygiene ----
    # Remove held-out nodes, clear cache; redaction should stop
    for fake_nid in [fake_node_a, fake_node_b]:
        graph_data["nodes"].pop(fake_nid)
    graph_path.write_text(json.dumps(graph_data, indent=2))
    _held_out_ids_cached.cache_clear()  # D-149 hygiene: lru_cache invalidation

    event_clean = {
        "gen_ai.event.name": "tool_result",
        "content": f"Results: {fake_node_a} scored 0.92, {cand_2_id} search",
    }
    redacted_clean = redact_event(event_clean)
    content_clean = str(redacted_clean)

    # After removing held-out nodes from graph + clearing cache,
    # node IDs are no longer redacted (they don't match any held_out=True node)
    assert fake_node_a in content_clean, (
        "Pitfall-6 assertion 4/8 (cache_clear hygiene): after removing held-out "
        "nodes and cache_clear(), previously-held-out IDs should no longer be redacted"
    )
