"""Scope B val-firewall: test-bearing artifacts are BORN-SEALED under
archive/<node>/certify/, never the agent-visible node-archive root.

Two gates:
  A. The orchestrator points AUTOMIL_RESULTS_DIR at archive/<node>/certify/, so
     every training-side writer (per-fold writer, results/ tree, SIGTERM flush)
     lands in the sealed subdir by construction — no benchmarks/ edits required.
  B. The real completion boundary (runner.collect_result + terminal_writer) leaves
     the agent-visible node archive test-free: root result.json is val-only, no
     fold_*_result.json / results/ at root, and every test-bearing artifact
     (held_out, per-fold files, results/, the raw result.json) is under certify/.

Gate B exercises the actual production functions, not a hand-rolled stand-in, so a
regression that reintroduces a test leak into an agent-facing surface fails here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _make_orch(tmp_path: Path) -> Any:
    """Minimal ExperimentOrchestrator over an isolated automil/ overlay."""
    from automil.orchestrator import ExperimentOrchestrator

    automil_dir = tmp_path / "automil"
    automil_dir.mkdir(parents=True, exist_ok=True)
    (automil_dir / "config.yaml").write_text("orchestrator: {}\n")
    (tmp_path / ".git").mkdir(exist_ok=True)
    return ExperimentOrchestrator(project_root=tmp_path, automil_dir=automil_dir)


def test_subprocess_env_points_results_dir_at_certify(tmp_path: Path) -> None:
    """Gate A: AUTOMIL_RESULTS_DIR is archive/<node>/certify/ and is created eagerly.

    This one env value is what makes fold_*_result.json, the results/ tree, and the
    SIGTERM flush born-sealed — they all derive from it with no writer-side changes.
    """
    orch = _make_orch(tmp_path)
    node_id = "node_0001"
    archive = orch.archive_dir / node_id

    env = orch._build_subprocess_env(
        gpu_id=0, node_id=node_id, archive=archive, spec={"description": "d"},
    )

    sealed = (archive / "certify").resolve()
    assert env["AUTOMIL_RESULTS_DIR"] == str(sealed), (
        "AUTOMIL_RESULTS_DIR must point at the sealed certify/ subdir so training "
        f"artifacts are born-sealed; got {env['AUTOMIL_RESULTS_DIR']!r}"
    )
    assert sealed.is_dir(), "the sealed dir must be created at env-build time"
    # It must NOT be the agent-visible node-archive root.
    assert env["AUTOMIL_RESULTS_DIR"] != str(archive.resolve())


def test_completion_leaves_agent_archive_test_free(tmp_path: Path) -> None:
    """Gate B: after the real completion boundary, no test leaks to any agent surface.

    Simulates the on-disk state of a born-sealed run, then drives the production
    completion functions (runner.collect_result + terminal_writer.write_terminal_state)
    and asserts the firewall property end-to-end.
    """
    from automil.runner import Runner
    from automil.terminal_writer import write_terminal_state
    from automil.graph import ExperimentGraph

    project = tmp_path
    (project / ".git").mkdir(exist_ok=True)
    node_id = "node_0001"
    archive = project / "orchestrator" / "archive" / node_id
    archive.mkdir(parents=True)
    completed = project / "orchestrator" / "completed"
    completed.mkdir(parents=True)

    # --- Simulate a born-sealed run on disk -------------------------------------
    # 1. The training script wrote result.json to its worktree with BOTH the val
    #    metrics (agent-facing) and the sealed held_out (test) block.
    runner = Runner(project_root=project, automil_dir=project / "automil")
    wt = runner.worktree_path(node_id)
    wt.mkdir(parents=True, exist_ok=True)
    raw_result = {
        "status": "completed",
        "metrics": {"val_auc": 0.88, "val_bacc": 0.81},
        "held_out": {"test_auc": 0.87, "test_bacc": 0.83},
        "primary_value": 0.845,
        "elapsed_seconds": 100,
        "peak_vram_mb": 4000,
    }
    (wt / "result.json").write_text(json.dumps(raw_result))
    # 2. Per-fold files + the results/ detail tree were born-sealed under certify/
    #    (AUTOMIL_RESULTS_DIR pointed there during the run).
    sealed = archive / "certify"
    sealed.mkdir(parents=True, exist_ok=True)
    (sealed / "fold_0_result.json").write_text(
        json.dumps({"primary_value": 0.845, "metrics": {"val_auc": 0.88},
                    "held_out": {"test_auc": 0.87}})
    )
    (sealed / "results").mkdir()
    (sealed / "results" / "summary.json").write_text(json.dumps({"test": {"auc": 0.87}}))

    # --- Run the REAL completion boundary ---------------------------------------
    collected = runner.collect_result(wt, archive)          # B1: raw copy -> certify/
    assert collected is not None and "held_out" in collected, (
        "collect_result must return the raw dict (held_out intact) for terminal_writer"
    )

    graph = ExperimentGraph(path=project / "graph.json")
    generated_id = graph.add_executed(
        parent_id=None, description="baseline", techniques=["baseline"],
        metrics={"primary_value": 0.0}, status="keep",
    )
    assert generated_id == node_id  # first executed node is node_0001
    graph.save()  # terminal_writer's locked_update re-reads the graph from disk

    tsv_rows: list = []

    def _tsv_writer(nid, result, description=""):
        tsv_rows.append((nid, result, description))

    write_terminal_state(
        node_id=node_id, result=collected, graph=graph,
        completed_dir=completed, archive_dir=archive,
        results_tsv_writer=_tsv_writer, spec={"description": "baseline"},
        elapsed_s=100.0, gpu_id=0,
    )

    # --- Firewall assertions ----------------------------------------------------
    # (1) Agent-visible node-archive root: val-only result.json, no test-bearing files.
    root_result = json.loads((archive / "result.json").read_text())
    assert "held_out" not in root_result, "held_out leaked into the agent-visible result.json"
    assert not any("test" in k for k in root_result.get("metrics", {})), (
        f"test-keyed metric leaked into root result.json: {root_result.get('metrics')}"
    )
    assert not list(archive.glob("fold_*_result.json")), "fold files leaked to node-archive root"
    assert not (archive / "results").exists(), "results/ tree leaked to node-archive root"

    # (2) Sealed vault holds every test-bearing artifact.
    assert (sealed / "certify.json").exists()
    certify = json.loads((sealed / "certify.json").read_text())
    assert certify["held_out"]["test_auc"] == 0.87
    assert (sealed / "result.json").exists(), "raw result.json (with held_out) must be sealed (B1)"
    assert "held_out" in json.loads((sealed / "result.json").read_text())
    assert (sealed / "fold_0_result.json").exists()
    assert (sealed / "results" / "summary.json").exists()

    # (3) Other agent-facing surfaces carry no test.
    completion = json.loads((completed / f"{node_id}.json").read_text())
    assert "held_out" not in completion
    assert not any("test" in k for k in completion.get("metrics", {}))
    assert tsv_rows, "results.tsv writer was not invoked"
    assert "held_out" not in tsv_rows[0][1], "held_out leaked into the results.tsv payload"

    # (4) The graph node stores val primary_value + val metrics only.
    gnode = ExperimentGraph(path=project / "graph.json").get_node(node_id)
    assert gnode["primary_value"] == 0.845
    assert not any("test" in k for k in gnode.get("metrics", {})), (
        f"test-keyed metric leaked into the graph node: {gnode.get('metrics')}"
    )
