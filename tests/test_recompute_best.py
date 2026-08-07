"""Coverage for ExperimentGraph.recompute_best + automil reconcile --recompute-best (CLI-07).

Locked decisions enforced verbatim:
- D-10: walk only nodes where type == "executed" AND status == "keep".
- D-11: composite formula = existing per-node `composite` field (Phase 0 does not
  redefine; Phase 8/DEC-04 owns).
- D-12: tie-break is lexicographic min on node_id.
- D-13: --dry-run prints same line, does NOT write graph.json. Output uses literal
  Unicode `→` arrow (NOT ASCII `->`).
- D-14: existing `automil reconcile` (no flag) behaviour unchanged.
- D-19: `meta.best_node_id` and `meta.best_composite` are the only mutation targets.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from automil.graph import ExperimentGraph


def _make_graph(graph_dir: Path, nodes: list[dict]) -> ExperimentGraph:
    """Build a graph at <graph_dir>/graph.json with the supplied node specs.

    Each spec dict has keys: id, type, status, composite (and optionally others).
    Returns an ExperimentGraph loaded from the freshly-written file.
    """
    graph_dir.mkdir(parents=True, exist_ok=True)
    path = graph_dir / "graph.json"
    data = {
        "schema_version": 1,
        "meta": {
            "best_composite": 0.0,
            "best_node_id": None,
            "total_executed": 0,
            "total_proposed": 0,
            "next_id": 1,
            "baseline_composite": 0.0,
            "scoring": {"exploration_weight": 0.005, "novelty_weight": 0.003},
        },
        "nodes": {},
        "technique_stats": {},
    }
    for n in nodes:
        data["nodes"][n["id"]] = {
            "id": n["id"],
            "type": n["type"],
            "status": n["status"],
            "composite": n.get("composite", 0.0),
            "metrics": n.get("metrics", {}),
            "techniques": n.get("techniques", []),
        }
    path.write_text(json.dumps(data, indent=2))
    return ExperimentGraph.load(path)


# --- Unit tests for ExperimentGraph.recompute_best ---


def test_basic_walk_picks_max_keep(tmp_path):
    g = _make_graph(tmp_path, [
        {"id": "node_0001", "type": "executed", "status": "keep", "composite": 0.70},
        {"id": "node_0002", "type": "executed", "status": "keep", "composite": 0.85},
        {"id": "node_0003", "type": "executed", "status": "keep", "composite": 0.60},
    ])
    old_id, old_c, new_id, new_c = g.recompute_best()
    assert new_id == "node_0002"
    assert new_c == pytest.approx(0.85)
    assert g.meta["best_node_id"] == "node_0002"
    assert g.meta["best_composite"] == pytest.approx(0.85)


def test_excludes_non_keep_status(tmp_path):
    g = _make_graph(tmp_path, [
        {"id": "node_0001", "type": "executed", "status": "keep", "composite": 0.70},
        {"id": "node_0002", "type": "executed", "status": "discard", "composite": 0.95},
        {"id": "node_0003", "type": "executed", "status": "crash", "composite": 0.99},
        {"id": "node_0004", "type": "executed", "status": "cancelled", "composite": 0.98},
    ])
    _, _, new_id, new_c = g.recompute_best()
    assert new_id == "node_0001"
    assert new_c == pytest.approx(0.70)


def test_excludes_proposed_type(tmp_path):
    g = _make_graph(tmp_path, [
        {"id": "node_0001", "type": "proposed", "status": "pending", "composite": 0.99},
        {"id": "node_0002", "type": "executed", "status": "keep", "composite": 0.50},
    ])
    _, _, new_id, _ = g.recompute_best()
    assert new_id == "node_0002"


def test_lex_tiebreak_picks_min_node_id(tmp_path):
    """D-12: equal composites resolve to the lexicographically smaller node_id."""
    g = _make_graph(tmp_path, [
        {"id": "node_0125", "type": "executed", "status": "keep", "composite": 0.80},
        {"id": "node_0048", "type": "executed", "status": "keep", "composite": 0.80},
    ])
    _, _, new_id, new_c = g.recompute_best()
    assert new_id == "node_0048"
    assert new_c == pytest.approx(0.80)


def test_no_keep_nodes_resets_meta(tmp_path):
    g = _make_graph(tmp_path, [
        {"id": "node_0001", "type": "proposed", "status": "pending", "composite": 0.0},
        {"id": "node_0002", "type": "executed", "status": "discard", "composite": 0.95},
    ])
    # Pre-existing meta.best from some old state.
    g.meta["best_node_id"] = "node_0002"
    g.meta["best_composite"] = 0.95
    _, _, new_id, new_c = g.recompute_best()
    assert new_id is None
    assert new_c == 0.0
    assert g.meta["best_node_id"] is None
    assert g.meta["best_composite"] == 0.0


def test_already_correct_is_idempotent(tmp_path):
    g = _make_graph(tmp_path, [
        {"id": "node_0001", "type": "executed", "status": "keep", "composite": 0.70},
        {"id": "node_0002", "type": "executed", "status": "keep", "composite": 0.85},
    ])
    g.meta["best_node_id"] = "node_0002"
    g.meta["best_composite"] = 0.85
    old_id, old_c, new_id, new_c = g.recompute_best()
    assert old_id == new_id == "node_0002"
    assert old_c == pytest.approx(new_c)


def test_recompute_best_does_not_save(tmp_path):
    """recompute_best mutates in-memory only; caller decides whether to save."""
    g = _make_graph(tmp_path, [
        {"id": "node_0001", "type": "executed", "status": "keep", "composite": 0.85},
    ])
    before = json.loads((tmp_path / "graph.json").read_text())
    g.recompute_best()
    after = json.loads((tmp_path / "graph.json").read_text())
    # File contents unchanged because we did not call save().
    assert before == after


# --- CLI integration tests ---


def _setup_cli_env(tmp_path: Path) -> tuple[Path, Path]:
    """Create the minimal automil/ skeleton + graph.json so the CLI invocation works.

    Returns (automil_dir, graph_path).
    """
    automil_dir = tmp_path / "automil"
    automil_dir.mkdir()
    (automil_dir / "config.yaml").write_text("orchestrator: {}\n")
    (tmp_path / ".git").mkdir()
    nodes = [
        {"id": "node_0001", "type": "executed", "status": "keep", "composite": 0.70},
        {"id": "node_0002", "type": "executed", "status": "keep", "composite": 0.85},
    ]
    _make_graph(automil_dir, nodes)  # writes <automil_dir>/graph.json
    return automil_dir, automil_dir / "graph.json"


def test_cli_recompute_best_writes(tmp_path, monkeypatch):
    automil_dir, graph_path = _setup_cli_env(tmp_path)
    monkeypatch.chdir(tmp_path)
    # Pre-set meta.best to something incorrect so we can detect the write.
    data = json.loads(graph_path.read_text())
    data["meta"]["best_node_id"] = "node_0001"
    data["meta"]["best_composite"] = 0.70
    graph_path.write_text(json.dumps(data, indent=2))

    from automil.cli import main
    runner = CliRunner()
    result = runner.invoke(main, ["reconcile", "--recompute-best"])
    assert result.exit_code == 0, result.output
    assert "node_0002" in result.output
    # Reload graph: should now be node_0002.
    reloaded = json.loads(graph_path.read_text())
    assert reloaded["meta"]["best_node_id"] == "node_0002"
    assert reloaded["meta"]["best_composite"] == pytest.approx(0.85)


def test_cli_recompute_best_dry_run_does_not_write(tmp_path, monkeypatch):
    automil_dir, graph_path = _setup_cli_env(tmp_path)
    monkeypatch.chdir(tmp_path)
    data = json.loads(graph_path.read_text())
    data["meta"]["best_node_id"] = "node_0001"  # incorrect
    data["meta"]["best_composite"] = 0.70
    graph_path.write_text(json.dumps(data, indent=2))
    before_mtime = graph_path.stat().st_mtime
    time.sleep(0.05)  # ensure mtime would actually advance if rewritten

    from automil.cli import main
    runner = CliRunner()
    result = runner.invoke(main, ["reconcile", "--recompute-best", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "node_0002" in result.output
    # File NOT rewritten — mtime unchanged.
    after_mtime = graph_path.stat().st_mtime
    assert before_mtime == after_mtime
    # Graph contents still show the wrong best.
    reloaded = json.loads(graph_path.read_text())
    assert reloaded["meta"]["best_node_id"] == "node_0001"


def test_cli_output_format_changed(tmp_path, monkeypatch):
    """D-13 verbatim format check for the changed-best case (Unicode arrow only).

    The locked decision specifies the literal Unicode `→` (U+2192). ASCII
    `->` is NOT an accepted fallback — silently weakening the decision is
    forbidden.
    """
    automil_dir, graph_path = _setup_cli_env(tmp_path)
    monkeypatch.chdir(tmp_path)
    data = json.loads(graph_path.read_text())
    data["meta"]["best_node_id"] = "node_0001"
    data["meta"]["best_composite"] = 0.70
    graph_path.write_text(json.dumps(data, indent=2))

    from automil.cli import main
    runner = CliRunner()
    result = runner.invoke(main, ["reconcile", "--recompute-best", "--dry-run"])
    pattern_unicode = (
        r"best_node_id: \S+ \(composite \d+\.\d{6}\) → "
        r"\S+ \(composite \d+\.\d{6}\)"
    )
    assert re.search(pattern_unicode, result.output), (
        f"Output did not match D-13 verbatim Unicode format:\n{result.output!r}"
    )
    # Belt-and-braces: the literal arrow character must be present in the stream.
    assert "→" in result.output, (
        f"Unicode → (U+2192) missing from output:\n{result.output!r}"
    )


def test_cli_output_format_unchanged(tmp_path, monkeypatch):
    """D-13 verbatim format check for the unchanged case."""
    automil_dir, graph_path = _setup_cli_env(tmp_path)
    monkeypatch.chdir(tmp_path)
    # Pre-set meta.best CORRECTLY so recompute is a no-op.
    data = json.loads(graph_path.read_text())
    data["meta"]["best_node_id"] = "node_0002"
    data["meta"]["best_composite"] = 0.85
    graph_path.write_text(json.dumps(data, indent=2))

    from automil.cli import main
    runner = CliRunner()
    result = runner.invoke(main, ["reconcile", "--recompute-best", "--dry-run"])
    assert re.search(
        r"best_node_id unchanged: \S+ \(composite \d+\.\d{6}\)",
        result.output,
    ), f"Output did not match D-13 unchanged format:\n{result.output!r}"


def test_cli_reconcile_without_flag_unchanged(tmp_path, monkeypatch):
    """D-14: reconcile WITHOUT --recompute-best behaves as before.

    The existing orchestrator-state-sync code path must not call recompute_best,
    so a wrong meta.best_node_id stays wrong after invocation.
    """
    automil_dir, graph_path = _setup_cli_env(tmp_path)
    monkeypatch.chdir(tmp_path)
    # Pre-set wrong best.
    data = json.loads(graph_path.read_text())
    data["meta"]["best_node_id"] = "node_0001"  # incorrect
    graph_path.write_text(json.dumps(data, indent=2))

    from automil.cli import main
    runner = CliRunner()
    result = runner.invoke(main, ["reconcile"])
    assert result.exit_code == 0, result.output
    # Best should still be wrong because --recompute-best wasn't passed.
    reloaded = json.loads(graph_path.read_text())
    assert reloaded["meta"]["best_node_id"] == "node_0001"


def test_candidate_and_registered_stay_in_the_best_walk(tmp_path):
    """B5 (claims-alignment): nominating a node (status keep -> candidate) or
    passing the gate (-> registered) must not evict it from best_node."""
    from automil.graph import ExperimentGraph

    g = ExperimentGraph(path=tmp_path / "graph.json")
    low = g.add_executed(parent_id=None, description="low", techniques=[],
                         metrics={"composite": 0.5}, status="keep")
    high = g.add_executed(parent_id=None, description="high", techniques=[],
                          metrics={"composite": 0.9}, status="keep")
    g.recompute_best()
    assert g.meta["best_node_id"] == high

    for better_state in ("candidate", "registered"):
        g.nodes[high]["status"] = better_state
        g.recompute_best()
        assert g.meta["best_node_id"] == high, (
            f"{better_state} is a better-validated keep-state; best must not "
            f"fall back to the lower keep node {low}"
        )
