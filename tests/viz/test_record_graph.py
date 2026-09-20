"""viz.record_graph: validation-only projection and the verdict."""
from __future__ import annotations

from pathlib import Path

from automil.graph import effective_accept_margin, guard_basis, keep_or_discard, margin_se_basis
from automil.viz.clock import host_clock
from automil.viz.record_files import (
    RunPaths,
    load_gpu_state,
    load_run_log_tail,
    load_overlay_file,
    load_spec,
    overlay_files,
)
from automil.viz.record_graph import (
    node_verdict,
    project_graph,
    project_node,
    project_result,
    running_node_ids,
    verdict_unavailable,
    walk_keys,
)

from tests.viz.conftest import graph_payload, write_project

CLOCK = host_clock(tz_name="America/Vancouver")


def test_project_node_converts_created_at_and_flags_held_out():
    node = graph_payload()["nodes"]["node_0002"]
    projected = project_node(node, CLOCK)
    assert projected["created_at"] == "2026-09-07T00:30:28.759Z"
    assert projected["gate_child"] is False
    assert projected["metrics"] == node["metrics"]
    gate_child = dict(node, metadata={"held_out": True, "gate_eval": True})
    assert project_node(gate_child, CLOCK)["gate_child"] is True


def test_project_graph_applies_the_running_overlay_and_keeps_meta():
    raw = graph_payload()
    graph = project_graph(raw, CLOCK, running=("node_0003", "node_0009"))
    assert graph["nodes"]["node_0003"]["status"] == "running"
    assert graph["nodes"]["node_0002"]["status"] == "discard"
    assert graph["running"] == ["node_0003", "node_0009"]
    assert graph["meta"]["best_node_id"] == "node_0003"
    assert graph["technique_stats"]["dropout"]["times_tried"] == 2
    assert graph["schema"] == 1


def test_running_ids_read_typed_slots_and_legacy_gpus():
    typed = {"execution_slots": {"cuda:0": {"running": ["node_0002"]}, "cpu:0": {"running": ["node_0007"]}}}
    assert running_node_ids(typed) == ("node_0002", "node_0007")
    assert running_node_ids({"gpus": {"0": {"running": ["node_0003"]}}}) == ("node_0003",)
    assert running_node_ids(None) == () and running_node_ids({"gpus": "junk"}) == ()


def test_verdict_matches_the_graph_helpers_on_the_same_dicts():
    raw = graph_payload()
    meta, nodes = raw["meta"], raw["nodes"]
    parent, kept, lost = nodes["node_0001"], nodes["node_0003"], nodes["node_0002"]
    verdict = node_verdict(meta, parent, kept)
    assert verdict["decision"] == keep_or_discard(meta, parent, kept) == "keep"
    assert verdict["consistent"] is True
    assert verdict["bar"] == effective_accept_margin(meta, parent, kept)
    assert (verdict["basis"], verdict["basis_se"]) == margin_se_basis(meta, parent, kept)
    assert verdict["basis"] == "paired"
    g_verdict, g_delta, g_metric, g_margin = guard_basis(meta, parent, kept)
    assert verdict["guard"] == {"verdict": g_verdict, "delta": g_delta, "metric": g_metric, "margin": g_margin, "decisive": False}
    assert verdict["explanation"].startswith("Kept: +0.0533 over the parent, above the bar")
    assert verdict["accept_margin"] == 0.015 and verdict["se_multiplier"] == 1.0

    lost_verdict = node_verdict(meta, parent, lost)
    assert lost_verdict["decision"] == "discard" and lost_verdict["consistent"] is True
    assert lost_verdict["explanation"].startswith("Discarded:")


def test_verdict_reports_a_decisive_guard_and_inconsistency():
    raw = graph_payload()
    meta, nodes = raw["meta"], raw["nodes"]
    parent = nodes["node_0001"]
    vetoed = dict(nodes["node_0003"], metrics={"val_auc": 0.66, "val_bacc": 0.40})
    verdict = node_verdict(meta, parent, vetoed)
    assert verdict["decision"] == "discard"
    assert verdict["guard"]["verdict"] == "fail" and verdict["guard"]["decisive"] is True
    assert "The guard decided." in verdict["explanation"]
    assert verdict["consistent"] is False  # stored status says keep


def test_verdict_is_unavailable_for_root_partial_crash_and_proposals():
    raw = graph_payload()
    nodes = raw["nodes"]
    root = node_verdict(raw["meta"], None, nodes["node_0001"])
    assert root["basis"] == "none" and root["decision"] == "keep" and root["parent_id"] is None
    assert verdict_unavailable(nodes["node_0004"]) == "crash"
    assert verdict_unavailable(nodes["node_0005"]) == "not executed"
    assert verdict_unavailable(dict(nodes["node_0002"], status="partial")) == "partial"
    assert verdict_unavailable(dict(nodes["node_0002"], status="running")) == "not terminal"
    assert node_verdict(raw["meta"], nodes["node_0001"], nodes["node_0004"]) is None


def test_project_result_strips_sealed_blocks_and_converts_completed_at():
    result = {
        "status": "completed", "metrics": {"val_auc": 0.6, "test_auc": 0.9}, "held_out": {"test_auc": 0.9},
        "summary": {"x": 1}, "validation_folds": [{"fold_index": 0, "metrics": {"val_auc": 0.6, "test_bacc": 0.8}}],
        "completed_at": "2026-09-06T19:12:51.003299", "error": "ok line\ntest_auc=0.9 line",
    }
    projected = project_result(result, CLOCK)
    assert "held_out" not in projected and "summary" not in projected
    assert projected["metrics"] == {"val_auc": 0.6}
    assert projected["validation_folds"][0]["metrics"] == {"val_auc": 0.6}
    assert projected["completed_at"] == "2026-09-07T02:12:51.003Z"
    assert "0.9" not in projected["error"] and "ok line" in projected["error"]
    assert project_result(None, CLOCK) is None


def test_loaders_read_the_archive_without_listing_it(tmp_path: Path):
    automil = write_project(tmp_path)
    paths = RunPaths(automil)
    spec = load_spec(paths, "node_0003")
    assert overlay_files(spec) == [{"path": "automil/variants/_policies/dropout.py", "sha256": "ab" * 32}]
    assert overlay_files(load_spec(paths, "node_0002")) == []
    file = load_overlay_file(paths, "node_0003", "automil/variants/_policies/dropout.py")
    assert file == {"path": "automil/variants/_policies/dropout.py", "text": "DROPOUT = 0.5\n", "truncated": False}
    assert load_overlay_file(paths, "node_0003", "../node_0002/result.json") is None
    assert load_overlay_file(paths, "node_0003", "certify/certify.json") is None
    assert load_overlay_file(paths, "node_0003", "/etc/passwd") is None
    tail = load_run_log_tail(paths, "node_0002", lines=2)
    assert tail["tail"] == ["[selected] epoch=2", "Experiment complete"] and tail["n_lines_total"] == 3
    assert running_node_ids(load_gpu_state(paths)) == ()
    for bad in ("node_1", "../x", "node_0002/certify"):
        try:
            paths.node_archive(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} accepted as a node id")


def test_walk_keys_visits_nested_payloads():
    payload = {"a": {"b": [1, {"c": 2}]}, "d": 3}
    assert sorted(k for _, k in walk_keys(payload)) == ["a", "b", "c", "d"]
