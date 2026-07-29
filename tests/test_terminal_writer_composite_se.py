"""CR-4 (audit 2026-07-23): ``composite_se`` must survive the whole write path.

The noise-derived keep-margin is only real if the SE actually reaches the graph
node — the margin a child faces is derived from its PARENT's stored SE, and the
parent's SE is written by ``terminal_writer`` at the parent's own completion.

Also pinned here: ``composite_se`` stays OUT of ``metrics``. CR-1b recomputes the
selection composite as the mean of ``metrics``' values, so an SE smuggled in
there would silently corrupt the val-firewall's selection signal.
"""
from __future__ import annotations

import json
from pathlib import Path


def _noop_tsv(nid, result, description=""):
    return None


def _dirs(tmp_path: Path, node_id: str):
    completed_dir = tmp_path / "completed"
    completed_dir.mkdir(exist_ok=True)
    archive_dir = tmp_path / "archive" / node_id
    archive_dir.mkdir(parents=True, exist_ok=True)
    return completed_dir, archive_dir


def _write(tmp_path: Path, graph, node_id: str, result: dict):
    from automil.terminal_writer import write_terminal_state
    completed_dir, archive_dir = _dirs(tmp_path, node_id)
    write_terminal_state(
        node_id=node_id,
        result=result,
        graph=graph,
        completed_dir=completed_dir,
        archive_dir=archive_dir,
        results_tsv_writer=_noop_tsv,
        spec={"description": "d", "graph_metadata": {}},
        elapsed_s=1.0,
        gpu_id=0,
    )
    return completed_dir, archive_dir


def test_composite_se_lands_on_the_graph_node(tmp_path: Path) -> None:
    from automil.graph import ExperimentGraph

    graph = ExperimentGraph(path=str(tmp_path / "graph.json"))
    node_id = graph.add_executed(parent_id=None, description="root", techniques=[],
                                 metrics={"composite": 0.50}, status="keep")
    graph.nodes[node_id]["status"] = "running"
    graph.save()

    _write(tmp_path, graph, node_id, {
        "status": "completed", "composite": 0.80,
        "metrics": {"val_auc": 0.82, "val_bacc": 0.78},
        "composite_se": 0.041,
    })

    reloaded = ExperimentGraph(path=str(tmp_path / "graph.json"))
    node = reloaded.get_node(node_id)
    assert node["composite_se"] == 0.041
    # CR-1b guard: the SE must NOT be inside metrics (it would enter the mean).
    assert "composite_se" not in node["metrics"]
    assert node["composite"] == 0.80   # (0.82 + 0.78) / 2 — unchanged by the SE


def test_missing_composite_se_records_none_not_zero(tmp_path: Path) -> None:
    from automil.graph import ExperimentGraph

    graph = ExperimentGraph(path=str(tmp_path / "graph.json"))
    node_id = graph.add_executed(parent_id=None, description="root", techniques=[],
                                 metrics={"composite": 0.50}, status="keep")
    graph.nodes[node_id]["status"] = "running"
    graph.save()

    _write(tmp_path, graph, node_id, {
        "status": "completed", "composite": 0.80,
        "metrics": {"val_auc": 0.82, "val_bacc": 0.78},
    })

    node = ExperimentGraph(path=str(tmp_path / "graph.json")).get_node(node_id)
    assert node.get("composite_se") is None


def test_completed_artifact_carries_composite_se(tmp_path: Path) -> None:
    """reconcile() rebuilds nodes from completed/<id>.json — the SE must be there."""
    from automil.graph import ExperimentGraph

    graph = ExperimentGraph(path=str(tmp_path / "graph.json"))
    node_id = graph.add_executed(parent_id=None, description="root", techniques=[],
                                 metrics={"composite": 0.50}, status="keep")
    graph.nodes[node_id]["status"] = "running"
    graph.save()

    completed_dir, archive_dir = _write(tmp_path, graph, node_id, {
        "status": "completed", "composite": 0.80,
        "metrics": {"val_auc": 0.80, "val_bacc": 0.80},
        "composite_se": 0.033,
    })

    completion = json.loads((completed_dir / f"{node_id}.json").read_text())
    assert completion["composite_se"] == 0.033
    archived = json.loads((archive_dir / "result.json").read_text())
    assert archived["composite_se"] == 0.033


def test_terminal_writer_gates_a_child_on_the_parent_se(tmp_path: Path) -> None:
    """THE DEFECT, end to end: +0.02 on a ±0.04-SE parent used to be kept."""
    from automil.graph import ExperimentGraph

    graph = ExperimentGraph(path=str(tmp_path / "graph.json"))
    parent = graph.add_executed(parent_id=None, description="parent", techniques=[],
                                metrics={"composite": 0.80, "composite_se": 0.04},
                                status="keep")
    child = graph.add_proposed(parent_id=parent, description="child", techniques=[])
    graph.nodes[child]["status"] = "running"
    graph.save()

    _write(tmp_path, graph, child, {
        "status": "completed", "composite": 0.82,
        "metrics": {"val_auc": 0.82, "val_bacc": 0.82},
        "composite_se": 0.04,
    })

    node = ExperimentGraph(path=str(tmp_path / "graph.json")).get_node(child)
    assert node["status"] == "discard"


def test_reconcile_completed_path_applies_the_noise_floor(tmp_path: Path) -> None:
    """graph.reconcile() rebuilds from completed/ — same gate as terminal_writer."""
    from automil.graph import ExperimentGraph

    adir = tmp_path
    for sub in ("queue", "running", "completed", "archive"):
        (adir / sub).mkdir()

    graph = ExperimentGraph(path=str(adir / "graph.json"))
    parent = graph.add_executed(parent_id=None, description="parent", techniques=[],
                                metrics={"composite": 0.80, "composite_se": 0.04},
                                status="keep")
    child = graph.add_proposed(parent_id=parent, description="child", techniques=[])
    graph.save()

    (adir / "completed" / f"{child}.json").write_text(json.dumps({
        "id": child, "status": "completed", "composite": 0.82,
        "metrics": {"val_auc": 0.82, "val_bacc": 0.82},
        "composite_se": 0.04,
        "graph_metadata": {"parent_id": parent},
    }))

    graph.reconcile(str(adir / "queue"), str(adir / "running"),
                    str(adir / "completed"), str(adir / "archive"))
    node = graph.get_node(child)
    assert node["status"] == "discard"
    assert node["composite_se"] == 0.04
    assert "composite_se" not in node["metrics"]


def test_archive_recovery_path_applies_the_noise_floor(tmp_path: Path) -> None:
    """The third keep/discard site: archive/<id>/result.json recovery."""
    from automil.graph import ExperimentGraph

    adir = tmp_path
    for sub in ("queue", "running", "completed", "archive"):
        (adir / sub).mkdir()

    graph = ExperimentGraph(path=str(adir / "graph.json"))
    parent = graph.add_executed(parent_id=None, description="parent", techniques=[],
                                metrics={"composite": 0.80, "composite_se": 0.04},
                                status="keep")
    graph.save()

    recovered = adir / "archive" / "node_0099"
    recovered.mkdir()
    (recovered / "spec.json").write_text(json.dumps({
        "description": "recovered child", "graph_metadata": {"parent_id": parent},
    }))
    (recovered / "result.json").write_text(json.dumps({
        "status": "completed", "composite": 0.82,
        "metrics": {"val_auc": 0.82, "val_bacc": 0.82},
        "composite_se": 0.04,
    }))

    graph.reconcile(str(adir / "queue"), str(adir / "running"),
                    str(adir / "completed"), str(adir / "archive"))
    node = graph.get_node("node_0099")
    assert node["status"] == "discard"
    assert node["composite_se"] == 0.04
