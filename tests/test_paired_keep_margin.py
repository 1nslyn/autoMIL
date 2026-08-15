"""Paired-fold keep margin: the Ladder bar scales with the paired-delta SE.

Runs share folds under a locked seed, so the fold effect cancels in the
per-fold child−parent difference. The margin basis switches to
``SE(paired deltas)`` whenever both nodes carry composites for the same fold
set (and the reducer is ``mean``); otherwise it falls back to the marginal
parent SE (CR-4).

The canonical regression numbers are the virchow2 runtime-canary cell:
baseline folds (0.6770, 0.4299, 0.6064) → marginal SE 0.0735, and its best
child node_0004 folds (0.6863, 0.4716, 0.6113) → paired SE 0.0116. Under the
marginal basis the child was discarded against a bar (0.6446) its per-fold
oracle could not reach; under the paired basis it clears max(δ=0.015, 0.0116).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

BASELINE_FOLDS = {0: 0.6770, 1: 0.4299, 2: 0.6064}
CHILD_FOLDS = {0: 0.6863, 1: 0.4716, 2: 0.6113}
DELTA = 0.015  # the campaign's predeclared accept_margin


def _entries(folds: dict[int, float]) -> list[dict]:
    return [{"fold_index": i, "composite": c} for i, c in sorted(folds.items())]


def _mean(folds: dict[int, float]) -> float:
    return sum(folds.values()) / len(folds)


# ---------------------------------------------------------------------------
# Unit surface
# ---------------------------------------------------------------------------

def test_paired_delta_se_matches_hand_computation() -> None:
    from automil.scoring import paired_delta_se

    se = paired_delta_se(CHILD_FOLDS, BASELINE_FOLDS)
    assert se == pytest.approx(0.011603, abs=1e-5)


def test_paired_delta_se_requires_identical_fold_sets() -> None:
    from automil.scoring import paired_delta_se

    assert paired_delta_se({0: 0.6, 1: 0.5}, {0: 0.6, 2: 0.5}) is None
    assert paired_delta_se({0: 0.6}, {0: 0.5}) is None          # n=1
    assert paired_delta_se(None, BASELINE_FOLDS) is None
    assert paired_delta_se({}, {}) is None


def test_paired_delta_se_uniform_deltas_are_zero_not_none() -> None:
    from automil.scoring import paired_delta_se

    child = {i: v + 0.02 for i, v in BASELINE_FOLDS.items()}
    assert paired_delta_se(child, BASELINE_FOLDS) == 0.0


def test_fold_composite_map_skips_junk_entries() -> None:
    from automil.scoring import fold_composite_map

    entries = [
        {"fold_index": 0, "composite": 0.6},
        {"fold_index": True, "composite": 0.5},        # bool index
        {"fold_index": 1, "composite": float("nan")},  # non-finite
        {"fold_index": 2},                             # missing composite
        "not-a-mapping",
        {"fold_index": 3, "composite": 0.7},
    ]
    assert fold_composite_map(entries) == {0: 0.6, 3: 0.7}
    assert fold_composite_map([]) is None
    assert fold_composite_map(None) is None


def test_node_fold_composites_reads_both_sources() -> None:
    from automil.graph import node_fold_composites

    # Regular ingested node: top-level fold_composites.
    assert node_fold_composites({"fold_composites": _entries(CHILD_FOLDS)}) == CHILD_FOLDS
    # Baseline root: the campaign controller writes metadata.validation_folds
    # (entries additionally carry a metrics dict — same reader).
    baseline_shaped = {
        "metadata": {
            "validation_folds": [
                {"fold_index": i, "metrics": {"val_auc": c}, "composite": c}
                for i, c in sorted(BASELINE_FOLDS.items())
            ]
        }
    }
    assert node_fold_composites(baseline_shaped) == BASELINE_FOLDS
    # The top-level field wins when both are present.
    both = dict(baseline_shaped, fold_composites=_entries(CHILD_FOLDS))
    assert node_fold_composites(both) == CHILD_FOLDS
    assert node_fold_composites({}) is None
    assert node_fold_composites(None) is None


def test_effective_accept_margin_canonical_virchow2_numbers() -> None:
    from automil.graph import _accept, effective_accept_margin

    meta = {"scoring": {"accept_margin": DELTA, "se_multiplier": 1.0, "formula": "mean"}}
    parent = {
        "composite": _mean(BASELINE_FOLDS),
        "composite_se": 0.07347,
        "metadata": {"validation_folds": _entries(BASELINE_FOLDS)},
    }
    child = {
        "composite": _mean(CHILD_FOLDS),
        "fold_composites": _entries(CHILD_FOLDS),
    }

    marginal = effective_accept_margin(meta, parent)
    assert marginal == pytest.approx(0.07347)
    assert not _accept(child["composite"], parent["composite"], marginal)

    paired = effective_accept_margin(meta, parent, child)
    assert paired == pytest.approx(DELTA)  # max(0.015, 0.0116) — δ floor binds
    assert _accept(child["composite"], parent["composite"], paired)


def test_paired_basis_requires_mean_formula() -> None:
    from automil.graph import effective_accept_margin

    parent = {
        "composite_se": 0.07347,
        "metadata": {"validation_folds": _entries(BASELINE_FOLDS)},
    }
    child = {"fold_composites": _entries(CHILD_FOLDS)}
    for formula in ("max", "min", "trust_reported"):
        meta = {"scoring": {"accept_margin": DELTA, "se_multiplier": 1.0,
                            "formula": formula}}
        assert effective_accept_margin(meta, parent, child) == pytest.approx(0.07347)


def test_paired_margin_monotone_over_delta_floor() -> None:
    """A noisy paired SE still RAISES the bar above δ, never lowers below it."""
    from automil.graph import effective_accept_margin

    meta = {"scoring": {"accept_margin": 0.005, "se_multiplier": 1.0, "formula": "mean"}}
    parent = {"metadata": {"validation_folds": _entries(BASELINE_FOLDS)}}
    child = {"fold_composites": _entries(CHILD_FOLDS)}
    assert effective_accept_margin(meta, parent, child) == pytest.approx(0.011603, abs=1e-5)


# ---------------------------------------------------------------------------
# Real ingest path (the artifact under test is the graph node + decision; the
# result payload is written exactly as a training script would write it).
# ---------------------------------------------------------------------------

def _noop_tsv(nid, result, description=""):
    return None


def _ingest(tmp_path: Path, graph, node_id: str, result: dict) -> None:
    from automil.terminal_writer import write_terminal_state

    completed_dir = tmp_path / "completed"
    completed_dir.mkdir(exist_ok=True)
    archive_dir = tmp_path / "archive" / node_id
    archive_dir.mkdir(parents=True, exist_ok=True)
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


def _result_payload(folds: dict[int, float]) -> dict:
    """A result.json exactly as the benchmark runner emits it: per-fold val
    metrics whose composite is their mean, plus the aggregate metrics block."""
    n = len(folds)
    mean_auc = sum(folds.values()) / n
    return {
        "status": "completed",
        "composite": mean_auc,
        "metrics": {"val_auc": mean_auc},
        "validation_folds": [
            {"fold_index": i, "metrics": {"val_auc": c}, "composite": c}
            for i, c in sorted(folds.items())
        ],
    }


def _campaign_graph(tmp_path: Path):
    """A graph whose meta matches the campaign scoring config, with a
    baseline root shaped the way ``_ensure_discovery_baseline_root`` writes it
    (folds in metadata.validation_folds, never through the terminal writer)."""
    from automil.graph import ExperimentGraph

    graph = ExperimentGraph(path=str(tmp_path / "graph.json"))
    graph.meta.setdefault("scoring", {}).update(
        {"accept_margin": DELTA, "se_multiplier": 1.0, "formula": "mean"}
    )
    root_id = graph.add_executed(
        parent_id=None, description="native baseline", techniques=[],
        metrics={"composite": _mean(BASELINE_FOLDS), "val_auc": _mean(BASELINE_FOLDS)},
        status="keep",
    )
    root = graph.get_node(root_id)
    root["composite"] = _mean(BASELINE_FOLDS)
    root["composite_se"] = 0.07347
    root["metadata"] = {
        "validation_folds": [
            {"fold_index": i, "metrics": {"val_auc": c}, "composite": c}
            for i, c in sorted(BASELINE_FOLDS.items())
        ]
    }
    graph.save()
    return graph, root_id


def test_ingest_keeps_virchow2_best_child_under_paired_margin(tmp_path: Path) -> None:
    from automil.graph import ExperimentGraph

    graph, root_id = _campaign_graph(tmp_path)
    child_id = graph.add_proposed(parent_id=root_id, description="anneal30",
                                  techniques=[], kind="hp")
    graph.nodes[child_id]["status"] = "running"
    graph.save()

    _ingest(tmp_path, graph, child_id, _result_payload(CHILD_FOLDS))

    node = ExperimentGraph(path=str(tmp_path / "graph.json")).get_node(child_id)
    assert node["status"] == "keep"          # discarded under the marginal basis
    assert node["fold_composites"] == _entries(CHILD_FOLDS)
    assert "fold_composites" not in node.get("metrics", {})


def test_ingest_without_fold_data_falls_back_to_marginal(tmp_path: Path) -> None:
    from automil.graph import ExperimentGraph

    graph, root_id = _campaign_graph(tmp_path)
    child_id = graph.add_proposed(parent_id=root_id, description="anneal30",
                                  techniques=[], kind="hp")
    graph.nodes[child_id]["status"] = "running"
    graph.save()

    payload = _result_payload(CHILD_FOLDS)
    del payload["validation_folds"]          # legacy result: no fold evidence
    _ingest(tmp_path, graph, child_id, payload)

    node = ExperimentGraph(path=str(tmp_path / "graph.json")).get_node(child_id)
    assert node["status"] == "discard"       # marginal bar 0.0735 holds
    assert node.get("fold_composites") is None


def test_completed_artifact_round_trips_fold_composites(tmp_path: Path) -> None:
    """reconcile() rebuilds nodes from completed/<id>.json — without the fold
    projection there, a recovered node would silently revert to the marginal
    basis. The completion is written by the REAL terminal writer; nothing here
    hand-writes the asserted artifact."""
    graph, root_id = _campaign_graph(tmp_path)
    child_id = graph.add_proposed(parent_id=root_id, description="anneal30",
                                  techniques=[], kind="hp")
    graph.nodes[child_id]["status"] = "running"
    graph.save()

    _ingest(tmp_path, graph, child_id, _result_payload(CHILD_FOLDS))

    completion = json.loads((tmp_path / "completed" / f"{child_id}.json").read_text())
    assert completion["fold_composites"] == _entries(CHILD_FOLDS)


def test_rank_leaderboard_shows_paired_delta_and_bar(tmp_path: Path, capsys) -> None:
    """The leaderboard surfaces composite ± SE, paired Δparent ± SE, and the
    required bar — the numbers both canary agents hand-parsed 30 archive
    JSONs to reconstruct."""
    from automil.cli.propose import _print_leaderboard
    from automil.graph import ExperimentGraph

    graph, root_id = _campaign_graph(tmp_path)
    child_id = graph.add_proposed(parent_id=root_id, description="anneal30",
                                  techniques=[], kind="hp")
    graph.nodes[child_id]["status"] = "running"
    graph.save()
    _ingest(tmp_path, graph, child_id, _result_payload(CHILD_FOLDS))

    reloaded = ExperimentGraph(path=str(tmp_path / "graph.json"))
    _print_leaderboard(reloaded)
    out = capsys.readouterr().out

    assert "Completed nodes" in out
    assert child_id in out
    assert "Δparent +0.0186" in out
    assert "±0.0116" in out          # paired SE, not the marginal 0.0735
    assert "(bar 0.0150)" in out     # max(δ=0.015, paired SE)
    assert "[keep]" in out


def test_reevaluate_descendants_uses_paired_basis(tmp_path: Path) -> None:
    """_reevaluate_descendants re-runs the accept with node-stored folds."""
    from automil.graph import ExperimentGraph

    graph, root_id = _campaign_graph(tmp_path)
    child_id = graph.add_proposed(parent_id=root_id, description="anneal30",
                                  techniques=[], kind="hp")
    graph.nodes[child_id]["status"] = "running"
    graph.save()
    _ingest(tmp_path, graph, child_id, _result_payload(CHILD_FOLDS))

    reloaded = ExperimentGraph(path=str(tmp_path / "graph.json"))
    reloaded._reevaluate_descendants(root_id)
    assert reloaded.get_node(child_id)["status"] == "keep"
