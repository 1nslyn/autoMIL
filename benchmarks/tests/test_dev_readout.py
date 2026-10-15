"""The dev-cell readout: the companion guard, the commit each node ran at and its run override."""
from __future__ import annotations

import json
import math
import statistics

import pytest
from _dev_cell_helpers import (
    ARCHIVED_OVERRIDES,
    LS_FOLDS,
    ROOT_COMMIT,
    ROOT_FOLDS,
    load_script,
    write_synthetic_cell,
)

from autobench.campaign import max_lift_winner

#: The root's val_bacc per fold; every companion below is this plus a per-fold change.
ROOT_BACC = [0.60, 0.62, 0.58, 0.61, 0.59]
MARGIN = 0.01  # the guard the synthetic cell declares (DEFAULT_GUARD)


@pytest.fixture(scope="module")
def dev_loop():
    return load_script("dev_loop")


@pytest.fixture(scope="module")
def dev_cell():
    return load_script("dev_cell")


def _shifted(changes: list[float]) -> list[float]:
    return [bacc + change for bacc, change in zip(ROOT_BACC, changes)]


def _nodes(data: dict) -> dict[str, dict]:
    return {node["id"]: node for node in data["nodes"]}


def _hand_pair(changes: list[float]) -> tuple[float, float]:
    """Mean per-fold change and its standard error, from the formula."""
    return statistics.fmean(changes), statistics.stdev(changes) / math.sqrt(len(changes))


# --- the companion guard -----------------------------------------------------------------------


@pytest.mark.parametrize("root_in_metadata", [False, True])
def test_a_node_whose_companion_fell_is_excluded_from_the_pool_and_listed_with_the_freezes_reason(
    dev_loop, tmp_path, root_in_metadata,
):
    cell = write_synthetic_cell(
        tmp_path / "cell", root_in_metadata=root_in_metadata,
        companions={"node_0001": ROOT_BACC,
                    "node_0002": _shifted([-0.05] * 5),  # the leader on the primary signal, but val_bacc collapsed
                    "node_0003": _shifted([-0.004, 0.002, -0.006, 0.001, -0.003])},
    )
    data = dev_loop.build_readout(cell)
    nodes = _nodes(data)

    collapsed = nodes["node_0002"]["guard"]
    assert collapsed["metric"] == "val_bacc" and collapsed["status"] == "fail"
    assert collapsed["reason"] == "companion guard: val_bacc fell 0.0500 below the baseline (margin 0.0100)"
    assert collapsed["vs_root"] == {"delta": pytest.approx(-0.05), "se": pytest.approx(0.0, abs=1e-12)}

    changes = [-0.004, 0.002, -0.006, 0.001, -0.003]
    delta, se = _hand_pair(changes)
    kept = nodes["node_0003"]["guard"]
    assert kept["status"] == "pass" and kept["reason"] is None
    assert kept["vs_root"] == {"delta": pytest.approx(delta), "se": pytest.approx(se)}

    for bare in (nodes["node_0001"], nodes["node_0004"]):  # the baseline, and a crash with no folds
        assert bare["guard"] is None
    assert data["guard"] == {"metric": "val_bacc", "margin": MARGIN, "se_multiplier": 1.0}

    verdict = data["verdict"]
    pool = [{"candidate_id": "node_0003", "candidate_sha256": "c3" * 8, "fold_values": LS_FOLDS}]
    assert verdict["guard_excluded"] == [{"node": "node_0002", "reason": collapsed["reason"]}]
    assert verdict["n_pool"] == 1 and verdict["frozen_rule"] == max_lift_winner(ROOT_FOLDS, pool)
    assert verdict["frozen_rule"]["leader"] == "node_0003"  # without the guard the leader is node_0002


@pytest.mark.parametrize(("changes", "status"), [
    ([-0.01] * 5, "pass"),                          # a drop of exactly the margin is one validation slide
    ([-0.011] * 5, "fail"),                         # more than one
    ([0.05] * 5, "pass"),                           # a gain
    ([-0.10, 0.08, -0.09, 0.07, -0.06], "pass"),    # mean -0.02, but the paired SE 0.0394 is larger: noise
    ([-0.05, 0.03, -0.04, 0.02, -0.06], "fail"),    # mean -0.02, paired SE 0.0187: a real drop
])
def test_a_drop_is_judged_against_the_larger_of_the_margin_and_k_times_the_paired_se(
    dev_loop, tmp_path, changes, status,
):
    cell = write_synthetic_cell(
        tmp_path / "cell", companions={"node_0001": ROOT_BACC, "node_0002": _shifted(changes)},
    )
    guard = _nodes(dev_loop.build_readout(cell))["node_0002"]["guard"]
    delta, se = _hand_pair(changes)
    assert guard["status"] == status
    assert guard["vs_root"] == {"delta": pytest.approx(delta), "se": pytest.approx(se, abs=1e-12)}
    expected_drop = max(0.0, -delta)
    if status == "fail":
        assert f"fell {expected_drop:.4f} below the baseline (margin {max(MARGIN, se):.4f})" in guard["reason"]


def test_a_companion_the_node_did_not_record_on_every_fold_fails_closed(dev_loop, tmp_path):
    cell = write_synthetic_cell(
        tmp_path / "cell", companions={"node_0001": ROOT_BACC, "node_0002": [0.6, None, 0.6, 0.6, 0.6]},
    )
    data = dev_loop.build_readout(cell)
    guard = _nodes(data)["node_0002"]["guard"]
    assert guard["status"] == "fail" and guard["vs_root"] is None
    assert guard["reason"] == "companion guard: val_bacc is not recorded on every fold"
    assert [excluded["node"] for excluded in data["verdict"]["guard_excluded"]] == ["node_0002"]
    json.dumps(data, allow_nan=False)  # the fail-closed shortfall is infinite; the readout must not carry it


def test_a_pool_the_guard_empties_leaves_the_baseline_and_says_why(dev_loop, tmp_path):
    cell = write_synthetic_cell(
        tmp_path / "cell",
        companions={"node_0001": ROOT_BACC, "node_0002": _shifted([-0.05] * 5), "node_0003": _shifted([-0.06] * 5)},
    )
    data = dev_loop.build_readout(cell)
    verdict = data["verdict"]
    assert verdict["available"] and verdict["n_pool"] == 0 and verdict["sidak"] is None
    assert verdict["frozen_rule"]["leader"] is None and verdict["frozen_rule"]["winner"] == "baseline"
    assert [excluded["node"] for excluded in verdict["guard_excluded"]] == ["node_0002", "node_0003"]
    text = dev_loop.format_readout(data)
    assert "no attempt in the winner pool; the baseline stays" in text
    assert text.count("excluded by the companion guard:") == 2


def test_a_cell_without_a_guard_says_so_and_excludes_nothing(dev_loop, tmp_path):
    cell = write_synthetic_cell(
        tmp_path / "cell", guard=None,
        companions={"node_0001": ROOT_BACC, "node_0002": _shifted([-0.2] * 5)},  # would fail any guard
    )
    data = dev_loop.build_readout(cell)
    assert data["guard"] is None and all(node["guard"] is None for node in data["nodes"])
    assert data["verdict"]["guard_excluded"] == [] and data["verdict"]["n_pool"] == 2
    text = dev_loop.format_readout(data)
    assert "companion guard: none declared" in text


def test_the_text_readout_prints_each_nodes_companion_delta_and_status_and_the_exclusions(dev_loop, tmp_path):
    cell = write_synthetic_cell(
        tmp_path / "cell",
        companions={"node_0001": ROOT_BACC, "node_0002": _shifted([-0.05] * 5), "node_0003": _shifted([0.01] * 5)},
    )
    text = dev_loop.format_readout(dev_loop.build_readout(cell))
    assert "companion guard: val_bacc, margin 0.0100" in text
    assert "guard       val_bacc vs root -0.0500 ± 0.0000  fail" in text
    assert "guard       val_bacc vs root +0.0100 ± 0.0000  pass" in text
    assert "excluded by the companion guard: node_0002 (companion guard: val_bacc fell 0.0500" in text
    assert text.count("guard       ") == 2  # the root and the crash are not judged


def test_a_guard_the_config_declares_unusably_or_the_root_cannot_carry_is_refused(dev_loop, tmp_path):
    cell = write_synthetic_cell(tmp_path / "bad_margin", guard={"metric": "val_bacc", "margin": -0.01})
    with pytest.raises(dev_loop.DevLoopError, match="scoring.guard"):
        dev_loop.build_readout(cell)

    cell = write_synthetic_cell(tmp_path / "no_companion", companions={"node_0001": [0.6, 0.6, None, 0.6, 0.6]})
    with pytest.raises(dev_loop.DevLoopError, match="root.*val_bacc"):
        dev_loop.build_readout(cell)

    cell = write_synthetic_cell(tmp_path / "no_config")
    (cell / "automil" / "config.yaml").unlink()
    with pytest.raises(dev_loop.DevLoopError, match="config.yaml"):
        dev_loop.build_readout(cell)


def test_the_guard_follows_the_metric_the_config_names(dev_loop, tmp_path):
    cell = write_synthetic_cell(tmp_path / "cell", guard={"metric": "val_auc", "margin": 0.05})
    guard = _nodes(dev_loop.build_readout(cell))["node_0003"]["guard"]
    # the fixture's val_auc is its primary value
    delta, _ = _hand_pair([ls - root for ls, root in zip(LS_FOLDS, ROOT_FOLDS)])
    assert guard["metric"] == "val_auc" and guard["vs_root"]["delta"] == pytest.approx(delta)
    assert guard["status"] == "pass"


# --- the commit each node ran at, and its run override ----------------------------------------------


def test_each_row_carries_the_base_commit_and_run_override_from_its_archived_spec(dev_loop, tmp_path):
    data = dev_loop.build_readout(write_synthetic_cell(tmp_path / "cell"))
    nodes = _nodes(data)
    for node_id in ("node_0001", "node_0002", "node_0003"):
        assert nodes[node_id]["base_commit"] == ROOT_COMMIT
        assert nodes[node_id]["run_command_override"] == ARCHIVED_OVERRIDES[node_id]
    assert nodes["node_0004"]["base_commit"] is None and nodes["node_0004"]["run_command_override"] is None


def test_the_text_readout_prints_the_commit_and_the_override_cut_to_about_a_hundred_characters(dev_loop, tmp_path):
    text = dev_loop.format_readout(dev_loop.build_readout(write_synthetic_cell(tmp_path / "cell")))
    assert text.count(f"  base        {ROOT_COMMIT}\n") == 3
    assert "  run         --policy-variant ema --hparams '{}'\n" in text  # short enough: printed whole
    long_line = next(line for line in text.splitlines() if line.startswith("  run         --policy-variant ls"))
    assert len(ARCHIVED_OVERRIDES["node_0003"]) > 100
    assert long_line.endswith("...") and len(long_line.removeprefix("  run         ")) == 100
    assert "  base        -\n" in text and "  run         -\n" in text  # the crash never launched


def test_an_unreadable_archived_spec_stops_the_readout_with_its_path(dev_loop, tmp_path):
    cell = write_synthetic_cell(tmp_path / "cell")
    spec = cell / "automil/orchestrator/archive/node_0002/spec.json"
    spec.write_text("{")
    with pytest.raises(dev_loop.DevLoopError, match="node_0002/spec.json"):
        dev_loop.build_readout(cell)


def test_readout_cli_json_carries_the_guard_and_the_commit(dev_cell, tmp_path, capsys):
    cell = write_synthetic_cell(
        tmp_path / "cell", companions={"node_0001": ROOT_BACC, "node_0002": _shifted([-0.05] * 5)},
    )
    dev_cell.main(["readout", str(cell), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert data["guard"]["metric"] == "val_bacc" and data["verdict"]["guard_excluded"][0]["node"] == "node_0002"
    assert _nodes(data)["node_0002"]["base_commit"] == ROOT_COMMIT
