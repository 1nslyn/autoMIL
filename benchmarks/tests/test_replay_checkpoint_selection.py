"""Behavioural tests for benchmarks/scripts/replay_checkpoint_selection.py.

A synthetic two-fold run.log where the loss minimum and the AUC maximum
disagree, with one nnMIL-style ``val_auc=0.0`` sentinel epoch, one ``nan``
epoch and a ``[selected]`` line per fold. Every assertion is on parsed values
or produced files, never on source text.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "replay_checkpoint_selection.py"

FOLD_0 = """\
[prep] Splits already exist (noise the parser must ignore)
  hoptimus1:  50%|█████     | 2/4 [00:00<00:00, 38.16slide/s]
[epoch 0] val_loss=0.60 val_bacc=0.5 val_f1=0.5 val_auc=0.55
[epoch 1] val_loss=0.70 val_bacc=0.5 val_f1=0.5 val_auc=0.70
[epoch 2] val_loss=0.65 val_bacc=0.5 val_f1=0.5 val_auc=0.0
[epoch 3] val_loss=0.80 val_bacc=0.5 val_f1=0.5 val_auc=0.70
[selected] epoch=0 source=best
"""
FOLD_1 = """\
[epoch 0] val_loss=nan val_bacc=0.5 val_f1=0.5 val_auc=nan
[epoch 1] val_loss=0.50 val_bacc=0.5 val_f1=0.5 val_auc=0.60
[epoch 2] val_loss=0.40 val_bacc=0.5 val_f1=0.5 val_auc=0.58
[epoch 3] val_loss=0.45 val_bacc=0.5 val_f1=0.5 val_auc=0.57
[epoch 4] val_loss=0.30 val_bacc=0.5 val_f1=0.5 val_auc=0.65
[selected] epoch=4 source=best
"""


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("replay_checkpoint_selection", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def _cell(root: Path, arm: str = "nnmil", task_family: str = "binary", log: str = FOLD_0 + FOLD_1) -> Path:
    (root / "automil").mkdir(parents=True)
    (root / "automil" / "campaign_cell.json").write_text(json.dumps(
        {"cell_id": root.name, "framework": arm, "task_type": "classification", "task_family": task_family}
    ))
    (root / "baseline-execution" / "archive").mkdir(parents=True)
    (root / "baseline-execution" / "archive" / "run.log").write_text(log)
    return root


def _rows(mod, root: Path, rule):
    cell = mod.load_cell(root)
    return mod.replay_log(cell, "baseline", root / "baseline-execution" / "archive" / "run.log", rule)


def test_segments_split_on_selected_lines_and_ignore_noise(mod):
    segments = mod.parse_segments((FOLD_0 + FOLD_1).splitlines())
    assert [len(s.epochs) for s in segments] == [4, 5]
    assert [(s.selected_epoch, s.source) for s in segments] == [(0, "best"), (4, "best")]
    assert [index for index, _ in segments[1].epochs] == [0, 1, 2, 3, 4]
    assert segments[0].epochs[1][1] == {"val_loss": 0.70, "val_bacc": 0.5, "val_f1": 0.5, "val_auc": 0.70}
    assert math.isnan(segments[1].epochs[0][1]["val_auc"])


def test_new_rule_moves_the_selection_from_the_loss_minimum_to_the_auc_maximum(mod, tmp_path):
    fold_0, fold_1 = _rows(mod, _cell(tmp_path / "c"), mod.StopRule(patience=10))
    assert (fold_0["old_epoch"], fold_0["new_epoch"], fold_0["changed"]) == (0, 1, True)
    assert (fold_0["primary@old"], fold_0["primary@new"]) == (0.55, 0.70)
    assert (fold_0["loss@old"], fold_0["loss@new"]) == (0.60, 0.70)
    assert fold_0["n_epochs_at_new_max"] == 2  # the tie at epoch 3 keeps epoch 1
    assert fold_0["would_stop_epoch"] is None
    assert (fold_1["old_epoch"], fold_1["new_epoch"], fold_1["changed"]) == (4, 4, False)
    assert (fold_1["epochs_run"], fold_1["old_rule_recomputed"]) == (5, 4)


def test_sentinel_and_nan_epochs_are_absent_and_counted_separately(mod, tmp_path):
    fold_0, fold_1 = _rows(mod, _cell(tmp_path / "c"), mod.StopRule(patience=10))
    assert (fold_0["n_absent_metric_epochs"], fold_0["n_zero_sentinel_epochs"]) == (1, 1)
    assert (fold_1["n_absent_metric_epochs"], fold_1["n_zero_sentinel_epochs"]) == (1, 0)


def test_zero_auc_is_a_sentinel_for_nnmil_only(mod):
    metrics = {"val_auc": 0.0, "val_loss": 0.5}
    nnmil = mod.observe_epoch(0, metrics, "val_auc", "nnmil")
    abmil = mod.observe_epoch(0, metrics, "val_auc", "abmil")
    assert math.isnan(nnmil.value) and nnmil.zero_sentinel
    assert abmil.value == 0.0 and not abmil.zero_sentinel
    assert mod.replay((nnmil,), mod.StopRule(patience=10)).best_epoch == -1
    assert mod.replay((abmil,), mod.StopRule(patience=10)).best_epoch == 0


def test_survival_cells_select_on_the_c_index(mod, tmp_path):
    log = "[epoch 0] val_loss=1.0 val_c_index=0.50\n[epoch 1] val_loss=1.2 val_c_index=0.61\n[selected] epoch=0 source=best\n"
    (row,) = _rows(mod, _cell(tmp_path / "s", arm="titan", task_family="survival", log=log), mod.StopRule(patience=10))
    assert (row["old_epoch"], row["new_epoch"], row["primary@new"]) == (0, 1, 0.61)


def test_replay_stops_where_the_v4_trainer_would_and_flags_a_later_maximum(mod, tmp_path):
    fold_0, fold_1 = _rows(mod, _cell(tmp_path / "c"), mod.StopRule(patience=2))
    # fold 1: selected at 1, non-improving at 2 and 3 -> stop at 3; epoch 4 (0.65) is never seen
    assert (fold_1["would_stop_epoch"], fold_1["new_epoch"], fold_1["primary@new"]) == (3, 1, 0.60)
    assert fold_1["later_max_ignored"] is True
    # fold 0: selected at 1, sentinel at 2 and tie at 3 both count toward patience -> stop at 3
    assert (fold_0["would_stop_epoch"], fold_0["new_epoch"], fold_0["later_max_ignored"]) == (3, 1, False)


def test_clam_floor_delays_the_stop_past_epoch_50(mod, tmp_path):
    (_, fold_1) = _rows(mod, _cell(tmp_path / "c", arm="clam"), mod.StopRule(patience=2, floor=50))
    assert fold_1["would_stop_epoch"] is None
    assert (fold_1["new_epoch"], fold_1["primary@new"]) == (4, 0.65)


def test_self_check_flags_a_logged_epoch_that_is_not_the_loss_minimum(mod, tmp_path):
    log = FOLD_0.replace("[selected] epoch=0", "[selected] epoch=1")
    (row,) = _rows(mod, _cell(tmp_path / "d", log=log), mod.StopRule(patience=10))
    assert (row["old_epoch"], row["old_rule_recomputed"]) == (1, 0)
    assert mod.self_check_lines([row])[0].endswith(": 1")


def test_cli_writes_the_csv_and_the_report(mod, tmp_path):
    _cell(tmp_path / "cells" / "good")
    out, report = tmp_path / "replay.csv", tmp_path / "replay.md"
    assert mod.main(["--cells", str(tmp_path / "cells" / "*"), "--out", str(out), "--report", str(report)]) == 0
    with out.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert set(rows[0]) >= {
        "cell_id", "arm", "task_family", "kind", "fold", "epochs_run", "old_epoch", "new_epoch",
        "primary@old", "primary@new", "loss@old", "loss@new", "changed", "would_stop_epoch",
        "n_absent_metric_epochs", "old_rule_recomputed",
    }
    assert [(r["kind"], r["fold"], r["old_epoch"], r["new_epoch"], r["changed"]) for r in rows] == [
        ("baseline", "0", "0", "1", "true"), ("baseline", "1", "4", "4", "false"),
    ]
    assert (rows[0]["arm"], rows[0]["task_family"], rows[0]["would_stop_epoch"]) == ("nnmil", "binary", "")
    assert float(rows[1]["primary@old"]) == 0.65 and rows[1]["n_absent_metric_epochs"] == "1"
    text = report.read_text()
    assert "good" in text and "| nnmil | baseline | 2 | 0.5000 |" in text


def test_malformed_log_is_reported_and_skipped_without_losing_the_others(mod, tmp_path):
    _cell(tmp_path / "cells" / "good")
    _cell(tmp_path / "cells" / "bad", log=FOLD_0.replace("val_auc=0.55", "val_auc=abc"))
    _cell(tmp_path / "cells" / "empty", log="no epochs ran\n")
    rows, skipped = mod.replay_cells(mod.expand_cells([str(tmp_path / "cells" / "*")]), None)
    assert [row["cell_id"] for row in rows] == ["good", "good"]
    assert {s.path.parts[-4] for s in skipped} == {"bad", "empty"}
    assert all(s.reason for s in skipped)


def test_patience_override_must_be_positive(mod, tmp_path):
    with pytest.raises(SystemExit):
        mod.parse_args(["--cells", str(tmp_path), "--out", "x.csv", "--patience-override", "0"])
