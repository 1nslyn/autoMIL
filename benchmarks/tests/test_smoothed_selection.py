"""Protocol v5: a fold is scored on the validation curve around its restored epoch.

The restored checkpoint is still the argmax epoch (``SelectionTracker``). What
candidates are compared on is the mean primary metric over the five evaluated
epochs around it, so a single lucky epoch on a ~47-slide split no longer
decides a comparison on its own. The per-epoch values come from the one seam
every arm calls once per evaluated epoch, ``PolicyRuntime.should_stop``.
"""
from __future__ import annotations

import math

import pytest

from autobench.pipeline.policy_dispatch import PolicyRuntime
from autobench.pipeline.selection import smoothed_selection_value

NAN = float("nan")


def _curve(*values: float) -> dict[int, float]:
    return dict(enumerate(values))


def _mean(*values: float) -> float:
    return math.fsum(values) / len(values)


class TestSmoothedSelectionValue:
    def test_an_interior_epoch_averages_two_epochs_on_each_side(self):
        history = _curve(0.1, 0.2, 0.3, 0.9, 0.5, 0.6, 0.7)
        assert smoothed_selection_value(history, 3) == _mean(0.2, 0.3, 0.9, 0.5, 0.6)

    def test_the_window_shifts_inward_at_the_first_epoch(self):
        history = _curve(0.9, 0.2, 0.3, 0.4, 0.5, 0.6)
        assert smoothed_selection_value(history, 0) == _mean(0.9, 0.2, 0.3, 0.4, 0.5)

    def test_the_window_shifts_inward_at_the_last_epoch(self):
        history = _curve(0.1, 0.2, 0.3, 0.4, 0.5, 0.9)
        assert smoothed_selection_value(history, 5) == _mean(0.2, 0.3, 0.4, 0.5, 0.9)

    def test_a_run_shorter_than_the_window_averages_every_epoch(self):
        assert smoothed_selection_value(_curve(0.4, 0.8, 0.6), 1) == _mean(0.4, 0.8, 0.6)

    def test_non_finite_epochs_are_dropped_before_the_window_is_taken(self):
        history = _curve(0.1, NAN, 0.3, 0.9, -math.inf, 0.6, 0.7, 0.8)
        # Finite epochs 0, 2, 3, 5, 6, 7; the window centres on epoch 3.
        assert smoothed_selection_value(history, 3) == _mean(0.1, 0.3, 0.9, 0.6, 0.7)

    def test_epochs_are_matched_by_number_not_by_position(self):
        # nnMIL and TITAN skip evaluations, so epoch numbers have gaps.
        history = {0: 0.5, 2: 0.6, 4: 0.9, 6: 0.7, 8: 0.8, 10: 0.4}
        assert smoothed_selection_value(history, 4) == _mean(0.5, 0.6, 0.9, 0.7, 0.8)
        assert smoothed_selection_value(history, 10) == _mean(0.6, 0.9, 0.7, 0.8, 0.4)

    def test_a_selected_epoch_without_a_finite_value_has_no_score(self):
        assert smoothed_selection_value(_curve(0.4, 0.5), 7) is None
        assert smoothed_selection_value(_curve(0.4, NAN, 0.5), 1) is None
        assert smoothed_selection_value({}, 0) is None

    def test_every_epoch_of_a_twenty_epoch_run_gets_a_full_window(self):
        values = [0.5 + 0.01 * epoch for epoch in range(20)]
        history = dict(enumerate(values))
        for selected in range(20):
            lo = max(0, min(selected - 2, 15))
            assert smoothed_selection_value(history, selected) == _mean(*values[lo:lo + 5])

    @pytest.mark.parametrize("window", [0, -1, True, 2.5])
    def test_the_window_must_be_a_positive_integer(self, window):
        with pytest.raises(ValueError):
            smoothed_selection_value(_curve(0.5), 0, window=window)


class TestRuntimeHistory:
    def test_the_runtime_scores_the_epochs_it_was_shown(self, capsys):
        runtime = PolicyRuntime().for_fold()
        values = [0.6, 0.7, 0.65, 0.8, 0.75, 0.7]
        for epoch, value in enumerate(values):
            runtime.should_stop(False, epoch=epoch, metrics={"val_auc": value, "val_loss": 1.0})
        capsys.readouterr()
        assert runtime.smoothed(3, "val_auc") == smoothed_selection_value(
            dict(enumerate(values)), 3,
        )

    def test_each_fold_starts_with_an_empty_history(self, capsys):
        base = PolicyRuntime()
        first = base.for_fold()
        first.should_stop(False, epoch=0, metrics={"val_auc": 0.9})
        second = base.for_fold()
        capsys.readouterr()
        assert second is not first
        assert first.smoothed(0, "val_auc") == 0.9
        assert second.smoothed(0, "val_auc") is None

    def test_a_metric_the_arm_never_passed_has_no_score(self, capsys):
        runtime = PolicyRuntime().for_fold()
        runtime.should_stop(False, epoch=0, metrics={"val_auc": 0.9})
        capsys.readouterr()
        assert runtime.smoothed(0, "val_c_index") is None

    def test_numpy_scalars_are_recorded_and_non_scalars_are_ignored(self, capsys):
        np = pytest.importorskip("numpy")
        runtime = PolicyRuntime().for_fold()
        runtime.should_stop(
            False, epoch=0,
            metrics={"val_auc": np.float64(0.75), "flag": True, "note": "x"},
        )
        capsys.readouterr()
        assert runtime.smoothed(0, "val_auc") == 0.75
        assert runtime.smoothed(0, "flag") is None
        assert runtime.smoothed(0, "note") is None
