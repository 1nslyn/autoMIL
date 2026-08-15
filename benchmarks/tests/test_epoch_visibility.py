"""A3: per-epoch and selected-epoch visibility across the five manifest arms.

Two log contracts, both born from the canary diagnosis:

1. ``[epoch <k>] key=value ...`` — one line per epoch from the ONE dispatch
   seam every arm already calls (``PolicyRuntime.should_stop``), printing
   exactly the metrics the trainer passed. Before this, abmil/dtfd/titan
   printed no per-epoch lines at all, so reading a trajectory cost a charged
   policy probe per cell.
2. ``[selected] epoch=<k>`` — exactly once per fold, at the point the best
   checkpoint is restored, so any epoch-selection-coupled axis can be judged
   with the logged selected epochs rather than the headline composite alone.

The training-run test drives the REAL cheapest arm (ABMIL, 2 folds, tiny CPU
bags) end to end and asserts on what the run actually printed — per
tasks/lessons.md, nothing here writes the artifact it asserts.
"""

from __future__ import annotations

import re

import pytest

from autobench.pipeline.policy_dispatch import PolicyRuntime

# Reuse the real 2-fold ABMIL benchmark fixture (H5 bags on disk, real splits).
from tests.test_abmil_arm import (
    _build_benchmark_fixture,
    _exp_cfg,
    _smoke_cfg,
    make_test_ds,
)
from autobench.pipeline.config import build_registries

_EPOCH_LINE = re.compile(r"^\[epoch (\d+)\](?: (.*))?$")
_SELECTED_LINE = re.compile(r"^\[selected\] epoch=(-?\d+)$")


class TestDispatchEpochLine:
    """The harness line prints from the dispatch itself, policy or not."""

    def test_no_policy_prints_exactly_the_passed_metrics(self, capsys):
        rt = PolicyRuntime()  # no policy selected — the baseline-run case
        stop = rt.should_stop(False, epoch=3, metrics={"val_auc": 0.5})
        assert stop is False
        out = capsys.readouterr().out.splitlines()
        assert out == ["[epoch 3] val_auc=0.5"]

    def test_full_precision_floats_survive(self, capsys):
        # 21/34 — the canary's exact-rank-fraction val_auc. Truncated rendering
        # would alias neighbouring runs; repr round-trips.
        value = 21 / 34
        PolicyRuntime().should_stop(False, epoch=0, metrics={"val_auc": value})
        line = capsys.readouterr().out.strip()
        match = _EPOCH_LINE.match(line)
        assert match is not None
        assert float(line.split("val_auc=")[1]) == value

    def test_multi_metric_arms_print_all_and_only_their_metrics(self, capsys):
        PolicyRuntime().should_stop(
            False, epoch=7, metrics={"val_loss": 1.25, "val_c_index": 0.5},
        )
        out = capsys.readouterr().out.strip()
        assert out == "[epoch 7] val_loss=1.25 val_c_index=0.5"

    def test_empty_metrics_still_marks_the_epoch(self, capsys):
        PolicyRuntime().should_stop(False, epoch=2, metrics=None)
        assert capsys.readouterr().out.strip() == "[epoch 2]"

    def test_numpy_scalars_render_as_plain_floats(self, capsys):
        np = pytest.importorskip("numpy")
        PolicyRuntime().should_stop(
            False, epoch=1, metrics={"val_auc": np.float64(0.75)},
        )
        assert capsys.readouterr().out.strip() == "[epoch 1] val_auc=0.75"

    def test_one_line_per_call_with_a_live_policy(self, capsys):
        class _Policy:
            def should_stop(self, *, default, epoch, metrics):
                return bool(default)

            def wrap_optimizer_for(self, optimizer, *, role):
                return optimizer

            def wrap_scheduler_for(self, scheduler, *, role):
                return scheduler

        rt = PolicyRuntime(name="p", policy_factory=_Policy)
        for epoch in range(3):
            rt.should_stop(False, epoch=epoch, metrics={"val_auc": 0.5})
        lines = capsys.readouterr().out.splitlines()
        assert [_EPOCH_LINE.match(l).group(1) for l in lines] == ["0", "1", "2"]


class TestCheapestArmTrainingLog:
    """Real 2-fold ABMIL run: `[epoch` lines, one `[selected]` per fold."""

    @pytest.fixture(scope="class")
    def run_output(self, tmp_path_factory):
        from autobench.pipeline.abmil.runner import run_abmil_experiment

        root = tmp_path_factory.mktemp("abmil-visibility")
        _build_benchmark_fixture(str(root), n_folds=2)
        exp = _exp_cfg(build_registries(make_test_ds()), n_folds=2)

        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            summary = run_abmil_experiment(
                exp, str(root), device="cpu", cfg=_smoke_cfg(),
            )
        return summary, buf.getvalue()

    def test_epoch_lines_present_with_val_auc(self, run_output):
        _, log = run_output
        epoch_lines = [
            l for l in log.splitlines() if _EPOCH_LINE.match(l.strip())
        ]
        assert epoch_lines, "no [epoch <k>] lines in a real training log"
        assert all("val_auc=" in l for l in epoch_lines)

    def test_exactly_one_selected_line_per_fold(self, run_output):
        summary, log = run_output
        selected = [
            _SELECTED_LINE.match(l.strip())
            for l in log.splitlines()
            if _SELECTED_LINE.match(l.strip())
        ]
        assert len(selected) == summary["n_folds"] == 2
        for match in selected:
            assert int(match.group(1)) >= 0

    def test_selected_epoch_is_one_of_the_printed_epochs(self, run_output):
        _, log = run_output
        epochs = {
            int(m.group(1))
            for m in (_EPOCH_LINE.match(l.strip()) for l in log.splitlines())
            if m
        }
        for line in log.splitlines():
            m = _SELECTED_LINE.match(line.strip())
            if m:
                assert int(m.group(1)) in epochs
