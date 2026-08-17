"""A3: per-epoch and selected-epoch visibility across the five manifest arms.

Two log contracts, both born from the canary diagnosis:

1. ``[epoch <k>] key=value ...`` — one line per epoch from the ONE dispatch
   seam every arm already calls (``PolicyRuntime.should_stop``), printing
   exactly the metrics the trainer passed. Before this, abmil/dtfd/titan
   printed no per-epoch lines at all, so reading a trajectory cost a charged
   policy probe per cell.
2. ``[selected] epoch=<k> source=<best|final|untrained>`` — exactly once per
   fold, at the point the selected weights are fixed, so any epoch-selection-
   coupled axis can be judged with the logged selected epochs rather than the
   headline composite alone. The ``source`` token is one convention across
   ALL arms and both vendored libs, because ``epoch=-1`` alone is ambiguous:
     - ``source=best``      a val-selected checkpoint was restored;
     - ``source=final``     the final trained weights were kept (no restore);
     - ``source=untrained`` the restored snapshot predates any training step
       (TITAN's pre-loop deepcopy, e.g. under an all-NaN val split).

The training-run tests drive REAL cheap arms (ABMIL and DTFD, 2 folds, tiny
CPU bags) end to end and assert on what the run actually printed — per
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
_SELECTED_LINE = re.compile(
    r"^\[selected\] epoch=(-?\d+) source=(best|final|untrained)$"
)


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


class TestArmTrainingLogs:
    """Real 2-fold runs of the two cheapest arms (ABMIL, DTFD): `[epoch`
    lines, one `[selected] ... source=` per fold — the tokens are a
    convention, not one arm's quirk."""

    @pytest.fixture(scope="class", params=["abmil", "dtfd"])
    def run_output(self, request, tmp_path_factory):
        import contextlib
        import io

        root = tmp_path_factory.mktemp(f"{request.param}-visibility")
        buf = io.StringIO()

        if request.param == "abmil":
            from autobench.pipeline.abmil.runner import run_abmil_experiment

            _build_benchmark_fixture(str(root), n_folds=2)
            exp = _exp_cfg(build_registries(make_test_ds()), n_folds=2)
            with contextlib.redirect_stdout(buf):
                summary = run_abmil_experiment(
                    exp, str(root), device="cpu", cfg=_smoke_cfg(),
                )
        else:
            from autobench.pipeline.dtfd.runner import run_dtfd_experiment
            from tests.test_dtfd_arm import (
                _build_benchmark_fixture as _build_dtfd_fixture,
                _exp_cfg as _dtfd_exp_cfg,
                _smoke_cfg as _dtfd_smoke_cfg,
            )

            _build_dtfd_fixture(str(root), n_folds=2)
            exp = _dtfd_exp_cfg(build_registries(make_test_ds()), n_folds=2)
            with contextlib.redirect_stdout(buf):
                summary = run_dtfd_experiment(
                    exp, str(root), device="cpu", cfg=_dtfd_smoke_cfg(),
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
            # Both arms have a val split and run real epochs here, so a
            # val-selected snapshot IS restored: epoch >= 0 and source=best.
            # (source=final marks final-weights-kept, source=untrained marks
            # TITAN's pre-training snapshot — neither can occur in this run.)
            assert int(match.group(1)) >= 0
            assert match.group(2) == "best"

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


class TestUntrainedSourceToken:
    """TITAN restores its pre-loop deepcopy when no epoch ever improved —
    that must NOT print like the final-trained-weights-kept `epoch=-1` of
    ABMIL/DTFD. Zero-epoch fold: cheapest real path to the case (an all-NaN
    val split reaches it the same way)."""

    def test_titan_zero_epoch_fold_prints_untrained(self, tmp_path, capsys):
        h5py = pytest.importorskip("h5py")
        import numpy as np

        from autobench.pipeline.config import (
            ExperimentConfig,
            Framework,
            ModelConfig,
            TaskConfig,
            TrainConfig,
        )
        from autobench.pipeline.titan.dataset import TitanSlideDataset
        from autobench.pipeline.titan.train import train_titan_fold

        feat_dir = tmp_path / "features_titan"
        feat_dir.mkdir()
        rng = np.random.default_rng(3)
        sids, labels = [f"s{i}" for i in range(4)], [0, 1, 0, 1]
        for sid in sids:
            with h5py.File(feat_dir / f"{sid}.h5", "w") as f:
                f.create_dataset(
                    "features", data=rng.standard_normal(8).astype("float32"),
                )

        ds = TitanSlideDataset(sids, labels, str(feat_dir))
        exp = ExperimentConfig(
            task=TaskConfig(
                name="brca", label_col="label", label_dict={"neg": 0, "pos": 1},
            ),
            encoder_key="titan",
            embed_dim=8,
            model=ModelConfig(model_type="titan"),
            train=TrainConfig(seed=1, max_epochs=0),
            n_folds=1,
            framework=Framework.TITAN,
            strategy="standard",
        )
        train_titan_fold(
            exp, ds, ds, ds, fold=0,
            results_dir=str(tmp_path / "results"), device="cpu",
        )

        lines = [l.strip() for l in capsys.readouterr().out.splitlines()]
        selected = [m for m in map(_SELECTED_LINE.match, lines) if m]
        assert len(selected) == 1
        assert (selected[0].group(1), selected[0].group(2)) == ("-1", "untrained")


class TestCallbackOwnsBestEpoch:
    """F-D4: the [selected] epoch is OWNED by nnMIL's early-stopping
    callbacks, set exactly where the checkpoint is saved — call sites never
    re-derive it from ``counter == 0``."""

    @staticmethod
    def _linear():
        import torch

        return torch.nn.Linear(2, 2)

    @staticmethod
    def _callbacks():
        from autobench.pipeline.nnmil import _imports  # noqa: F401 (sys.path)
        from nnMIL.training.callbacks.early_stopping import (
            EarlyStopping,
            EarlyStoppingSurvival,
        )

        return EarlyStopping, EarlyStoppingSurvival

    def test_starts_at_minus_one(self):
        EarlyStopping, EarlyStoppingSurvival = self._callbacks()
        assert EarlyStopping(patience=5, metric="bacc").best_epoch == -1
        assert EarlyStoppingSurvival(patience=5, mode="min").best_epoch == -1

    def test_classification_tracks_the_saving_epoch(self):
        EarlyStopping, _ = self._callbacks()
        es = EarlyStopping(patience=5, metric="bacc")
        model = self._linear()
        es(0.50, 0.6, 0.6, 0.6, model, epoch=0)  # initial save
        assert es.best_epoch == 0
        es(0.60, 0.5, 0.5, 0.5, model, epoch=1)  # worse loss: no save
        assert es.best_epoch == 0
        es(0.40, 0.9, 0.9, 0.9, model, epoch=2)  # better loss: save
        assert es.best_epoch == 2

    def test_survival_records_the_true_epoch_across_warmup_skips(self):
        # nnMIL's survival trainer only starts calling at epoch 2 (warmup),
        # so the callback must record the epoch the CALLER names, not its
        # own call index.
        _, EarlyStoppingSurvival = self._callbacks()
        es = EarlyStoppingSurvival(patience=5, mode="min")
        model = self._linear()
        es(1.0, 0.5, model, epoch=2)
        assert es.best_epoch == 2
        es(0.4, 0.5, model, epoch=3)
        assert es.best_epoch == 3
        es(0.9, 0.5, model, epoch=4)  # worse: best stays
        assert es.best_epoch == 3

    def test_internal_counter_stands_in_when_no_epoch_is_passed(self):
        EarlyStopping, _ = self._callbacks()
        es = EarlyStopping(patience=5, metric="bacc")
        model = self._linear()
        es(0.50, 0.6, 0.6, 0.6, model)
        es(0.40, 0.9, 0.9, 0.9, model)
        assert es.best_epoch == 1
