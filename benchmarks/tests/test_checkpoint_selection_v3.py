"""Protocol v3: checkpoints are selected on continuous validation loss.

Canary evidence (2026-08-16, docs/tutorials/run_agentic_campaign.md §8):
selecting the checkpoint on plan-BACC reported the max-over-epochs of a
34-valued statistic on 47 slides — `corr(epochs_run, composite) = +0.77`,
and the uni_v2 top-10 selected that way collapsed onto baseline on held
folds (mean lift +0.0023, corr(disc, held) = −0.28). Loss is continuous
and does not reward running longer; the plan metric is still reported at
the selected checkpoint — it just does not vote.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

# Installs benchmarks/lib/nnMIL on sys.path (same mechanism the trainer uses).
import autobench.pipeline.nnmil._imports  # noqa: F401
from training.callbacks.early_stopping import EarlyStopping  # noqa: E402


def _model():
    return torch.nn.Linear(2, 2)


@pytest.fixture
def stopper(tmp_path):
    return EarlyStopping(patience=3, metric="bacc", save_dir=str(tmp_path),
                         model_type="simple_mil")


class TestLossSelectsTheCheckpoint:
    def test_metric_gain_with_loss_regression_does_not_advance_checkpoint(
        self, stopper,
    ):
        """The exact leak shape: BACC jumps while loss worsens — the
        checkpoint must stay at the loss minimum."""
        m = _model()
        stopper(val_loss=0.60, val_bacc=0.50, val_f1=0.5, val_auc=0.5, model=m, epoch=0)
        stopper(val_loss=0.72, val_bacc=0.80, val_f1=0.8, val_auc=0.8, model=m, epoch=1)
        assert stopper.best_epoch == 0, (
            "a BACC improvement with worse val loss must not move the checkpoint"
        )

    def test_loss_improvement_advances_checkpoint_despite_metric_drop(
        self, stopper,
    ):
        m = _model()
        stopper(val_loss=0.70, val_bacc=0.70, val_f1=0.7, val_auc=0.7, model=m, epoch=0)
        stopper(val_loss=0.55, val_bacc=0.50, val_f1=0.5, val_auc=0.5, model=m, epoch=1)
        assert stopper.best_epoch == 1

    def test_patience_counts_loss_stagnation_not_metric_stagnation(self, stopper):
        m = _model()
        stopper(val_loss=0.50, val_bacc=0.5, val_f1=0.5, val_auc=0.5, model=m, epoch=0)
        for e in range(1, 4):  # loss flat/worse while bacc climbs: must stop
            stopper(val_loss=0.55, val_bacc=0.5 + 0.1 * e, val_f1=0.5,
                    val_auc=0.5, model=m, epoch=e)
        assert stopper.early_stop, (
            "rising BACC with stagnant loss must not reset patience — that is "
            "the run-longer-buy-more-draws exploit"
        )

    def test_nan_loss_never_becomes_the_checkpoint(self, stopper):
        m = _model()
        stopper(val_loss=float("nan"), val_bacc=0.9, val_f1=0.9, val_auc=0.9,
                model=m, epoch=0)
        stopper(val_loss=0.60, val_bacc=0.4, val_f1=0.4, val_auc=0.4,
                model=m, epoch=1)
        assert stopper.best_epoch == 1
