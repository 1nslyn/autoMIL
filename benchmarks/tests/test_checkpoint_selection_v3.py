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


class TestSharedCELoss:
    def test_known_value(self):
        import numpy as np
        from autobench.pipeline.val_loss import ce_loss
        expect = -(np.log(0.8) + np.log(0.7)) / 2
        assert abs(ce_loss([0, 1], [[0.8, 0.2], [0.3, 0.7]]) - expect) < 1e-12

    def test_zero_prob_clips_instead_of_inf(self):
        from autobench.pipeline.val_loss import ce_loss
        v = ce_loss([0], [[0.0, 1.0]])
        assert v == v and v < float("inf")  # finite, huge


class TestArmLoopsSelectOnLoss:
    """Scripted eval: AUC rises while loss worsens -> epoch 0 must stay
    selected in both arms. Exercises the real training loops."""

    def _scripted(self, calls, losses, aucs):
        import numpy as np

        def fake_evaluate(*a, **kw):
            if not kw.get("return_probs"):
                return {"auc_roc": aucs[-1], "accuracy": 0.5,
                        "balanced_accuracy": 0.5, "f1": 0.5,
                        "sensitivity": 0.5, "specificity": 0.5}
            i = min(len(calls), len(losses) - 1)
            calls.append(i)
            p = float(np.exp(-losses[i]))
            return ({"auc_roc": aucs[i], "accuracy": 0.5,
                     "balanced_accuracy": 0.5, "f1": 0.5,
                     "sensitivity": 0.5, "specificity": 0.5},
                    np.array([0, 1]),
                    np.array([[p, 1 - p], [1 - p, p]]))
        return fake_evaluate

    def test_abmil_keeps_the_loss_minimum(self, monkeypatch, capsys, tmp_path):
        import numpy as np
        from autobench.pipeline.abmil import train as abmil_train
        from test_abmil_arm import _make_split, _smoke_cfg  # same tests pkg

        calls = []
        monkeypatch.setattr(
            abmil_train, "_evaluate",
            self._scripted(calls, losses=[0.40, 0.70, 0.60], aucs=[0.60, 0.95, 0.90]),
        )
        rng = np.random.default_rng(0)
        import dataclasses
        cfg = dataclasses.replace(_smoke_cfg(), max_epochs=3, early_stopping=False)
        abmil_train.train_abmil_fold(
            "abmil", _make_split(rng, "t", 6), _make_split(rng, "v", 2),
            _make_split(rng, "e", 2), embed_dim=abmil_train_IN_DIM(),
            num_classes=2, cfg=cfg, device=__import__("torch").device("cpu"),
            seed=0,
        )
        assert "[selected] epoch=0 source=best" in capsys.readouterr().out


def abmil_train_IN_DIM():
    from test_abmil_arm import IN_DIM
    return IN_DIM


class TestDTFDLoopSelectsOnLoss:
    def test_dtfd_keeps_the_loss_minimum(self, monkeypatch, capsys):
        import dataclasses
        import numpy as np
        import torch
        from autobench.pipeline.dtfd import train as dtfd_train
        from test_dtfd_arm import _make_split, _smoke_cfg

        seq = iter([(0.60, 0.40), (0.95, 0.70), (0.90, 0.60)])  # (auc, loss)
        monkeypatch.setattr(dtfd_train, "val_scores", lambda *a, **k: next(seq))
        rng = np.random.default_rng(0)
        cfg = dataclasses.replace(_smoke_cfg(), max_epochs=3, early_stopping=False)
        dtfd_train.train_dtfd_fold(
            _make_split(rng, "t", 6), _make_split(rng, "v", 2),
            _make_split(rng, "e", 2), embed_dim=_dtfd_emb(),
            num_classes=2, cfg=cfg, device=torch.device("cpu"), seed=0,
        )
        assert "[selected] epoch=0 source=best" in capsys.readouterr().out


def _dtfd_emb():
    from test_dtfd_arm import EMB
    return EMB


class TestCLAMAlreadySelectsOnLoss:
    """CLAM's vendored EarlyStopping is upstream loss-selection — pin it so a
    future vendored bump cannot silently switch the fifth arm's rule."""

    def test_clam_checkpoint_follows_val_loss(self, tmp_path):
        import torch
        from autobench.pipeline.clam._imports import EarlyStopping

        es = EarlyStopping(patience=2, stop_epoch=0, verbose=False)
        model = torch.nn.Linear(2, 2)
        ck = str(tmp_path / "ck.pt")
        es(0, 0.60, model, ckpt_name=ck)
        assert es.counter == 0
        es(1, 0.70, model, ckpt_name=ck)   # worse loss -> counter up
        assert es.counter == 1
        es(2, 0.50, model, ckpt_name=ck)   # better loss -> reset + save
        assert es.counter == 0
        assert es.val_loss_min == 0.50
