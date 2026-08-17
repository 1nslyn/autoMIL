"""Protocol v3: checkpoints are selected on continuous validation loss.

Canary evidence (2026-08-16, docs/tutorials/run_agentic_campaign.md §8):
selecting the checkpoint on plan-BACC reported the max-over-epochs of a
34-valued statistic on 47 slides — `corr(epochs_run, primary_value) = +0.77`,
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


class TestTITANLoopSelectsOnLoss:
    def test_titan_keeps_the_loss_minimum(self, monkeypatch, capsys, tmp_path):
        import numpy as np
        import torch
        from autobench.pipeline.titan import train as titan_train
        from autobench.pipeline.config import (
            ExperimentConfig, TaskConfig, ModelConfig, TrainConfig, Framework,
        )

        class _DS(torch.utils.data.Dataset):
            def __init__(self, n):
                self.x = torch.randn(n, 768)
                self.y = torch.tensor([i % 2 for i in range(n)])
            def __len__(self):
                return len(self.x)
            def __getitem__(self, i):
                return self.x[i], self.y[i]

        losses = [0.40, 0.70, 0.60]
        aucs = [0.60, 0.95, 0.90]
        state = {"i": 0}

        real = titan_train._evaluate

        def fake(*a, **kw):
            if not kw.get("return_probs"):
                return real(*a, **kw)
            i = min(state["i"], len(losses) - 1)
            state["i"] += 1
            p = float(np.exp(-losses[i]))
            return ({"auc_roc": aucs[i]}, np.array([0, 1]),
                    np.array([[p, 1 - p], [1 - p, p]]))

        monkeypatch.setattr(titan_train, "_evaluate", fake)
        cfg = ExperimentConfig(
            task=TaskConfig(name="t", label_col="y",
                            label_dict={"a": 0, "b": 1}, n_classes=2),
            encoder_key="titan", embed_dim=768,
            model=ModelConfig(model_type="titan"),
            train=TrainConfig(max_epochs=3, patience=5, seed=0,
                              early_stopping=False),
            n_folds=2, framework=Framework.TITAN, strategy="standard",
        )
        titan_train.train_titan_fold(
            cfg, _DS(6), _DS(2), _DS(2),
            fold=0, results_dir=str(tmp_path / "r"), device="cpu",
        )
        assert "[selected] epoch=0 source=best" in capsys.readouterr().out


class TestCELossNonFinite:
    def test_pos_inf_prob_cannot_win_selection(self):
        from autobench.pipeline.val_loss import ce_loss
        assert ce_loss([0], [[float("inf"), 0.0]]) == float("inf")

    def test_neg_inf_prob_is_worst_not_finite(self):
        from autobench.pipeline.val_loss import ce_loss
        assert ce_loss([0], [[float("-inf"), 1.0]]) == float("inf")

    def test_nan_prob_is_worst(self):
        from autobench.pipeline.val_loss import ce_loss
        assert ce_loss([0], [[float("nan"), 1.0]]) == float("inf")


class TestNnMILAllNonFiniteRunSavesNothing:
    def test_initial_nan_epochs_never_checkpoint(self, tmp_path):
        import torch
        import autobench.pipeline.nnmil._imports  # noqa: F401
        from training.callbacks.early_stopping import EarlyStopping

        es = EarlyStopping(patience=2, metric="bacc", save_dir=str(tmp_path),
                           model_type="simple_mil")
        m = torch.nn.Linear(2, 2)
        es(float("nan"), 0.9, 0.9, 0.9, m, epoch=0)
        es(float("nan"), 0.9, 0.9, 0.9, m, epoch=1)
        assert es.best_epoch == -1, "no checkpoint may exist for an all-NaN run"
        assert es.early_stop, "patience must still count degenerate epochs"
        assert not list(tmp_path.iterdir()), "no checkpoint file written"


class TestPatienceResetsAtFirstValidCheckpoint:
    def test_classification_nan_prefix_does_not_linger(self, tmp_path):
        import torch
        import autobench.pipeline.nnmil._imports  # noqa: F401
        from training.callbacks.early_stopping import EarlyStopping

        es = EarlyStopping(patience=3, metric="bacc", save_dir=str(tmp_path),
                           model_type="simple_mil")
        m = torch.nn.Linear(2, 2)
        es(float("nan"), 0.5, 0.5, 0.5, m, epoch=0)
        es(float("nan"), 0.5, 0.5, 0.5, m, epoch=1)
        es(0.60, 0.5, 0.5, 0.5, m, epoch=2)  # first valid save
        assert es.counter == 0 and es.best_epoch == 2 and not es.early_stop
        es(0.65, 0.5, 0.5, 0.5, m, epoch=3)  # one bad epoch must not stop
        assert not es.early_stop


class TestSurvivalDegenerateGuards:
    def _es(self, tmp_path, mode):
        import autobench.pipeline.nnmil._imports  # noqa: F401
        from training.callbacks.early_stopping import EarlyStoppingSurvival
        return EarlyStoppingSurvival(patience=2, save_dir=str(tmp_path),
                                     model_type="simple_mil", mode=mode)

    def test_finite_zero_cindex_is_a_real_score_not_degenerate(self, tmp_path):
        import torch
        es = self._es(tmp_path, "max")
        m = torch.nn.Linear(2, 2)
        es(0.7, 0.0, m, epoch=0)  # terrible but REAL c-index
        assert es.best_epoch == 0, "finite 0.0 C-index must checkpoint"

    def test_nan_cindex_saves_nothing_then_recovers(self, tmp_path):
        import torch
        es = self._es(tmp_path, "max")
        m = torch.nn.Linear(2, 2)
        es(0.7, float("nan"), m, epoch=0)
        assert es.best_epoch == -1
        es(0.7, 0.55, m, epoch=1)
        assert es.best_epoch == 1 and es.counter == 0 and not es.early_stop

    def test_min_mode_nan_loss_saves_nothing(self, tmp_path):
        import torch
        es = self._es(tmp_path, "min")
        m = torch.nn.Linear(2, 2)
        es(float("nan"), 0.5, m, epoch=0)
        assert es.best_epoch == -1


class TestCacheFingerprintCarriesProtocol:
    def test_protocol_version_in_payload(self):
        from autobench.pipeline.results_cache import fingerprint_payload
        from autobench.campaign import PROTOCOL_VERSION

        class _Cfg:
            def to_dict(self):
                return {"task": {"name": "t"}}

        assert fingerprint_payload(_Cfg())["protocol_version"] == PROTOCOL_VERSION


class TestStaleCheckpointFromPriorAttempt:
    """A best_<model>.pth left by a prior attempt in the same save_dir must
    be deleted at constructor time: the end-of-training restore checks file
    existence, not authorship, and under the orchestrator a same-node
    relaunch reuses the results dir — an all-degenerate retry would
    otherwise certify the previous attempt's weights as source=best."""

    def test_classification_ctor_removes_stale_best(self, tmp_path):
        stale = tmp_path / "best_simple_mil.pth"
        stale.write_bytes(b"weights from a prior attempt")
        EarlyStopping(patience=3, metric="bacc", save_dir=str(tmp_path),
                      model_type="simple_mil")
        assert not stale.exists()

    def test_survival_ctor_removes_stale_best(self, tmp_path):
        from training.callbacks.early_stopping import EarlyStoppingSurvival
        stale = tmp_path / "best_simple_mil.pth"
        stale.write_bytes(b"weights from a prior attempt")
        EarlyStoppingSurvival(patience=3, save_dir=str(tmp_path),
                              model_type="simple_mil", mode="min")
        assert not stale.exists()

    def test_regression_ctor_removes_stale_best(self, tmp_path):
        from training.callbacks.early_stopping import RegressionEarlyStopping
        stale = tmp_path / "best_simple_mil.pth"
        stale.write_bytes(b"weights from a prior attempt")
        RegressionEarlyStopping(patience=3, save_dir=str(tmp_path),
                                model_type="simple_mil")
        assert not stale.exists()

    def test_all_degenerate_run_ends_with_no_checkpoint_file(self, tmp_path):
        """End to end: stale file + every epoch non-finite → nothing on disk
        to restore (the trainer's exists() restore finds no file)."""
        stale = tmp_path / "best_simple_mil.pth"
        stale.write_bytes(b"weights from a prior attempt")
        es = EarlyStopping(patience=2, metric="bacc", save_dir=str(tmp_path),
                           model_type="simple_mil")
        m = _model()
        es(val_loss=float("nan"), val_bacc=0.5, val_f1=0.5, val_auc=0.5,
           model=m, epoch=0)
        es(val_loss=float("nan"), val_bacc=0.5, val_f1=0.5, val_auc=0.5,
           model=m, epoch=1)
        assert es.early_stop and es.best_epoch == -1
        assert not stale.exists()


class TestMissingValLossIsNotAPerfectLoss:
    """utils.compute_metrics only emits {prefix}/loss when val probs are
    finite and the val split has >1 class. The trainer's extraction default
    must therefore be NaN (routes into the non-finite guard: epoch skipped),
    never 0.0 — score -0.0 is finite and beats every real loss, permanently
    capturing the checkpoint."""

    def test_trainer_defaults_missing_val_loss_to_nan(self):
        import inspect
        from training.trainers import classification_trainer
        src = inspect.getsource(classification_trainer)
        line = next(
            l for l in src.splitlines()
            if "val_loss = val_metrics.get" in l
        )
        assert "float('nan')" in line or 'float("nan")' in line, (
            f"missing val/loss must default to NaN, got: {line.strip()}"
        )
        assert "0.0" not in line, (
            f"a 0.0 default is a PERFECT loss and captures the checkpoint: "
            f"{line.strip()}"
        )

    def test_nan_default_flows_to_no_save(self, tmp_path):
        """The exact defect shape: first epoch has no val/loss key → NaN →
        no checkpoint; a later real epoch takes the checkpoint normally."""
        es = EarlyStopping(patience=3, metric="bacc", save_dir=str(tmp_path),
                           model_type="simple_mil")
        m = _model()
        val_metrics: dict = {}  # val/loss suppressed (NaN probs / single-class val)
        val_loss = val_metrics.get('val_val/loss',
                                   val_metrics.get('val/loss', float('nan')))
        es(val_loss=val_loss, val_bacc=0.9, val_f1=0.9, val_auc=0.9,
           model=m, epoch=0)
        assert es.best_epoch == -1, "missing val/loss must not checkpoint"
        es(val_loss=0.7, val_bacc=0.6, val_f1=0.6, val_auc=0.6, model=m, epoch=1)
        assert es.best_epoch == 1, "a real loss must beat the missing-loss epoch"


class TestCLAMFlagOffStillSelectsOnLoss:
    """`early_stopping` is a legal tunable knob (SEARCH_SPACE['clam']), and
    upstream coupled it to checkpointing: flag off meant no per-epoch
    checkpoint at all, so the FINAL epoch's weights were scored — one legal
    proposal silently opted the fifth arm out of the frozen v3 selection
    rule. The tracker must run unconditionally; the flag may only gate
    early termination; the restore must key on checkpoint existence (and a
    stale checkpoint from a prior attempt must be deleted up front)."""

    @staticmethod
    def _core_utils_src():
        import inspect
        from autobench.pipeline.clam._imports import EarlyStopping
        return inspect.getsource(inspect.getmodule(EarlyStopping))

    def test_tracker_constructed_unconditionally(self):
        # Scope to train()'s body: validate()/validate_clam() legitimately
        # declare `early_stopping = None` as a default parameter.
        src = self._core_utils_src()
        train_body = src.split("def train(", 1)[1].split("def train_loop_clam", 1)[0]
        assert "early_stopping = None" not in train_body, (
            "the val-loss checkpoint tracker must exist on every code path — "
            "a None tracker means the final epoch's weights get scored"
        )

    def test_flag_only_suppresses_the_stop_signal(self):
        src = self._core_utils_src()
        assert "if not args.early_stopping:" in src
        after = src.split("if not args.early_stopping:", 1)[1]
        assert "stop = False" in after.splitlines()[2], (
            "flag off must mean 'run every epoch', not 'do not checkpoint'"
        )

    def test_restore_keys_on_checkpoint_existence_not_the_flag(self):
        src = self._core_utils_src()
        assert "if os.path.exists(ckpt_path):\n        model.load_state_dict" in src, (
            "the restore must follow the saved val-loss checkpoint whenever "
            "one exists, regardless of the early_stopping flag"
        )

    def test_stale_prior_attempt_checkpoint_is_deleted_before_training(self):
        src = self._core_utils_src()
        pre_loop = src.split("for epoch in range(args.max_epochs):", 1)[0]
        assert "os.remove(ckpt_path)" in pre_loop, (
            "a checkpoint left by a prior attempt in the same results_dir "
            "must not be restorable as this run's selection"
        )


class TestCLAMNonFiniteLossGuard:
    """Upstream's else branch SAVES a NaN epoch (NaN fails `score < best`,
    so it falls through as an 'improvement'). With the tracker now
    unconditional, every CLAM run walks this code — the guard must map
    non-finite to -inf (never displaces a finite best, never saves first)."""

    def test_nan_loss_never_displaces_a_finite_best(self, tmp_path):
        import torch
        from autobench.pipeline.clam._imports import EarlyStopping

        es = EarlyStopping(patience=3, stop_epoch=0, verbose=False)
        m = torch.nn.Linear(2, 2)
        ck = str(tmp_path / "ck.pt")
        es(0, 0.60, m, ckpt_name=ck)
        assert es.best_epoch == 0
        es(1, float("nan"), m, ckpt_name=ck)
        assert es.best_epoch == 0, "a NaN epoch must not become the checkpoint"
        assert es.counter == 1, "a NaN epoch counts toward patience"

    def test_nan_first_epoch_saves_nothing(self, tmp_path):
        import os
        import torch
        from autobench.pipeline.clam._imports import EarlyStopping

        es = EarlyStopping(patience=2, stop_epoch=0, verbose=False)
        m = torch.nn.Linear(2, 2)
        ck = str(tmp_path / "ck.pt")
        es(0, float("nan"), m, ckpt_name=ck)
        assert es.best_epoch == -1
        assert not os.path.exists(ck)
        es(1, 0.5, m, ckpt_name=ck)   # recovery: first finite loss checkpoints
        assert es.best_epoch == 1 and os.path.exists(ck)
