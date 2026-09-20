"""Protocol v4: every arm selects its checkpoint on the PRIMARY validation metric.

Classification cells (binary, multiclass, ordinal) select on ``val_auc``;
survival cells on the in-fold ``val_c_index``. The same rule serves the native
baseline and every agent candidate. Under protocol v3 (continuous validation
loss) the loss minimum sat at epochs 0-3 on every rehearsal arm while the
objective kept rising for 7-20 epochs, so each arm reported a near-untrained
model and the search spent its budget moving the loss minimum later
(tasks/handoff.md §4.1, 2026-09-20).

One specification, three implementations (the shared ``SelectionTracker``,
CLAM's vendored ``EarlyStopping``, nnMIL's vendored callbacks), pinned by
``TestSelectorContract``: strictly-greater improves; a tie keeps the earlier
epoch; a non-finite value never selects and counts toward patience; patience
counts stagnation of the selection metric. Validation loss is still computed
and reported on the ``[epoch k]`` line for policies; it just does not vote.
"""
from __future__ import annotations

import logging
import math
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# Installs benchmarks/lib/nnMIL on sys.path (same mechanism the trainer uses).
import autobench.pipeline.nnmil._imports  # noqa: F401
from training.callbacks.early_stopping import (  # noqa: E402
    EarlyStopping,
    EarlyStoppingSurvival,
)

NAN = float("nan")


@pytest.fixture(autouse=True)
def _restore_torch_grad_state():
    """The nnMIL trainers are driven here without the autobench wrapper that
    restores torch's global grad switch after a fold; leave it as found."""
    was_enabled = torch.is_grad_enabled()
    yield
    torch.set_grad_enabled(was_enabled)


def _model():
    return torch.nn.Linear(2, 2)


# ---------------------------------------------------------------------------
# The shared primitive
# ---------------------------------------------------------------------------


class TestSelectionTracker:
    def _tracker(self, patience=3):
        from autobench.pipeline.selection import SelectionTracker
        return SelectionTracker(patience=patience)

    def test_starts_unselected(self):
        t = self._tracker()
        assert t.best_epoch == -1 and t.best_value is None
        assert not t.early_stop

    def test_observe_reports_only_improvements(self):
        t = self._tracker()
        assert t.observe(0, 0.5) is True
        assert t.observe(1, 0.4) is False
        assert t.observe(2, 0.6) is True
        assert t.best_epoch == 2 and t.best_value == 0.6

    def test_a_tie_keeps_the_earlier_epoch(self):
        t = self._tracker()
        t.observe(0, 0.6)
        assert t.observe(1, 0.6) is False
        assert t.best_epoch == 0 and t.counter == 1

    def test_non_finite_never_selects_and_counts_toward_patience(self):
        t = self._tracker(patience=2)
        assert t.observe(0, NAN) is False
        assert t.best_epoch == -1 and t.counter == 1
        assert t.observe(1, float("inf")) is False
        assert t.best_epoch == -1 and t.early_stop

    def test_a_non_finite_epoch_never_displaces_a_finite_best(self):
        t = self._tracker()
        t.observe(0, 0.6)
        t.observe(1, NAN)
        assert t.best_epoch == 0 and t.counter == 1

    def test_first_finite_value_resets_patience(self):
        t = self._tracker(patience=3)
        t.observe(0, NAN)
        t.observe(1, NAN)
        assert t.observe(2, 0.4) is True
        assert t.best_epoch == 2 and t.counter == 0 and not t.early_stop

    def test_patience_counts_stagnation_of_the_selection_metric(self):
        t = self._tracker(patience=3)
        t.observe(0, 0.6)
        for e in range(1, 3):
            t.observe(e, 0.5)
            assert not t.early_stop
        t.observe(3, 0.5)
        assert t.early_stop and t.best_epoch == 0

    def test_patience_must_be_positive(self):
        from autobench.pipeline.selection import SelectionTracker
        with pytest.raises(ValueError):
            SelectionTracker(patience=0)


# ---------------------------------------------------------------------------
# One contract, three implementations
# ---------------------------------------------------------------------------


def _feed_tracker(tmp_path):
    from autobench.pipeline.selection import SelectionTracker
    t = SelectionTracker(patience=3)
    return t, lambda epoch, value: t.observe(epoch, value)


def _feed_nnmil_classification(tmp_path):
    es = EarlyStopping(patience=3, save_dir=str(tmp_path), model_type="simple_mil")
    m = _model()
    return es, lambda epoch, value: es(value, m, epoch=epoch)


def _feed_nnmil_survival(tmp_path):
    es = EarlyStoppingSurvival(patience=3, save_dir=str(tmp_path), model_type="simple_mil")
    m = _model()
    return es, lambda epoch, value: es(value, m, epoch=epoch)


def _feed_clam(tmp_path):
    from autobench.pipeline.clam._imports import EarlyStopping as ClamEarlyStopping
    es = ClamEarlyStopping(patience=3, stop_epoch=0, verbose=False)
    m = _model()
    ck = str(tmp_path / "ck.pt")
    return es, lambda epoch, value: es(epoch, value, m, ckpt_name=ck)


SELECTORS = [_feed_tracker, _feed_nnmil_classification, _feed_nnmil_survival, _feed_clam]

# (trajectory, expected best_epoch, expected counter after the last value,
#  expected early_stop) under patience 3.
TRAJECTORIES = [
    ("rising", [0.5, 0.6, 0.7], 2, 0, False),
    ("drop_keeps_the_best", [0.7, 0.5, 0.6], 0, 2, False),
    ("tie_keeps_the_earlier", [0.6, 0.6, 0.6], 0, 2, False),
    ("nan_first_is_skipped", [NAN, 0.4, 0.5], 2, 0, False),
    ("nan_mid_never_displaces", [0.6, NAN, 0.5], 0, 2, False),
    ("all_nan_selects_nothing", [NAN, NAN, NAN], -1, 3, True),
    ("patience_exhausts_on_stagnation", [0.6, 0.5, 0.5, 0.5], 0, 3, True),
]


class TestSelectorContract:
    @pytest.mark.parametrize("factory", SELECTORS, ids=lambda f: f.__name__[6:])
    @pytest.mark.parametrize("name,values,best_epoch,counter,early_stop", TRAJECTORIES,
                             ids=[t[0] for t in TRAJECTORIES])
    def test_identical_verdicts(self, tmp_path, factory, name, values,
                                best_epoch, counter, early_stop):
        selector, feed = factory(tmp_path)
        for epoch, value in enumerate(values):
            feed(epoch, value)
        assert selector.best_epoch == best_epoch, name
        assert selector.counter == counter, name
        assert bool(selector.early_stop) is early_stop, name


# ---------------------------------------------------------------------------
# nnMIL callbacks: file effects that the contract test does not cover
# ---------------------------------------------------------------------------


class TestNnMILCallbackFileEffects:
    def test_all_nan_run_saves_nothing(self, tmp_path):
        es = EarlyStopping(patience=2, save_dir=str(tmp_path), model_type="simple_mil")
        m = _model()
        es(NAN, m, epoch=0)
        es(NAN, m, epoch=1)
        assert es.best_epoch == -1 and es.early_stop
        assert not list(tmp_path.iterdir()), "no checkpoint file written"

    def test_first_finite_auc_saves_and_resets_patience(self, tmp_path):
        es = EarlyStopping(patience=3, save_dir=str(tmp_path), model_type="simple_mil")
        m = _model()
        es(NAN, m, epoch=0)
        es(NAN, m, epoch=1)
        es(0.60, m, epoch=2)
        assert es.counter == 0 and es.best_epoch == 2 and not es.early_stop
        assert (tmp_path / "best_simple_mil.pth").exists()
        es(0.55, m, epoch=3)  # one worse epoch must not stop
        assert not es.early_stop and es.best_epoch == 2

    def test_survival_finite_zero_cindex_is_a_real_score(self, tmp_path):
        es = EarlyStoppingSurvival(patience=2, save_dir=str(tmp_path), model_type="simple_mil")
        es(0.0, _model(), epoch=0)
        assert es.best_epoch == 0, "finite 0.0 C-index must checkpoint"

    def test_survival_nan_cindex_saves_nothing_then_recovers(self, tmp_path):
        es = EarlyStoppingSurvival(patience=2, save_dir=str(tmp_path), model_type="simple_mil")
        m = _model()
        es(NAN, m, epoch=0)
        assert es.best_epoch == -1
        es(0.55, m, epoch=1)
        assert es.best_epoch == 1 and es.counter == 0 and not es.early_stop

    def test_best_model_state_is_a_true_copy(self, tmp_path):
        es = EarlyStopping(patience=3, save_dir=None, model_type=None)
        m = _model()
        es(0.6, m, epoch=0)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(1.0)
        saved = es.best_model_state["weight"]
        assert not torch.equal(saved, m.weight.detach()), (
            "best_model_state must not alias the live parameters"
        )


class TestStaleCheckpointFromPriorAttempt:
    """A best_<model>.pth left by a prior attempt in the same save_dir must
    be deleted at constructor time: the end-of-training restore checks file
    existence, not authorship, and under the orchestrator a same-node
    relaunch reuses the results dir."""

    def test_classification_ctor_removes_stale_best(self, tmp_path):
        stale = tmp_path / "best_simple_mil.pth"
        stale.write_bytes(b"weights from a prior attempt")
        EarlyStopping(patience=3, save_dir=str(tmp_path), model_type="simple_mil")
        assert not stale.exists()

    def test_survival_ctor_removes_stale_best(self, tmp_path):
        stale = tmp_path / "best_simple_mil.pth"
        stale.write_bytes(b"weights from a prior attempt")
        EarlyStoppingSurvival(patience=3, save_dir=str(tmp_path), model_type="simple_mil")
        assert not stale.exists()

    def test_regression_ctor_removes_stale_best(self, tmp_path):
        from training.callbacks.early_stopping import RegressionEarlyStopping
        stale = tmp_path / "best_simple_mil.pth"
        stale.write_bytes(b"weights from a prior attempt")
        RegressionEarlyStopping(patience=3, save_dir=str(tmp_path),
                                model_type="simple_mil")
        assert not stale.exists()

    def test_all_degenerate_run_ends_with_no_checkpoint_file(self, tmp_path):
        stale = tmp_path / "best_simple_mil.pth"
        stale.write_bytes(b"weights from a prior attempt")
        es = EarlyStopping(patience=2, save_dir=str(tmp_path), model_type="simple_mil")
        m = _model()
        es(NAN, m, epoch=0)
        es(NAN, m, epoch=1)
        assert es.early_stop and es.best_epoch == -1
        assert not stale.exists()


# ---------------------------------------------------------------------------
# Validation loss: still the companion on the epoch line, never the selector
# ---------------------------------------------------------------------------


class TestSharedCELoss:
    def test_known_value(self):
        from autobench.pipeline.val_loss import ce_loss
        expect = -(np.log(0.8) + np.log(0.7)) / 2
        assert abs(ce_loss([0, 1], [[0.8, 0.2], [0.3, 0.7]]) - expect) < 1e-12

    def test_zero_prob_clips_instead_of_inf(self):
        from autobench.pipeline.val_loss import ce_loss
        v = ce_loss([0], [[0.0, 1.0]])
        assert v == v and v < float("inf")  # finite, huge

    def test_non_finite_probabilities_are_inf(self):
        from autobench.pipeline.val_loss import ce_loss
        assert ce_loss([0], [[float("inf"), 0.0]]) == float("inf")
        assert ce_loss([0], [[float("-inf"), 1.0]]) == float("inf")
        assert ce_loss([0], [[NAN, 1.0]]) == float("inf")


# ---------------------------------------------------------------------------
# The classification loops (real trainers, scripted validation)
# ---------------------------------------------------------------------------


def _scripted_evaluate(calls, losses, aucs):
    """Per-epoch validation where loss and AUC disagree: the loss minimum is
    epoch 0, the AUC maximum epoch 1."""
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


LOSSES = [0.40, 0.70, 0.60]
AUCS = [0.60, 0.95, 0.90]


class TestArmLoopsSelectOnAUC:
    def test_abmil_selects_the_auc_maximum(self, monkeypatch, capsys):
        import dataclasses
        from autobench.pipeline.abmil import train as abmil_train
        from test_abmil_arm import IN_DIM, _make_split, _smoke_cfg  # same tests pkg

        monkeypatch.setattr(abmil_train, "_evaluate", _scripted_evaluate([], LOSSES, AUCS))
        rng = np.random.default_rng(0)
        cfg = dataclasses.replace(_smoke_cfg(), max_epochs=3, early_stopping=False)
        abmil_train.train_abmil_fold(
            "abmil", _make_split(rng, "t", 6), _make_split(rng, "v", 2),
            _make_split(rng, "e", 2), embed_dim=IN_DIM,
            num_classes=2, cfg=cfg, device=torch.device("cpu"), seed=0,
        )
        assert "[selected] epoch=1 source=best" in capsys.readouterr().out

    def test_dtfd_selects_the_auc_maximum(self, monkeypatch, capsys):
        import dataclasses
        from autobench.pipeline.dtfd import train as dtfd_train
        from test_dtfd_arm import EMB, _make_split, _smoke_cfg

        seq = iter(zip(AUCS, LOSSES))  # (auc, loss)
        monkeypatch.setattr(dtfd_train, "val_scores", lambda *a, **k: next(seq))
        rng = np.random.default_rng(0)
        cfg = dataclasses.replace(_smoke_cfg(), max_epochs=3, early_stopping=False)
        dtfd_train.train_dtfd_fold(
            _make_split(rng, "t", 6), _make_split(rng, "v", 2),
            _make_split(rng, "e", 2), embed_dim=EMB,
            num_classes=2, cfg=cfg, device=torch.device("cpu"), seed=0,
        )
        assert "[selected] epoch=1 source=best" in capsys.readouterr().out

    def test_titan_selects_the_auc_maximum(self, monkeypatch, capsys, tmp_path):
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

        state = {"i": 0}
        real = titan_train._evaluate

        def fake(*a, **kw):
            if not kw.get("return_probs"):
                return real(*a, **kw)
            i = min(state["i"], len(LOSSES) - 1)
            state["i"] += 1
            p = float(np.exp(-LOSSES[i]))
            return ({"auc_roc": AUCS[i]}, np.array([0, 1]),
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
        assert "[selected] epoch=1 source=best" in capsys.readouterr().out


class TestTITANUntrainedSource:
    def test_all_nan_val_split_keeps_the_pre_training_snapshot(self, monkeypatch, capsys, tmp_path):
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

        real = titan_train._evaluate

        def fake(*a, **kw):
            if not kw.get("return_probs"):
                return real(*a, **kw)
            return ({"auc_roc": NAN}, np.array([0, 1]),
                    np.array([[0.5, 0.5], [0.5, 0.5]]))

        monkeypatch.setattr(titan_train, "_evaluate", fake)
        cfg = ExperimentConfig(
            task=TaskConfig(name="t", label_col="y",
                            label_dict={"a": 0, "b": 1}, n_classes=2),
            encoder_key="titan", embed_dim=768,
            model=ModelConfig(model_type="titan"),
            train=TrainConfig(max_epochs=2, patience=5, seed=0,
                              early_stopping=False),
            n_folds=2, framework=Framework.TITAN, strategy="standard",
        )
        titan_train.train_titan_fold(
            cfg, _DS(6), _DS(2), _DS(2),
            fold=0, results_dir=str(tmp_path / "r"), device="cpu",
        )
        assert "[selected] epoch=-1 source=untrained" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# CLAM: the vendored callback and the real train() path
# ---------------------------------------------------------------------------


class TestCLAMSelectsOnAUC:
    def test_clam_checkpoint_follows_val_auc(self, tmp_path):
        from autobench.pipeline.clam._imports import EarlyStopping as ClamEarlyStopping

        es = ClamEarlyStopping(patience=2, stop_epoch=0, verbose=False)
        model = _model()
        ck = str(tmp_path / "ck.pt")
        es(0, 0.60, model, ckpt_name=ck)
        assert es.counter == 0 and es.best_epoch == 0
        es(1, 0.55, model, ckpt_name=ck)   # lower AUC -> counter up
        assert es.counter == 1 and es.best_epoch == 0
        es(2, 0.70, model, ckpt_name=ck)   # higher AUC -> reset + save
        assert es.counter == 0 and es.best_epoch == 2 and es.best_score == 0.70
        es(3, 0.70, model, ckpt_name=ck)   # tie -> the earlier epoch stays
        assert es.counter == 1 and es.best_epoch == 2

    def test_nan_auc_never_displaces_a_finite_best(self, tmp_path):
        from autobench.pipeline.clam._imports import EarlyStopping as ClamEarlyStopping

        es = ClamEarlyStopping(patience=3, stop_epoch=0, verbose=False)
        m = _model()
        ck = str(tmp_path / "ck.pt")
        es(0, 0.60, m, ckpt_name=ck)
        es(1, NAN, m, ckpt_name=ck)
        assert es.best_epoch == 0 and es.counter == 1

    def test_nan_first_epoch_saves_nothing(self, tmp_path):
        from autobench.pipeline.clam._imports import EarlyStopping as ClamEarlyStopping

        es = ClamEarlyStopping(patience=2, stop_epoch=0, verbose=False)
        m = _model()
        ck = str(tmp_path / "ck.pt")
        es(0, NAN, m, ckpt_name=ck)
        assert es.best_epoch == -1 and not os.path.exists(ck)
        es(1, 0.5, m, ckpt_name=ck)
        assert es.best_epoch == 1 and os.path.exists(ck)


def _clam_fixture(root, max_epochs):
    """A tiny but real CLAM classification fold (mirrors
    test_val_prediction_hash.TestClamClassificationRealRun)."""
    import pandas as pd
    from autobench.pipeline.clam.dataset import create_dataset, load_fold_splits
    from autobench.pipeline.config import (
        ExperimentConfig, Framework, ModelConfig, TaskConfig, TrainConfig,
    )

    n_slides, embed_dim, n_patches = 12, 64, 24
    benchmark_dir = str(root / "benchmark")
    encoder = "conch_v15"
    rng = np.random.default_rng(3)
    slide_ids = [f"c{i}" for i in range(n_slides)]
    labels = {sid: i % 2 for i, sid in enumerate(slide_ids)}
    pt_dir = os.path.join(benchmark_dir, "features", encoder, "pt_files")
    os.makedirs(pt_dir, exist_ok=True)
    for sid in slide_ids:
        feats = rng.standard_normal((n_patches, embed_dim)).astype("float32") + labels[sid] * 3.0
        torch.save(torch.from_numpy(feats), os.path.join(pt_dir, f"{sid}.pt"))
    csv_dir = os.path.join(benchmark_dir, "dataset_csv")
    os.makedirs(csv_dir, exist_ok=True)
    names = {0: "neg", 1: "pos"}
    pd.DataFrame({
        "slide_id": slide_ids, "case_id": slide_ids,
        "label": [names[labels[s]] for s in slide_ids],
    }).to_csv(os.path.join(csv_dir, "brca.csv"), index=False)
    splits_dir = os.path.join(benchmark_dir, "splits", "standard", "brca")
    os.makedirs(splits_dir, exist_ok=True)
    test_ids, val_ids = slide_ids[0:2], slide_ids[2:4]
    train_ids = slide_ids[4:]
    pd.DataFrame({
        "train": train_ids,
        "val": val_ids + [None] * (len(train_ids) - len(val_ids)),
        "test": test_ids + [None] * (len(train_ids) - len(test_ids)),
    }).to_csv(os.path.join(splits_dir, "splits_0.csv"), index=False)
    exp_cfg = ExperimentConfig(
        task=TaskConfig(name="brca", label_col="label",
                        label_dict={"neg": 0, "pos": 1}, n_classes=2),
        encoder_key=encoder, embed_dim=embed_dim,
        model=ModelConfig(model_type="clam_sb"),
        train=TrainConfig(max_epochs=max_epochs, early_stopping=False, seed=42),
        n_folds=1, framework=Framework.CLAM, strategy="standard",
    )
    dataset = create_dataset(exp_cfg, benchmark_dir)
    splits = load_fold_splits(
        dataset, benchmark_dir, os.path.join("standard", "brca"), 0, task_csv_name="brca",
    )
    return exp_cfg, splits


def _validate_clam_standin(aucs):
    """Honours validate_clam's seam contract while scripting the AUC the
    tracker sees: returns (stop, metrics) and feeds the callback."""
    calls = []

    def standin(cur, epoch, model, loader, n_classes, early_stopping=None,
                writer=None, loss_fn=None, results_dir=None):
        auc = aucs[min(len(calls), len(aucs) - 1)]
        calls.append(epoch)
        metrics = {"val_loss": 0.6 + 0.1 * epoch, "val_error": 0.5, "val_auc": auc}
        if early_stopping:
            early_stopping(epoch, auc, model,
                           ckpt_name=os.path.join(results_dir, f"s_{cur}_checkpoint.pt"))
        return False, metrics

    return standin


class TestCLAMFlagOffStillSelects:
    """`early_stopping` is a legal tunable knob. Flag off must mean "run every
    epoch"; the tracker still checkpoints and the restore keys on checkpoint
    existence, so a proposal cannot opt the arm out of the selection rule."""

    def test_flag_off_restores_the_auc_maximum(self, monkeypatch, capsys, tmp_path):
        import utils.core_utils as cu
        from autobench.pipeline.clam.train import train_fold
        from autobench.pipeline.policy_dispatch import PolicyRuntime

        monkeypatch.setattr(cu, "device", torch.device("cpu"))
        monkeypatch.setattr(cu, "validate_clam", _validate_clam_standin([0.60, 0.95, 0.90]))
        exp_cfg, (tr, va, te) = _clam_fixture(tmp_path, max_epochs=3)
        results_dir = str(tmp_path / "results")
        train_fold(exp_cfg, tr, va, te, fold=0, results_dir=results_dir,
                   device=torch.device("cpu"), policy_runtime=PolicyRuntime())
        out = capsys.readouterr().out
        assert "[selected] epoch=1 source=best" in out
        assert os.path.exists(os.path.join(results_dir, "fold_0", "s_0_checkpoint.pt"))

    def test_stale_prior_attempt_checkpoint_cannot_be_restored(self, monkeypatch, tmp_path):
        import utils.core_utils as cu
        from autobench.pipeline.clam.train import train_fold
        from autobench.pipeline.policy_dispatch import PolicyRuntime

        monkeypatch.setattr(cu, "device", torch.device("cpu"))
        monkeypatch.setattr(cu, "validate_clam", _validate_clam_standin([NAN]))
        exp_cfg, (tr, va, te) = _clam_fixture(tmp_path, max_epochs=1)
        results_dir = str(tmp_path / "results")
        fold_dir = os.path.join(results_dir, "fold_0")
        os.makedirs(fold_dir, exist_ok=True)
        with open(os.path.join(fold_dir, "s_0_checkpoint.pt"), "wb") as fh:
            fh.write(b"weights from a prior attempt")
        with pytest.raises(RuntimeError, match="non-finite validation AUC"):
            train_fold(exp_cfg, tr, va, te, fold=0, results_dir=results_dir,
                       device=torch.device("cpu"), policy_runtime=PolicyRuntime())


# ---------------------------------------------------------------------------
# The survival loops (real trainers, scripted C-index)
# ---------------------------------------------------------------------------


CINDEX = [0.50, 0.70, 0.60]  # maximum at epoch 1


def _scripted_cindex(values):
    seq = {"i": 0}

    def fake(*a, **kw):
        v = values[min(seq["i"], len(values) - 1)]
        seq["i"] += 1
        return v

    return fake


class TestSurvivalArmsSelectOnCIndex:
    def test_abmil_survival(self, monkeypatch, capsys, tmp_path):
        import h5py
        from autobench.pipeline.abmil import survival_train as st
        from autobench.pipeline.abmil.config import ABMILConfig
        from autobench.pipeline.abmil.dataset import ABMILSurvivalSlide

        rng = np.random.default_rng(11)

        def samples(prefix, n):
            out = []
            for i in range(n):
                path = str(tmp_path / f"{prefix}{i}.h5")
                with h5py.File(path, "w") as f:
                    f.create_dataset("features", data=rng.standard_normal((15, 32)).astype("float32"))
                out.append(ABMILSurvivalSlide(slide_id=f"{prefix}{i}", h5_path=path,
                                              status=i % 2, time=float(100 + 50 * i),
                                              patient_id=f"P{prefix}{i}"))
            return out

        monkeypatch.setattr(st, "survival_c_index", _scripted_cindex(CINDEX))
        fold_dir = str(tmp_path / "fold_0")
        os.makedirs(fold_dir, exist_ok=True)
        st.train_abmil_survival_fold(
            "abmil", samples("tr", 8), samples("va", 4), samples("te", 4),
            embed_dim=32, survival_loss="cox", nll_bins=4,
            cfg=ABMILConfig(M=16, L=8, max_epochs=3, early_stopping=False),
            device=torch.device("cpu"), seed=7, fold_dir=fold_dir,
        )
        assert "[selected] epoch=1 source=best" in capsys.readouterr().out

    def test_clam_survival(self, monkeypatch, capsys, tmp_path):
        import pandas as pd
        from autobench.pipeline.clam import survival_train as st
        from autobench.pipeline.config import (
            ExperimentConfig, Framework, ModelConfig, TaskConfig, TrainConfig,
        )

        rng = np.random.default_rng(5)
        benchmark_dir = str(tmp_path / "benchmark")
        pt_dir = os.path.join(benchmark_dir, "features", "e", "pt_files")
        os.makedirs(pt_dir, exist_ok=True)
        slide_ids = [f"s{i}" for i in range(16)]
        for sid in slide_ids:
            torch.save(torch.from_numpy(rng.standard_normal((15, 32)).astype("float32")),
                       os.path.join(pt_dir, f"{sid}.pt"))
        csv_dir = os.path.join(benchmark_dir, "dataset_csv")
        os.makedirs(csv_dir, exist_ok=True)
        pd.DataFrame({
            "slide_id": slide_ids, "case_id": [f"P{i}" for i in range(16)],
            "status": [i % 2 for i in range(16)], "time": [100.0 + 50 * i for i in range(16)],
        }).to_csv(os.path.join(csv_dir, "os.csv"), index=False)
        splits_dir = os.path.join(benchmark_dir, "splits", "standard", "os")
        os.makedirs(splits_dir, exist_ok=True)
        train_ids, val_ids, test_ids = slide_ids[:8], slide_ids[8:12], slide_ids[12:]
        pad = lambda ids: ids + [None] * (len(train_ids) - len(ids))
        pd.DataFrame({"train": train_ids, "val": pad(val_ids), "test": pad(test_ids)}).to_csv(
            os.path.join(splits_dir, "splits_0.csv"), index=False)

        monkeypatch.setattr(st, "survival_c_index", _scripted_cindex(CINDEX))
        exp_cfg = ExperimentConfig(
            task=TaskConfig(name="os", label_col="status", label_dict={}, n_classes=2,
                            task_type="survival"),
            encoder_key="e", embed_dim=32,
            model=ModelConfig(model_type="clam_sb"),
            train=TrainConfig(max_epochs=3, early_stopping=False, seed=3),
            n_folds=1, framework=Framework.CLAM, strategy="standard",
            survival_loss="cox",
        )
        st.train_survival_fold(exp_cfg, benchmark_dir, 0, str(tmp_path / "results"),
                               torch.device("cpu"))
        assert "[selected] epoch=1 source=best" in capsys.readouterr().out

    def test_titan_survival(self, monkeypatch, capsys, tmp_path):
        from autobench.pipeline.titan import survival_train as st
        from autobench.pipeline.config import (
            ExperimentConfig, Framework, ModelConfig, TaskConfig, TrainConfig,
        )

        class _DS(torch.utils.data.Dataset):
            def __init__(self, n):
                self.x = torch.randn(n, 768)
                self.status = torch.tensor([i % 2 for i in range(n)])
                self.time = torch.tensor([100.0 + 50 * i for i in range(n)])

            def __len__(self):
                return len(self.x)

            def __getitem__(self, i):
                return self.x[i], self.status[i], self.time[i], f"P{i}"

        monkeypatch.setattr(st, "survival_c_index", _scripted_cindex(CINDEX))
        cfg = ExperimentConfig(
            task=TaskConfig(name="os", label_col="status", label_dict={}, n_classes=2,
                            task_type="survival"),
            encoder_key="titan", embed_dim=768,
            model=ModelConfig(model_type="titan"),
            train=TrainConfig(max_epochs=3, patience=5, seed=0, early_stopping=False),
            n_folds=1, framework=Framework.TITAN, strategy="standard",
            survival_loss="cox",
        )
        st.train_titan_survival_fold(
            cfg, _DS(8), _DS(4), _DS(4), fold=0, results_dir=str(tmp_path / "r"), device="cpu",
        )
        assert "[selected] epoch=1 source=best" in capsys.readouterr().out

    def test_dtfd_survival(self, monkeypatch, capsys, tmp_path):
        import dataclasses
        import h5py
        from autobench.pipeline.dtfd import survival_train as st
        from autobench.pipeline.dtfd.config import DTFDConfig
        from autobench.pipeline.dtfd.dataset import DTFDSurvivalSlide

        rng = np.random.default_rng(2)

        def samples(prefix, n):
            out = []
            for i in range(n):
                path = str(tmp_path / f"{prefix}{i}.h5")
                with h5py.File(path, "w") as f:
                    f.create_dataset("features", data=rng.standard_normal((20, 32)).astype("float32"))
                out.append(DTFDSurvivalSlide(slide_id=f"{prefix}{i}", h5_path=path,
                                             status=i % 2, time=float(100 + 50 * i),
                                             patient_id=f"P{prefix}{i}"))
            return out

        monkeypatch.setattr(st, "_c_index", _scripted_cindex(CINDEX))
        fold_dir = str(tmp_path / "fold_0")
        os.makedirs(fold_dir, exist_ok=True)
        cfg = dataclasses.replace(DTFDConfig(), max_epochs=3, early_stopping=False)
        st.train_dtfd_survival_fold(
            samples("tr", 8), samples("va", 4), samples("te", 4),
            embed_dim=32, nll_bins=4, cfg=cfg, device=torch.device("cpu"),
            seed=1, fold_dir=fold_dir,
        )
        assert "[selected] epoch=1 source=best" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# nnMIL trainers: what they hand the callback
# ---------------------------------------------------------------------------


class _MeanBagModel(torch.nn.Module):
    def __init__(self, in_dim=8, n_out=2):
        super().__init__()
        self.head = torch.nn.Linear(in_dim, n_out)

    def forward(self, features, **kwargs):
        return self.head(features.float().mean(dim=1))


class _Recorder:
    """Stands in for a vendored callback: records the selection value the
    trainer passes and exposes the attributes the trainer reads."""

    instances: list = []

    def __init__(self, *a, **kw):
        self.values = []
        self.best_epoch = -1
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
        _Recorder.instances.append(self)

    def __call__(self, value, model, epoch=None):
        self.values.append(value)
        if self.best_model_state is None and math.isfinite(value):
            self.best_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            self.best_epoch = epoch


def _bare_trainer(cls, tmp_path, *, config, model, batches, evaluate, **extra):
    t = cls.__new__(cls)
    t.model = model
    t.train_loader = batches
    t.val_loader = batches
    t.test_loader = batches
    t.config = config
    t.device = torch.device("cpu")
    t.save_dir = str(tmp_path)
    t.model_type = "simple_mil"
    t.logger = logging.getLogger("nnmil-selection-test")
    t.writer = None
    t.dataset_info = {}
    t.policy_runtime = None
    t.save_training_config = lambda: None
    t.evaluate = evaluate
    for k, v in extra.items():
        setattr(t, k, v)
    return t


def _scripted(rows):
    seq = {"i": 0}

    def evaluate(split="val"):
        row = rows[min(seq["i"], len(rows) - 1)]
        seq["i"] += 1
        return dict(row)

    return evaluate


class TestNnMILTrainersFeedThePrimaryMetric:
    def _classification_trainer(self, tmp_path, rows, num_epochs):
        from training.trainers.classification_trainer import ClassificationTrainer
        batches = [(torch.randn(2, 5, 8), torch.zeros(2, 5, 2), torch.tensor([5, 5]),
                    torch.tensor([0, 1]))]
        return _bare_trainer(
            ClassificationTrainer, tmp_path,
            config={"num_epochs": num_epochs, "warmup_epochs": 0, "patience": 10,
                    "learning_rate": 1e-3},
            model=_MeanBagModel(), batches=batches, evaluate=_scripted(rows),
        )

    def test_classification_hands_val_auc_to_the_callback(self, monkeypatch, tmp_path):
        from training.trainers import classification_trainer as ct
        _Recorder.instances.clear()
        monkeypatch.setattr(ct, "EarlyStopping", _Recorder)
        rows = [{"val/loss": 0.4, "val/bacc": 0.5, "val/weighted_f1": 0.5, "val/auroc": 0.60},
                {"val/loss": 0.7, "val/bacc": 0.5, "val/weighted_f1": 0.5, "val/auroc": 0.95},
                {"val/loss": 0.6, "val/bacc": 0.5, "val/weighted_f1": 0.5}]  # auroc absent
        self._classification_trainer(tmp_path, rows, num_epochs=3).train()
        values = _Recorder.instances[-1].values
        assert values[:2] == [0.60, 0.95]
        assert math.isnan(values[2]), "a missing val/auroc must arrive as NaN, never 0.0"

    def test_classification_selects_the_auc_maximum(self, tmp_path, capsys):
        rows = [{"val/loss": 0.4, "val/bacc": 0.5, "val/weighted_f1": 0.5, "val/auroc": 0.60},
                {"val/loss": 0.7, "val/bacc": 0.5, "val/weighted_f1": 0.5, "val/auroc": 0.95},
                {"val/loss": 0.6, "val/bacc": 0.5, "val/weighted_f1": 0.5, "val/auroc": 0.90}]
        self._classification_trainer(tmp_path, rows, num_epochs=3).train()
        assert "[selected] epoch=1 source=best" in capsys.readouterr().out

    def test_cox_survival_hands_val_c_index_to_the_callback(self, monkeypatch, tmp_path):
        from training.trainers import survival_trainer as stm
        _Recorder.instances.clear()
        monkeypatch.setattr(stm, "EarlyStoppingSurvival", _Recorder)
        batches = [(torch.randn(4, 5, 8), torch.zeros(4, 5, 2), torch.tensor([5] * 4),
                    torch.tensor([1.0, 0.0, 1.0, 0.0]), torch.tensor([100.0, 200.0, 300.0, 400.0]),
                    ["p0", "p1", "p2", "p3"], ["s0", "s1", "s2", "s3"])]
        # validation starts at epoch 2, so five epochs give three validated ones
        rows = [{"val_c_index": 0.50}, {"val_c_index": 0.70}, {"val_c_index": 0.60}]
        t = _bare_trainer(
            stm.SurvivalTrainer, tmp_path,
            config={"num_epochs": 5, "warmup_epochs": 0, "patience": 10, "learning_rate": 1e-3},
            model=_MeanBagModel(n_out=1), batches=batches, evaluate=_scripted(rows),
            survival_loss="cox", _compute_val_loss=lambda loss_fn: 0.5,
        )
        t.train()
        assert _Recorder.instances[-1].values == [0.50, 0.70, 0.60]

    def test_nllsurv_survival_hands_val_c_index_to_the_callback(self, monkeypatch, tmp_path):
        from training.trainers import survival_porpoise_trainer as spm
        _Recorder.instances.clear()
        monkeypatch.setattr(spm, "EarlyStoppingSurvival", _Recorder)
        batches = [(torch.randn(1, 5, 8), torch.zeros(1, 5, 2), torch.tensor([5]),
                    torch.tensor([1.0]), torch.tensor([150.0]), ["p0"], ["s0"])]
        rows = [{"val_c_index": 0.50}, {"val_c_index": 0.70}, {"val_c_index": 0.60}]
        t = _bare_trainer(
            spm.SurvivalPorpoiseTrainer, tmp_path,
            config={"num_epochs": 3, "warmup_epochs": 0, "patience": 10, "learning_rate": 1e-3},
            model=_MeanBagModel(n_out=4), batches=batches, evaluate=_scripted(rows),
            nll_bin_edges=torch.tensor([0.0, 100.0, 200.0, 300.0, 1e9]),
            _compute_val_loss=lambda loss_fn: 0.5,
        )
        t.train()
        assert _Recorder.instances[-1].values == [0.50, 0.70, 0.60]


class TestCacheFingerprintCarriesProtocol:
    def test_protocol_version_in_payload(self):
        from autobench.pipeline.results_cache import fingerprint_payload
        from autobench.campaign import PROTOCOL_VERSION

        class _Cfg:
            def to_dict(self):
                return {"task": {"name": "t"}}

        assert fingerprint_payload(_Cfg())["protocol_version"] == PROTOCOL_VERSION
