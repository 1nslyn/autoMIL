"""Tests for the standalone ABMIL arm (pipeline/abmil/).

Covers, on tiny CPU fixtures:
  (a) model correctness -- forward shape, return-dict contract, gated vs.
      non-gated distinction, gradient flow (ported from the deleted
      test_abmil_gated.py, now against pipeline/abmil/model.py);
  (b) grid generation for Framework.ABMIL yields experiments per
      (task, encoder, abmil_model, fold);
  (c) dispatch (_run_single_experiment_dispatch) routes an ABMIL experiment to
      the real runner;
  (d) a fold trains and writes metrics.json in the shared schema;
  (e) run_abmil_experiment on a 2-fold H5 fixture writes summary.json.
"""

import itertools
import json
import os
import pathlib
import sys
import tempfile

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

from autobench.pipeline.abmil.config import ABMILConfig  # noqa: E402
from autobench.pipeline.abmil.dataset import ABMILSlide  # noqa: E402
from autobench.pipeline.abmil.model import ABMIL, ABMILGated, build_abmil_model  # noqa: E402
from autobench.pipeline.abmil.runner import run_abmil_experiment  # noqa: E402
from autobench.pipeline.abmil.train import train_abmil_fold  # noqa: E402
from autobench.pipeline.config import (  # noqa: E402
    BenchmarkConfig,
    ExperimentConfig,
    Framework,
    ModelConfig,
    TrainConfig,
    build_registries,
    generate_all_experiments,
)
from autobench.pipeline.orchestrator import _run_single_experiment_dispatch  # noqa: E402
from _helpers import make_test_ds  # noqa: E402


IN_DIM = 32
N_INSTANCES = 17
BATCH = 4
SHARED_KEYS = {
    "auc_roc", "accuracy", "balanced_accuracy", "f1", "sensitivity", "specificity",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(gated, num_classes=2, in_dim=IN_DIM, M=16, L=8):
    """Small dims for fast tests; production dims (500/128) checked separately."""
    cls = ABMILGated if gated else ABMIL
    return cls(in_dim=in_dim, M=M, L=L, num_classes=num_classes, dropout=0.0, K=1)


def _random_bag(batch=BATCH, n=N_INSTANCES, in_dim=IN_DIM):
    return torch.randn(batch, n, in_dim, requires_grad=False)


@pytest.fixture
def registries():
    return build_registries(make_test_ds())


# ---------------------------------------------------------------------------
# (a) Forward shape + return-dict contract (both variants)
# ---------------------------------------------------------------------------


class TestForwardShape:
    @pytest.mark.parametrize("gated", [False, True])
    def test_forward_returns_logits_key(self, gated):
        model = _make_model(gated, num_classes=2)
        x = _random_bag()
        out = model(x)
        assert isinstance(out, dict)
        assert "logits" in out

    @pytest.mark.parametrize("gated", [False, True])
    def test_logits_shape_binary(self, gated):
        model = _make_model(gated, num_classes=2)
        x = _random_bag()
        out = model(x)
        assert out["logits"].shape == (BATCH, 2)

    @pytest.mark.parametrize("gated", [False, True])
    @pytest.mark.parametrize("num_classes", [2, 3, 6, 7])
    def test_logits_shape_multiclass(self, gated, num_classes):
        """Must work for binary (num_classes=2) AND multi-class (>2, e.g. CLWD 6/7-class)."""
        model = _make_model(gated, num_classes=num_classes)
        x = _random_bag()
        out = model(x)
        assert out["logits"].shape == (BATCH, num_classes)

    @pytest.mark.parametrize("gated", [False, True])
    def test_no_sigmoid_probabilities_emitted(self, gated):
        """Logits, not bounded [0,1] sigmoid probabilities (unlike the reference's binary head)."""
        model = _make_model(gated, num_classes=2)
        x = _random_bag() * 10  # scale up so unclipped logits would likely exceed [0, 1]
        out = model(x)
        logits = out["logits"]
        assert (logits < 0).any() or (logits > 1).any()

    @pytest.mark.parametrize("gated", [False, True])
    def test_variable_bag_size(self, gated):
        model = _make_model(gated, num_classes=2)
        for n in [1, 5, 50]:
            x = torch.randn(2, n, IN_DIM)
            out = model(x)
            assert out["logits"].shape == (2, 2)


class TestReturnDictContract:
    @pytest.mark.parametrize("gated", [False, True])
    def test_return_WSI_feature(self, gated):
        model = _make_model(gated, num_classes=2, M=16)
        x = _random_bag()
        out = model(x, return_WSI_feature=True)
        assert "WSI_feature" in out
        assert out["WSI_feature"].shape == (BATCH, 16 * 1)  # K*M

    @pytest.mark.parametrize("gated", [False, True])
    def test_return_WSI_attn(self, gated):
        model = _make_model(gated, num_classes=2)
        x = _random_bag()
        out = model(x, return_WSI_attn=True)
        assert "WSI_attn" in out
        assert out["WSI_attn"].shape == (BATCH, N_INSTANCES, 1)  # [B, N, K]

    @pytest.mark.parametrize("gated", [False, True])
    def test_extra_keys_absent_by_default(self, gated):
        model = _make_model(gated, num_classes=2)
        x = _random_bag()
        out = model(x)
        assert "WSI_feature" not in out
        assert "WSI_attn" not in out

    @pytest.mark.parametrize("gated", [False, True])
    def test_both_extras_together(self, gated):
        model = _make_model(gated, num_classes=2)
        x = _random_bag()
        out = model(x, return_WSI_attn=True, return_WSI_feature=True)
        assert set(out.keys()) == {"logits", "WSI_feature", "WSI_attn"}


class TestAttentionWeights:
    @pytest.mark.parametrize("gated", [False, True])
    def test_attention_non_negative_and_sums_to_one(self, gated):
        model = _make_model(gated, num_classes=2)
        x = _random_bag()
        out = model(x, return_WSI_attn=True)
        attn = out["WSI_attn"]  # [B, N, K]
        assert (attn >= 0).all()
        summed = attn.sum(dim=1)  # sum over N instances -> [B, K]
        assert torch.allclose(summed, torch.ones_like(summed), atol=1e-5)

    @pytest.mark.parametrize("gated", [False, True])
    @pytest.mark.parametrize("n", [1, 3, 100])
    def test_attention_sums_to_one_various_bag_sizes(self, gated, n):
        model = _make_model(gated, num_classes=2)
        x = torch.randn(2, n, IN_DIM)
        out = model(x, return_WSI_attn=True)
        summed = out["WSI_attn"].sum(dim=1)
        assert torch.allclose(summed, torch.ones_like(summed), atol=1e-5)


class TestGradientFlow:
    @pytest.fixture(autouse=True)
    def _seed(self):
        # Determinism: with a tiny random batch a ReLU projection unit can be dead
        # across the whole batch, zeroing its gradient -- a batch-size artifact, not
        # a model bug. Seed so the gradient-flow assertions are stable.
        torch.manual_seed(0)

    @pytest.mark.parametrize("gated", [False, True])
    def test_gradients_reach_projection(self, gated):
        model = _make_model(gated, num_classes=2)
        x = _random_bag()
        loss = model(x)["logits"].sum()
        loss.backward()
        proj_linear = model.feature_extractor[0]
        assert proj_linear.weight.grad is not None
        assert torch.any(proj_linear.weight.grad != 0)

    def test_gradients_reach_attention_non_gated(self):
        model = _make_model(False, num_classes=2)
        x = _random_bag()
        loss = model(x)["logits"].sum()
        loss.backward()
        attn_linear = model.attention[0]
        assert attn_linear.weight.grad is not None
        assert torch.any(attn_linear.weight.grad != 0)

    def test_gradients_reach_attention_V_branch_gated(self):
        model = _make_model(True, num_classes=2)
        x = _random_bag()
        loss = model(x)["logits"].sum()
        loss.backward()
        v_linear = model.attention_V[0]
        assert v_linear.weight.grad is not None
        assert torch.any(v_linear.weight.grad != 0)

    def test_gradients_reach_attention_U_branch_gated(self):
        model = _make_model(True, num_classes=2)
        x = _random_bag()
        loss = model(x)["logits"].sum()
        loss.backward()
        u_linear = model.attention_U[0]
        assert u_linear.weight.grad is not None
        assert torch.any(u_linear.weight.grad != 0)

    @pytest.mark.parametrize("gated", [False, True])
    def test_gradients_reach_classifier(self, gated):
        model = _make_model(gated, num_classes=2)
        x = _random_bag()
        loss = model(x)["logits"].sum()
        loss.backward()
        assert model.classifier.weight.grad is not None
        assert torch.any(model.classifier.weight.grad != 0)

    @pytest.mark.parametrize("gated", [False, True])
    def test_gradients_reach_all_params_via_ce_loss(self, gated):
        """End-to-end: the CE training objective is autograd-connected to every param."""
        model = _make_model(gated, num_classes=3)
        x = _random_bag()
        y = torch.randint(0, 3, (BATCH,))
        logits = model(x)["logits"]
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"no gradient reached {name}"
        assert sum(p.grad.abs().sum() for p in model.parameters()) > 0


class TestGatedVsNonGatedFidelity:
    """Distinguish the two variants: gated has both Tanh and Sigmoid branches; non-gated does not."""

    def test_gated_has_tanh_branch(self):
        model = _make_model(True, num_classes=2)
        assert isinstance(model.attention_V[1], torch.nn.Tanh)

    def test_gated_has_sigmoid_branch(self):
        model = _make_model(True, num_classes=2)
        assert isinstance(model.attention_U[1], torch.nn.Sigmoid)

    def test_non_gated_lacks_gate_branches(self):
        model = _make_model(False, num_classes=2)
        assert not hasattr(model, "attention_U")
        assert not hasattr(model, "attention_V")
        assert hasattr(model, "attention")

    def test_gating_is_multiplicative(self):
        """A = attention_w(V(H) * U(H)) -- element-wise product of both branches, per the reference."""
        model = _make_model(True, num_classes=2, M=16, L=8)
        x = _random_bag()
        H = model.feature_extractor(x)
        A_V = model.attention_V(H)
        A_U = model.attention_U(H)
        expected_pre_softmax = model.attention_w(A_V * A_U).transpose(-1, -2)
        actual = torch.nn.functional.softmax(expected_pre_softmax, dim=-1)
        out = model(x, return_WSI_attn=True)
        actual_attn = out["WSI_attn"].transpose(-1, -2)
        assert torch.allclose(actual, actual_attn, atol=1e-6)

    def test_zeroing_either_gate_changes_attention(self):
        """Sanity check both branches actually participate: removing either changes A."""
        model = _make_model(True, num_classes=2, M=16, L=8)
        x = _random_bag()
        with torch.no_grad():
            baseline = model(x, return_WSI_attn=True)["WSI_attn"].clone()
            model.attention_U[0].weight.zero_()
            model.attention_U[0].bias.zero_()
            after_zero_U = model(x, return_WSI_attn=True)["WSI_attn"]
        assert not torch.allclose(baseline, after_zero_U, atol=1e-6)


class TestPaperExactDims:
    def test_build_abmil_dims_are_paper_exact_by_default(self):
        """Locked decision: both variants default to M=500, L=128 (Ilse et al. 2018)."""
        gated = build_abmil_model("abmil_gated", in_dim=1024, num_classes=2)
        non_gated = build_abmil_model("abmil", in_dim=1024, num_classes=2)
        assert isinstance(gated, ABMILGated)
        assert gated.M == 500 and gated.L == 128
        assert isinstance(non_gated, ABMIL)
        assert non_gated.M == 500 and non_gated.L == 128

    def test_build_abmil_forward_smoke(self):
        model = build_abmil_model("abmil_gated", in_dim=64, num_classes=4)
        x = torch.randn(2, 10, 64)
        out = model(x)
        assert out["logits"].shape == (2, 4)

    def test_abmil_key_is_non_gated_abmil_gated_key_is_gated(self):
        """Clean separation: `abmil` = original non-gated, `abmil_gated` = gated."""
        gated = build_abmil_model("abmil_gated", in_dim=64, num_classes=2)
        non_gated = build_abmil_model("abmil", in_dim=64, num_classes=2)
        assert isinstance(gated, ABMILGated)
        assert not isinstance(non_gated, ABMILGated)
        assert hasattr(gated, "attention_U")
        assert not hasattr(non_gated, "attention_U")

    def test_build_abmil_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown ABMIL model_type"):
            build_abmil_model("bogus", in_dim=64, num_classes=2)


class TestInputValidation:
    @pytest.mark.parametrize("gated", [False, True])
    def test_rejects_wrong_ndim(self, gated):
        model = _make_model(gated, num_classes=2)
        with pytest.raises(ValueError):
            model(torch.randn(N_INSTANCES, IN_DIM))  # missing batch dim

    @pytest.mark.parametrize("gated", [False, True])
    def test_rejects_wrong_feature_dim(self, gated):
        model = _make_model(gated, num_classes=2, in_dim=IN_DIM)
        with pytest.raises(ValueError):
            model(torch.randn(BATCH, N_INSTANCES, IN_DIM + 1))

    def test_rejects_k_other_than_one_non_gated(self):
        with pytest.raises(ValueError):
            ABMIL(in_dim=IN_DIM, num_classes=2, K=2)

    def test_rejects_k_other_than_one_gated(self):
        with pytest.raises(ValueError):
            ABMILGated(in_dim=IN_DIM, num_classes=2, K=2)


# ---------------------------------------------------------------------------
# Fixture builders for grid / runner / dispatch tests
# ---------------------------------------------------------------------------


#: Bags are lazy (``h5_path``, not a tensor), so fixtures write real H5 files
#: and the trainer reads them back through the same ``_read_bag`` production
#: uses. A counter keeps filenames unique across tests that reuse slide ids.
_BAG_DIR = pathlib.Path(tempfile.mkdtemp(prefix="abmil-bags-"))
_BAG_SEQ = itertools.count()


def _make_slide(rng, sid, label, n=N_INSTANCES, emb=IN_DIM, sep=3.0):
    """Class-separable bag so the training loop has a learnable signal."""
    feats = rng.standard_normal((n, emb)).astype("float32") + label * sep
    path = _BAG_DIR / f"{sid}-{next(_BAG_SEQ)}.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=feats)
    return ABMILSlide(slide_id=sid, h5_path=str(path), label=label)


def _make_split(rng, prefix, n_slides):
    return [_make_slide(rng, f"{prefix}{i}", i % 2) for i in range(n_slides)]


def _smoke_cfg():
    return ABMILConfig(M=16, L=8, max_epochs=5, lr=1e-2, early_stopping=False)


def _write_h5_bag(path, rng, label, n=N_INSTANCES, emb=IN_DIM, sep=3.0):
    feats = rng.standard_normal((n, emb)).astype("float32") + label * sep
    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=feats)


def _build_benchmark_fixture(root, task="brca", strategy="standard", encoder="conch_v15",
                              n_folds=2, n_slides=12):
    """Lay out the on-disk artifacts run_abmil_experiment reads.

    Creates: features_<encoder>/<sid>.h5, dataset_csv/<task>.csv,
    splits/<strategy>/<task>/splits_<fold>.csv, and the nnMIL
    dataset_plan.json (whose feature_dir the ABMIL runner resolves).
    """
    rng = np.random.default_rng(7)
    h5_dir = os.path.join(root, "features", f"features_{encoder}")
    os.makedirs(h5_dir, exist_ok=True)

    slide_ids = [f"s{i}" for i in range(n_slides)]
    labels = {sid: (i % 2) for i, sid in enumerate(slide_ids)}
    for sid in slide_ids:
        _write_h5_bag(os.path.join(h5_dir, f"{sid}.h5"), rng, labels[sid])

    csv_dir = os.path.join(root, "dataset_csv")
    os.makedirs(csv_dir, exist_ok=True)
    label_names = {0: "neg", 1: "pos"}
    pd.DataFrame({
        "slide_id": slide_ids,
        "case_id": slide_ids,
        "label": [label_names[labels[sid]] for sid in slide_ids],
    }).to_csv(os.path.join(csv_dir, f"{task}.csv"), index=False)

    splits_dir = os.path.join(root, "splits", strategy, task)
    os.makedirs(splits_dir, exist_ok=True)
    for fold in range(n_folds):
        test_ids = [slide_ids[fold * 2], slide_ids[fold * 2 + 1]]
        val_ids = [slide_ids[(fold * 2 + 2) % n_slides], slide_ids[(fold * 2 + 3) % n_slides]]
        train_ids = [s for s in slide_ids if s not in test_ids and s not in val_ids]
        col = {
            "train": train_ids,
            "val": val_ids + [None] * (len(train_ids) - len(val_ids)),
            "test": test_ids + [None] * (len(train_ids) - len(test_ids)),
        }
        pd.DataFrame(col).to_csv(os.path.join(splits_dir, f"splits_{fold}.csv"), index=False)

    # nnMIL dataset_plan.json -- ABMIL reuses nnMIL prep; only feature_dir is read.
    plan_dir = os.path.join(root, "nnmil", strategy, f"{task}_{encoder}")
    os.makedirs(plan_dir, exist_ok=True)
    with open(os.path.join(plan_dir, "dataset_plan.json"), "w") as f:
        json.dump({"feature_dir": h5_dir}, f)

    return h5_dir


def _exp_cfg(registries, task="brca", encoder="conch_v15", n_folds=2, model_type="abmil"):
    return ExperimentConfig(
        task=registries.task_registry[task],
        encoder_key=encoder,
        embed_dim=IN_DIM,
        model=ModelConfig(model_type=model_type),
        train=TrainConfig(seed=42),
        n_folds=n_folds,
        framework=Framework.ABMIL,
        strategy="standard",
    )


# ---------------------------------------------------------------------------
# (b) Grid generation for Framework.ABMIL
# ---------------------------------------------------------------------------


class TestAbmilGrid:
    def _cfg(self, ds, **kw):
        params = dict(
            frameworks=[Framework.ABMIL], strategies=["standard"],
            tasks=["brca"], encoder_keys=["conch_v15"],
        )
        params.update(kw)
        return BenchmarkConfig.from_dataset_config(ds, **params)

    def test_uses_abmil_models(self, registries):
        # make_test_ds sets abmil_models=["abmil", "abmil_gated"].
        ds = make_test_ds()
        exps = generate_all_experiments(self._cfg(ds), registries)
        assert len(exps) == 2
        assert all(e.framework == Framework.ABMIL for e in exps)
        assert {e.model.model_type for e in exps} == {"abmil", "abmil_gated"}

    def test_experiment_id_format(self, registries):
        ds = make_test_ds()
        exps = generate_all_experiments(self._cfg(ds), registries)
        ids = {e.experiment_id for e in exps}
        assert "abmil__standard__brca__conch_v15__abmil__s42" in ids
        assert "abmil__standard__brca__conch_v15__abmil_gated__s42" in ids

    def test_results_subdir(self, registries):
        ds = make_test_ds()
        exps = generate_all_experiments(self._cfg(ds), registries)
        subdirs = {e.results_subdir for e in exps}
        assert "abmil/standard/brca/conch_v15/abmil/s42" in subdirs

    def test_sweeps_encoders(self, registries):
        ds = make_test_ds()
        cfg = self._cfg(ds, encoder_keys=["conch_v15", "uni_v2"])
        exps = generate_all_experiments(cfg, registries)
        assert len(exps) == 4  # 2 encoders x 2 abmil model types
        assert {e.encoder_key for e in exps} == {"conch_v15", "uni_v2"}


# ---------------------------------------------------------------------------
# (c)/(d) Fold trains + writes metrics; abmil != gated attention-wise
# ---------------------------------------------------------------------------


class TestFoldTraining:
    @pytest.mark.parametrize("model_type", ["abmil", "abmil_gated"])
    def test_fold_trains_and_returns_shared_schema(self, model_type):
        rng = np.random.default_rng(0)
        train = _make_split(rng, "t", 12)
        val = _make_split(rng, "v", 4)
        test = _make_split(rng, "e", 4)

        result = train_abmil_fold(
            model_type, train, val, test, embed_dim=IN_DIM, num_classes=2,
            cfg=_smoke_cfg(), device=torch.device("cpu"), seed=42,
        )

        assert set(result["test_metrics"].keys()) == SHARED_KEYS
        assert set(result["val_metrics"].keys()) == SHARED_KEYS
        assert "elapsed_seconds" in result

    def test_restores_pristine_grad_state(self):
        """Must not leave torch grad disabled (would corrupt a following arm)."""
        rng = np.random.default_rng(1)
        torch.set_grad_enabled(True)
        train_abmil_fold(
            "abmil", _make_split(rng, "t", 6), _make_split(rng, "v", 2),
            _make_split(rng, "e", 2), embed_dim=IN_DIM, num_classes=2,
            cfg=_smoke_cfg(), device=torch.device("cpu"), seed=3,
        )
        assert torch.is_grad_enabled() is True

    def test_abmil_and_abmil_gated_are_different_architectures(self):
        """abmil != gated (no attention_U); abmil_gated == gated (has attention_U)."""
        rng = np.random.default_rng(2)
        train = _make_split(rng, "t", 8)

        non_gated_model = build_abmil_model("abmil", in_dim=IN_DIM, num_classes=2, M=16, L=8)
        gated_model = build_abmil_model("abmil_gated", in_dim=IN_DIM, num_classes=2, M=16, L=8)
        assert not hasattr(non_gated_model, "attention_U")
        assert hasattr(gated_model, "attention_U")
        assert type(non_gated_model) is not type(gated_model)
        del train  # only used to document the intended usage context above


# ---------------------------------------------------------------------------
# (e) Runner writes summary.json in the shared schema over a 2-fold fixture
# ---------------------------------------------------------------------------


class TestRunnerSummary:
    @pytest.mark.parametrize("model_type", ["abmil", "abmil_gated"])
    def test_run_abmil_experiment_writes_summary(self, tmp_path, registries, model_type):
        _build_benchmark_fixture(str(tmp_path), n_folds=2)
        exp = _exp_cfg(registries, n_folds=2, model_type=model_type)

        summary = run_abmil_experiment(exp, str(tmp_path), device="cpu", cfg=_smoke_cfg())

        for key in (
            "experiment_id", "task", "encoder", "embed_dim", "model_type",
            "framework", "strategy", "n_folds", "seed", "test", "val",
            "per_fold_test", "per_fold_val",
        ):
            assert key in summary, f"missing summary key: {key}"
        assert summary["framework"] == "abmil"
        assert summary["model_type"] == model_type
        assert summary["n_folds"] == 2
        assert len(summary["per_fold_test"]) == 2

        summary_path = os.path.join(
            str(tmp_path), "results", exp.results_subdir, "summary.json"
        )
        assert os.path.exists(summary_path)
        on_disk = json.loads(open(summary_path).read())
        assert on_disk["experiment_id"] == exp.experiment_id

        for fold in range(2):
            fm_path = os.path.join(
                str(tmp_path), "results", exp.results_subdir, f"fold_{fold}", "metrics.json"
            )
            assert os.path.exists(fm_path)
            fm = json.loads(open(fm_path).read())
            assert set(fm["test_metrics"].keys()) == SHARED_KEYS

    def test_runner_missing_plan_fails_fast(self, tmp_path, registries):
        exp = _exp_cfg(registries, n_folds=1)
        with pytest.raises(FileNotFoundError, match="plan not found"):
            run_abmil_experiment(exp, str(tmp_path), device="cpu", cfg=_smoke_cfg())


# ---------------------------------------------------------------------------
# (c) Dispatch routes ABMIL to the real runner
# ---------------------------------------------------------------------------


class TestDispatchRunsRealArm:
    def test_dispatch_abmil_runs_instead_of_raising(self, tmp_path, registries):
        _build_benchmark_fixture(str(tmp_path), n_folds=2)
        exp = _exp_cfg(registries, n_folds=2)

        summary = _run_single_experiment_dispatch(
            exp, str(tmp_path), torch.device("cpu"),
        )
        assert summary["framework"] == "abmil"
        assert summary["model_type"] == "abmil"
        assert "test" in summary and "val" in summary
