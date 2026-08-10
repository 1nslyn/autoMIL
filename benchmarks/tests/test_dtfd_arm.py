"""Tests for the REAL two-tier DTFD-MIL arm (pipeline/dtfd/).

Covers, on tiny CPU fixtures (design spec §6, §10):
  (a) SMOKE: the two-tier loop learns — tier-2 training loss decreases from the
      first to the last epoch — and a valid shared-schema metrics.json is written;
  (b) the model instantiates all FOUR reference modules (two-tier, not a
      single-tier stand-in);
  (c) run_dtfd_experiment on a 2-fold H5 fixture writes summary.json in the
      shared schema;
  (d) dispatch (_run_single_experiment_dispatch) routes a DTFD experiment to the
      real runner instead of raising.
"""

import json
import math
import os
import sys

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))

from autobench.pipeline.config import (  # noqa: E402
    ExperimentConfig,
    Framework,
    ModelConfig,
    TaskConfig,
    TrainConfig,
    build_registries,
)
from autobench.pipeline.dtfd._imports import (  # noqa: E402
    Attention_Gated,
    Attention_with_Classifier,
    Classifier_1fc,
    DimReduction,
)
from autobench.pipeline.dtfd.config import DTFDConfig  # noqa: E402
from autobench.pipeline.dtfd.dataset import (  # noqa: E402
    DTFDSlide,
    DTFDSurvivalSlide,
)
from autobench.pipeline.dtfd.model import build_dtfd_bundle  # noqa: E402
from autobench.pipeline.dtfd.runner import run_dtfd_experiment  # noqa: E402
from autobench.pipeline.dtfd.survival_train import (  # noqa: E402
    train_dtfd_survival_fold,
)
from autobench.pipeline.dtfd.train import train_dtfd_fold  # noqa: E402
from autobench.pipeline.policy_dispatch import PolicyRuntime  # noqa: E402
from autobench.pipeline.orchestrator import _run_single_experiment_dispatch  # noqa: E402
from _helpers import make_test_ds  # noqa: E402


EMB = 64
N_PATCHES = 20
SHARED_KEYS = {
    "auc_roc", "accuracy", "balanced_accuracy", "f1", "sensitivity", "specificity",
}


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_slide(rng, sid, label, n=N_PATCHES, emb=EMB, sep=3.0):
    """Class-separable bag so the two-tier loop has a learnable signal."""
    feats = rng.standard_normal((n, emb)).astype("float32") + label * sep
    return DTFDSlide(slide_id=sid, features=torch.from_numpy(feats), label=label)


def _make_split(rng, prefix, n_slides):
    return [_make_slide(rng, f"{prefix}{i}", i % 2) for i in range(n_slides)]


def _smoke_cfg():
    # lr/epochs raised over the 1e-4/200-epoch reference default purely so the
    # learning signal is visible in a <30s CPU test; numGroup=2 per the spec.
    return DTFDConfig(
        numGroup=2, mDim=32, max_epochs=10, lr=1e-3, early_stopping=False,
    )


def _write_h5_bag(path, rng, label, n=N_PATCHES, emb=EMB, sep=3.0):
    feats = rng.standard_normal((n, emb)).astype("float32") + label * sep
    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=feats)


def _build_benchmark_fixture(root, task="brca", strategy="standard", encoder="conch_v15",
                             n_folds=2, n_slides=12):
    """Lay out the on-disk artifacts run_dtfd_experiment reads.

    Creates: features_<encoder>/<sid>.h5, dataset_csv/<task>.csv,
    splits/<strategy>/<task>/splits_<fold>.csv, and the nnMIL
    dataset_plan.json (whose feature_dir the DTFD runner resolves).
    """
    rng = np.random.default_rng(7)
    h5_dir = os.path.join(root, "features", f"features_{encoder}")
    os.makedirs(h5_dir, exist_ok=True)

    slide_ids = [f"s{i}" for i in range(n_slides)]
    labels = {sid: (i % 2) for i, sid in enumerate(slide_ids)}
    for sid in slide_ids:
        _write_h5_bag(os.path.join(h5_dir, f"{sid}.h5"), rng, labels[sid])

    # dataset_csv/<task>.csv — slide_id + label (string labels mapped by label_dict)
    csv_dir = os.path.join(root, "dataset_csv")
    os.makedirs(csv_dir, exist_ok=True)
    import pandas as pd
    label_names = {0: "neg", 1: "pos"}
    pd.DataFrame({
        "slide_id": slide_ids,
        "case_id": slide_ids,
        "label": [label_names[labels[sid]] for sid in slide_ids],
    }).to_csv(os.path.join(csv_dir, f"{task}.csv"), index=False)

    # splits/<strategy>/<task>/splits_<fold>.csv — rotate the held-out pair
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

    # nnMIL dataset_plan.json — DTFD reuses nnMIL prep; only feature_dir is read.
    plan_dir = os.path.join(root, "nnmil", strategy, f"{task}_{encoder}")
    os.makedirs(plan_dir, exist_ok=True)
    with open(os.path.join(plan_dir, "dataset_plan.json"), "w") as f:
        json.dump({"feature_dir": h5_dir}, f)

    return h5_dir


def _exp_cfg(registries, task="brca", encoder="conch_v15", n_folds=2):
    return ExperimentConfig(
        task=registries.task_registry[task],
        encoder_key=encoder,
        embed_dim=EMB,
        model=ModelConfig(model_type="dtfd_mil"),
        train=TrainConfig(seed=42),
        n_folds=n_folds,
        framework=Framework.DTFD,
        strategy="standard",
    )


@pytest.fixture
def registries():
    return build_registries(make_test_ds())


# ---------------------------------------------------------------------------
# (b) Model is genuinely two-tier (all four reference modules)
# ---------------------------------------------------------------------------


class TestTwoTierModel:
    def test_bundle_instantiates_all_four_reference_modules(self):
        bundle = build_dtfd_bundle(embed_dim=EMB, num_classes=2, cfg=_smoke_cfg())
        # These are the VENDORED reference classes, imported not reimplemented.
        assert isinstance(bundle.dim_reduction, DimReduction)
        assert isinstance(bundle.attention, Attention_Gated)
        assert isinstance(bundle.classifier, Classifier_1fc)
        assert isinstance(bundle.att_cls, Attention_with_Classifier)

    def test_tier2_is_attention_with_classifier_not_single_tier(self):
        """The tier-2 head must be Attention_with_Classifier (contains its own
        gated attention + classifier), which is what makes this two-tier —
        not a bare tier-1 stand-in like the nnMIL dtfd_mil wrapper."""
        bundle = build_dtfd_bundle(embed_dim=EMB, num_classes=3, cfg=_smoke_cfg())
        assert isinstance(bundle.att_cls.attention, Attention_Gated)
        assert isinstance(bundle.att_cls.classifier, Classifier_1fc)

    def test_numgroup_guard_rejects_too_many_groups(self):
        cfg = DTFDConfig(numGroup=50)
        with pytest.raises(ValueError, match="exceeds the smallest bag"):
            cfg.validate(n_patches=20)


# ---------------------------------------------------------------------------
# (a) SMOKE — the two-tier loop learns; metrics.json is valid shared schema
# ---------------------------------------------------------------------------


class TestSmokeTraining:
    def test_tier2_loss_decreases_and_metrics_written(self, tmp_path):
        rng = np.random.default_rng(0)
        train = _make_split(rng, "t", 12)
        val = _make_split(rng, "v", 4)
        test = _make_split(rng, "e", 4)

        result = train_dtfd_fold(
            train, val, test, embed_dim=EMB, num_classes=2, cfg=_smoke_cfg(),
            device=torch.device("cpu"), seed=42, return_history=True,
        )

        history = result["epoch_tier2_loss"]
        assert len(history) == 10
        # The real signal that the two-tier objective is learning.
        assert history[-1] < history[0], (
            f"tier-2 loss did not decrease: first={history[0]:.4f} "
            f"last={history[-1]:.4f}"
        )

        # Metrics are the shared schema (same keys CLAM/nnMIL/ABMIL produce).
        assert set(result["test_metrics"].keys()) == SHARED_KEYS
        assert set(result["val_metrics"].keys()) == SHARED_KEYS

        # And a metrics.json in that schema can be persisted/reloaded.
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps(result["test_metrics"]))
        reloaded = json.loads(metrics_path.read_text())
        assert set(reloaded.keys()) == SHARED_KEYS

    def test_restores_pristine_grad_state(self):
        """Must not leave torch grad disabled (would corrupt a following arm)."""
        rng = np.random.default_rng(1)
        torch.set_grad_enabled(True)
        train_dtfd_fold(
            _make_split(rng, "t", 6), _make_split(rng, "v", 2),
            _make_split(rng, "e", 2), embed_dim=EMB, num_classes=2,
            cfg=_smoke_cfg(), device=torch.device("cpu"), seed=3,
        )
        assert torch.is_grad_enabled() is True


# ---------------------------------------------------------------------------
# (c) Runner writes summary.json in the shared schema over a 2-fold fixture
# ---------------------------------------------------------------------------


class TestRunnerSummary:
    def test_run_dtfd_experiment_writes_summary(self, tmp_path, registries):
        _build_benchmark_fixture(str(tmp_path), n_folds=2)
        exp = _exp_cfg(registries, n_folds=2)

        summary = run_dtfd_experiment(exp, str(tmp_path), device="cpu", cfg=_smoke_cfg())

        # Summary schema matches CLAM/nnMIL runners exactly.
        for key in (
            "experiment_id", "task", "encoder", "embed_dim", "model_type",
            "framework", "strategy", "n_folds", "seed", "test", "val",
            "per_fold_test", "per_fold_val",
        ):
            assert key in summary, f"missing summary key: {key}"
        assert summary["framework"] == "dtfd"
        assert summary["model_type"] == "dtfd_mil"
        assert summary["n_folds"] == 2
        assert len(summary["per_fold_test"]) == 2

        # summary.json is written to results/<subdir>/summary.json
        summary_path = os.path.join(
            str(tmp_path), "results", exp.results_subdir, "summary.json"
        )
        assert os.path.exists(summary_path)
        on_disk = json.loads(open(summary_path).read())
        assert on_disk["experiment_id"] == exp.experiment_id

        # Per-fold metrics.json exists in the shared schema.
        for fold in range(2):
            fm_path = os.path.join(
                str(tmp_path), "results", exp.results_subdir, f"fold_{fold}", "metrics.json"
            )
            assert os.path.exists(fm_path)
            fm = json.loads(open(fm_path).read())
            assert set(fm["test_metrics"].keys()) == SHARED_KEYS

    def test_runner_missing_plan_fails_fast(self, tmp_path, registries):
        # No fixture laid down -> the nnMIL plan is absent -> clear error.
        exp = _exp_cfg(registries, n_folds=1)
        with pytest.raises(FileNotFoundError, match="plan not found"):
            run_dtfd_experiment(exp, str(tmp_path), device="cpu", cfg=_smoke_cfg())


# ---------------------------------------------------------------------------
# (d) Dispatch routes DTFD to the real runner (no longer raises)
# ---------------------------------------------------------------------------


class TestDispatchRunsRealArm:
    def test_dispatch_dtfd_runs_instead_of_raising(self, tmp_path, registries):
        _build_benchmark_fixture(str(tmp_path), n_folds=2)
        exp = _exp_cfg(registries, n_folds=2)

        # Would have raised NotImplementedError while stubbed; now it completes.
        summary = _run_single_experiment_dispatch(
            exp, str(tmp_path), torch.device("cpu"),
        )
        assert summary["framework"] == "dtfd"
        assert summary["model_type"] == "dtfd_mil"
        assert "test" in summary and "val" in summary


class TestLoadSplitIdsPreservesUuid:
    """Regression: TCGA slide_ids carry a '.' before the UUID suffix; the split
    loader must NOT strip it. The old ``split('.')[0]`` stripped the UUID so the
    id matched no H5/label and silently emptied the split -> the observed
    'DTFD train split is empty' failure on TCGA-LUAD."""

    def test_uuid_suffixed_id_preserved(self, tmp_path):
        from autobench.pipeline.dtfd.dataset import _load_split_ids

        sid = "TCGA-05-4244-01Z-00-DX1.d4ff32cd-38cf-40ea-8213-45c2b100ac01"
        csv = tmp_path / "splits_0.csv"
        csv.write_text(f"train,val,test\n{sid},{sid},{sid}\n")
        assert _load_split_ids(str(csv), "train") == [sid]

    def test_numeric_float_coercion_still_stripped(self, tmp_path):
        from autobench.pipeline.dtfd.dataset import _load_split_ids

        csv = tmp_path / "splits_0.csv"
        csv.write_text("train,val,test\n1234.0,5678.0,9012.0\n")
        assert _load_split_ids(str(csv), "train") == ["1234"]


# ---------------------------------------------------------------------------
# Survival (nllsurv) — the path that shipped broken
# ---------------------------------------------------------------------------


def _make_survival_slide(tmpdir, rng, sid, status, time_val, sep=3.0):
    """One survival bag on disk; risk signal correlates with the event status."""
    path = os.path.join(tmpdir, f"{sid}.h5")
    _write_h5_bag(path, rng, status, sep=sep)
    return DTFDSurvivalSlide(
        slide_id=sid, h5_path=path, status=status, time=time_val, patient_id=sid,
    )


def _survival_split(tmpdir, rng, prefix, n_slides):
    return [
        _make_survival_slide(
            tmpdir, rng, f"{prefix}{i}", status=i % 2, time_val=10.0 + 5.0 * (i % 4),
        )
        for i in range(n_slides)
    ]


def test_dtfd_survival_fold_runs_end_to_end(tmp_path):
    """DTFD survival trains a fold and returns the shared c-index schema.

    Regression test for a defect that made this path unreachable: the runner
    passed ``policy_runtime=`` to a trainer that never declared it, so every
    survival cell died with ``TypeError`` before its first epoch. No test
    exercised survival, so five datasets' ``os`` cells were only ever filled by
    an older code version.
    """
    rng = np.random.default_rng(11)
    bags = tmp_path / "bags"
    bags.mkdir()
    train = _survival_split(str(bags), rng, "tr", 12)
    val = _survival_split(str(bags), rng, "va", 6)
    test = _survival_split(str(bags), rng, "te", 6)

    result = train_dtfd_survival_fold(
        train, val, test,
        embed_dim=EMB, nll_bins=4, cfg=_smoke_cfg(),
        device=torch.device("cpu"), seed=0,
        policy_runtime=PolicyRuntime(),
    )

    assert set(result) == {
        "test_metrics", "val_metrics", "val_records", "elapsed_seconds",
    }
    assert "c_index" in result["test_metrics"]
    assert "c_index" in result["val_metrics"]
    # CR-3: val risk records must carry one entry per val slide for pooling.
    records = result["val_records"]
    assert len(records["risks"]) == len(val)
    assert len(records["patient_ids"]) == len(val)
    assert result["elapsed_seconds"] > 0


def test_dtfd_survival_fold_defaults_policy_runtime(tmp_path):
    """Omitting ``policy_runtime`` must not raise ``UnboundLocalError``.

    The body does ``policy_runtime = policy_runtime or PolicyRuntime()``, which
    only works because the name is a parameter. Calling without it is the exact
    shape that would regress if the parameter were dropped again.
    """
    rng = np.random.default_rng(12)
    bags = tmp_path / "bags"
    bags.mkdir()
    train = _survival_split(str(bags), rng, "tr", 8)
    val = _survival_split(str(bags), rng, "va", 4)

    result = train_dtfd_survival_fold(
        train, val, [],
        embed_dim=EMB, nll_bins=4, cfg=_smoke_cfg(),
        device=torch.device("cpu"), seed=0,
    )
    assert "c_index" in result["val_metrics"]
    # No test split: the trainer reports an unestimable c-index, not a crash.
    assert math.isnan(result["test_metrics"]["c_index"])


# ---------------------------------------------------------------------------
# Survival THROUGH THE RUNNER — the call path a campaign cell actually takes
# ---------------------------------------------------------------------------
#
# The two tests above call ``train_dtfd_survival_fold`` directly, which pins the
# trainer but NOT the hand-off that broke: it was ``runner.py`` supplying a
# keyword the trainer never declared. A direct call cannot reproduce that, and
# neither can a static check — this repo has been bitten before by wiring that
# satisfies a grep or ``py_compile`` and dies at launch. These tests execute
# ``run_dtfd_experiment`` itself on the survival branch, so the runner's own
# survival-only code (loss-suffixed plan leaf, ``load_dtfd_survival_split``, the
# cox guard, the pooled val block) runs rather than merely parsing.


def _survival_task_cfg(name="os", nll_bins=4):
    """A survival TaskConfig shaped exactly as ``build_registries`` emits one."""
    return TaskConfig(
        name=name,
        label_col=None,
        label_dict=None,
        task_type="survival",
        event_col="status",
        time_col="time",
        survival_losses=["nllsurv"],
        nll_bins=nll_bins,
    )


def _survival_exp_cfg(task="os", encoder="conch_v15", n_folds=2, survival_loss="nllsurv"):
    return ExperimentConfig(
        task=_survival_task_cfg(task),
        encoder_key=encoder,
        embed_dim=EMB,
        model=ModelConfig(model_type="dtfd_mil"),
        train=TrainConfig(seed=42),
        n_folds=n_folds,
        framework=Framework.DTFD,
        strategy="standard",
        survival_loss=survival_loss,
    )


def _build_survival_fixture(
    root, task="os", strategy="standard", encoder="conch_v15", n_folds=2,
    n_slides=12, survival_loss="nllsurv", suffix_plan_leaf=True,
):
    """On-disk artifacts ``run_dtfd_experiment`` reads for a SURVIVAL cell.

    Mirrors ``_build_benchmark_fixture`` but writes the survival task CSV
    (``status``/``time``/``case_id``, per ``load_dtfd_survival_split``) and puts
    the nnMIL plan under the ``_{survival_loss}``-suffixed leaf that
    ``_resolve_h5_dir`` builds for survival. ``suffix_plan_leaf=False`` writes it
    at the classification leaf instead, so a test can prove the suffix is live.
    """
    import pandas as pd

    rng = np.random.default_rng(23)
    h5_dir = os.path.join(root, "features", f"features_{encoder}")
    os.makedirs(h5_dir, exist_ok=True)

    slide_ids = [f"s{i}" for i in range(n_slides)]
    # Status alternates so every split carries both events and censored cases;
    # the bag mean tracks status, giving concordance a signal to find.
    statuses = {sid: (i % 2) for i, sid in enumerate(slide_ids)}
    for sid in slide_ids:
        _write_h5_bag(os.path.join(h5_dir, f"{sid}.h5"), rng, statuses[sid])

    csv_dir = os.path.join(root, "dataset_csv")
    os.makedirs(csv_dir, exist_ok=True)
    pd.DataFrame({
        "slide_id": slide_ids,
        "case_id": slide_ids,
        "status": [statuses[sid] for sid in slide_ids],
        "time": [10.0 + 5.0 * (i % 4) for i in range(n_slides)],
    }).to_csv(os.path.join(csv_dir, f"{task}.csv"), index=False)

    splits_dir = os.path.join(root, "splits", strategy, task)
    os.makedirs(splits_dir, exist_ok=True)
    for fold in range(n_folds):
        test_ids = [slide_ids[fold * 2], slide_ids[fold * 2 + 1]]
        val_ids = [slide_ids[(fold * 2 + 2) % n_slides], slide_ids[(fold * 2 + 3) % n_slides]]
        train_ids = [s for s in slide_ids if s not in test_ids and s not in val_ids]
        pd.DataFrame({
            "train": train_ids,
            "val": val_ids + [None] * (len(train_ids) - len(val_ids)),
            "test": test_ids + [None] * (len(train_ids) - len(test_ids)),
        }).to_csv(os.path.join(splits_dir, f"splits_{fold}.csv"), index=False)

    leaf = f"{task}_{encoder}_{survival_loss}" if suffix_plan_leaf else f"{task}_{encoder}"
    plan_dir = os.path.join(root, "nnmil", strategy, leaf)
    os.makedirs(plan_dir, exist_ok=True)
    with open(os.path.join(plan_dir, "dataset_plan.json"), "w") as f:
        json.dump({"feature_dir": h5_dir}, f)

    return h5_dir


class TestRunnerSurvival:
    def test_run_dtfd_experiment_survival_end_to_end(self, tmp_path):
        """The whole survival cell runs: runner -> trainer -> summary on disk.

        This is the test that would have caught the shipped defect. The runner
        passed ``policy_runtime=`` to ``train_dtfd_survival_fold``, which never
        declared it, so every ``os`` cell raised ``TypeError`` before its first
        epoch — and dropping the keyword would only have traded that for an
        ``UnboundLocalError`` inside the trainer. Both failures happen here, at
        the real call site, not in a signature comparison.
        """
        _build_survival_fixture(str(tmp_path), n_folds=2)
        exp = _survival_exp_cfg(n_folds=2)

        summary = run_dtfd_experiment(exp, str(tmp_path), device="cpu", cfg=_smoke_cfg())

        assert summary["framework"] == "dtfd"
        assert summary["survival_loss"] == "nllsurv"
        assert summary["n_folds"] == 2
        assert len(summary["per_fold_test"]) == 2

        # Survival reports concordance, never the classification key set.
        for block in (*summary["per_fold_test"], *summary["per_fold_val"]):
            assert set(block) == {"c_index"}
        assert "c_index" in summary["test"]
        assert "c_index" in summary["val"]

        # CR-3: val_pooled is populated for survival (it is {} for
        # classification), which only happens if val_records survived the
        # trainer -> runner hand-off intact.
        assert "c_index" in summary["val_pooled"], (
            "val_pooled empty: the runner did not receive per-fold val_records"
        )

        summary_path = os.path.join(
            str(tmp_path), "results", exp.results_subdir, "summary.json"
        )
        assert os.path.exists(summary_path)
        assert json.loads(open(summary_path).read())["experiment_id"] == exp.experiment_id

        for fold in range(2):
            fm = json.loads(open(os.path.join(
                str(tmp_path), "results", exp.results_subdir,
                f"fold_{fold}", "metrics.json",
            )).read())
            assert set(fm["test_metrics"]) == {"c_index"}
            assert len(fm["val_records"]["risks"]) == 2  # two val slides per fold

    def test_survival_runner_policy_runtime_reaches_both_tiers(self, tmp_path, monkeypatch):
        """The runner's PolicyRuntime is used by the trainer, not just accepted.

        Removing ``policy_runtime=`` from the call site would make the runner
        run green again while every policy silently stopped applying — the
        "wired but inert" failure this arm already shipped once. Spy on the real
        ``PolicyRuntime`` (delegating to the original, so the trainer genuinely
        runs) and require both DTFD tiers to be wrapped by the object the runner
        constructed.
        """
        # Identity matters, not just that *a* runtime was used: the trainer
        # falls back to ``policy_runtime or PolicyRuntime()``, so a runner that
        # stopped passing the argument would still wrap both tiers — with its
        # own default object, every configured policy silently gone. Track the
        # runtimes the RUNNER produced and require the trainer to have used one.
        from_runner: list[int] = []
        seen: list[tuple[str, int]] = []
        original_from_experiment = PolicyRuntime.from_experiment.__func__
        original_for_fold = PolicyRuntime.for_fold
        original_wrap = PolicyRuntime.wrap_optimizer

        def _spy_from_experiment(cls, *args, **kwargs):
            runtime = original_from_experiment(cls, *args, **kwargs)
            from_runner.append(id(runtime))
            return runtime

        def _spy_for_fold(self):
            runtime = original_for_fold(self)
            if id(self) in from_runner:
                from_runner.append(id(runtime))
            return runtime

        def _spy_wrap(self, optimizer, *, role="main"):
            seen.append((role, id(self)))
            return original_wrap(self, optimizer, role=role)

        monkeypatch.setattr(
            PolicyRuntime, "from_experiment", classmethod(_spy_from_experiment)
        )
        monkeypatch.setattr(PolicyRuntime, "for_fold", _spy_for_fold)
        monkeypatch.setattr(PolicyRuntime, "wrap_optimizer", _spy_wrap)

        _build_survival_fixture(str(tmp_path), n_folds=1)
        exp = _survival_exp_cfg(n_folds=1)

        summary = run_dtfd_experiment(exp, str(tmp_path), device="cpu", cfg=_smoke_cfg())

        assert len(summary["per_fold_val"]) == 1
        assert {role for role, _ in seen} == {"tier1", "tier2"}, (
            "the survival trainer did not wrap both DTFD tiers; roles were "
            f"{[role for role, _ in seen]}"
        )
        orphans = [role for role, runtime_id in seen if runtime_id not in from_runner]
        assert not orphans, (
            "the survival trainer wrapped "
            f"{orphans} with a PolicyRuntime the runner never handed it — the "
            "runner is no longer threading policy_runtime, so every configured "
            "policy is silently inert on this path"
        )

    def test_survival_runner_resolves_loss_suffixed_plan(self, tmp_path):
        """Survival must read the ``_{survival_loss}`` plan leaf, not the bare one.

        nnMIL writes a separate plan per survival loss (bin counts differ), so
        silently falling back to the classification leaf would train on the
        wrong bags. Only the unsuffixed plan exists here.
        """
        _build_survival_fixture(str(tmp_path), n_folds=1, suffix_plan_leaf=False)
        exp = _survival_exp_cfg(n_folds=1)

        with pytest.raises(FileNotFoundError, match="plan not found"):
            run_dtfd_experiment(exp, str(tmp_path), device="cpu", cfg=_smoke_cfg())

    def test_survival_runner_rejects_cox(self, tmp_path):
        """cox has no within-slide risk set; the runner must refuse it up front."""
        _build_survival_fixture(str(tmp_path), n_folds=1, survival_loss="cox")
        exp = _survival_exp_cfg(n_folds=1, survival_loss="cox")

        with pytest.raises(ValueError, match="does not support cox"):
            run_dtfd_experiment(exp, str(tmp_path), device="cpu", cfg=_smoke_cfg())
