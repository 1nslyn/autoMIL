"""Protocol v5: every runner hands each fold's policy runtime that fold's seed and index.

A policy's bag transform draws from a generator private to its fold, seeded from
``(seed, fold)``; a runtime built without them refuses the first transform. The
seed is whatever the arm already seeds that fold's training with: ``seed + fold``
for ABMIL, DTFD, TITAN and nnMIL, the bare ``seed`` for CLAM (whose folds are
told apart by the fold index alone). Each test runs a real runner and records
what its fold trainer is handed; the heavy arms (nnMIL, CLAM survival) run
against a stub trainer, since only the hand-off is under test.
"""
from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")

from _helpers import make_test_ds  # noqa: E402
from autobench.pipeline.abmil import runner as abmil_runner  # noqa: E402
from autobench.pipeline.clam import runner as clam_runner  # noqa: E402
from autobench.pipeline.config import (  # noqa: E402
    ExperimentConfig,
    Framework,
    ModelConfig,
    TaskConfig,
    TrainConfig,
    build_registries,
)
from autobench.pipeline.dtfd import runner as dtfd_runner  # noqa: E402
from autobench.pipeline.nnmil import runner as nnmil_runner  # noqa: E402
from autobench.pipeline.nnmil.prepare import nnmil_plan_dir  # noqa: E402
from autobench.pipeline.titan import runner as titan_runner  # noqa: E402
from autobench.pipeline.titan.prepare import prepare_titan_experiment  # noqa: E402
from tests import test_abmil_arm, test_dtfd_arm  # noqa: E402
from tests.test_checkpoint_selection import _clam_fixture  # noqa: E402

# The TITAN runner's fixture chain: a prepared manifest, features and 2-fold splits.
from tests.test_titan_arm import (  # noqa: E402, F401
    benchmark_dir,
    ds,
    registries,
    splits_2fold,
    task_csv,
    titan_exp_cfg,
    titan_features_dir,
)


def _hand_offs(monkeypatch, module, name, *, stub=None):
    """Record the ``(seed, fold)`` of the runtime each call of ``module.name`` is handed.

    The real fold trainer still runs, unless ``stub`` stands in for it.
    """
    handed: list[tuple[int | None, int | None]] = []
    real = stub or getattr(module, name)

    def spy(*args, **kwargs):
        runtime = kwargs["policy_runtime"]
        handed.append((runtime.seed, runtime.fold))
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, spy)
    return handed


def _canned_fold(*, metric: str, fold: int) -> dict:
    scores = {metric: 0.6}
    return {
        "test_metrics": dict(scores), "val_metrics": dict(scores),
        "val_predictions_sha256": None, "fold": fold, "elapsed_seconds": 0.0,
    }


def test_the_abmil_runner_seeds_each_fold_from_its_training_seed(tmp_path, monkeypatch):
    handed = _hand_offs(monkeypatch, abmil_runner, "train_abmil_fold")
    test_abmil_arm._build_benchmark_fixture(str(tmp_path), n_folds=2)
    exp = test_abmil_arm._exp_cfg(build_registries(make_test_ds()), n_folds=2)
    abmil_runner.run_abmil_experiment(
        exp, str(tmp_path), device="cpu", cfg=test_abmil_arm._smoke_cfg(),
    )
    assert handed == [(exp.train.seed, 0), (exp.train.seed + 1, 1)]


def test_the_dtfd_runner_seeds_each_fold_from_its_training_seed(tmp_path, monkeypatch):
    handed = _hand_offs(monkeypatch, dtfd_runner, "train_dtfd_fold")
    test_dtfd_arm._build_benchmark_fixture(str(tmp_path), n_folds=2)
    exp = test_dtfd_arm._exp_cfg(build_registries(make_test_ds()), n_folds=2)
    dtfd_runner.run_dtfd_experiment(
        exp, str(tmp_path), device="cpu", cfg=test_dtfd_arm._smoke_cfg(),
    )
    assert handed == [(exp.train.seed, 0), (exp.train.seed + 1, 1)]


def test_the_titan_runner_seeds_each_fold_from_its_training_seed(
    benchmark_dir, task_csv, titan_features_dir, splits_2fold, titan_exp_cfg, monkeypatch,
):
    handed = _hand_offs(monkeypatch, titan_runner, "train_titan_fold")
    prepare_titan_experiment(
        benchmark_dir=benchmark_dir, task_name="brca", features_base_dir=benchmark_dir,
    )
    titan_runner.run_titan_experiment(titan_exp_cfg, benchmark_dir, device="cpu")
    assert handed == [(titan_exp_cfg.train.seed, 0), (titan_exp_cfg.train.seed + 1, 1)]


def test_the_nnmil_runner_seeds_each_fold_from_its_training_seed(tmp_path, monkeypatch):
    handed = _hand_offs(
        monkeypatch, nnmil_runner, "train_nnmil_fold",
        stub=lambda exp_cfg, plan_path, fold, results_dir, **kwargs: _canned_fold(
            metric="auc_roc", fold=fold,
        ),
    )
    exp = ExperimentConfig(
        task=TaskConfig(name="brca", label_col="label", label_dict={"neg": 0, "pos": 1}),
        encoder_key="conch_v15", embed_dim=8, model=ModelConfig(model_type="simple_mil"),
        train=TrainConfig(seed=42), n_folds=2, framework=Framework.NNMIL, strategy="standard",
    )
    plan_dir = nnmil_plan_dir(str(tmp_path), "standard", "brca", "conch_v15")
    os.makedirs(plan_dir)
    with open(os.path.join(plan_dir, "dataset_plan.json"), "w") as handle:
        json.dump({"training_configuration": {}}, handle)

    nnmil_runner.run_nnmil_experiment(exp, str(tmp_path), device="cpu")
    assert handed == [(42, 0), (43, 1)]


def test_the_clam_runner_seeds_each_fold_from_the_one_training_seed(tmp_path, monkeypatch):
    handed = _hand_offs(monkeypatch, clam_runner, "train_fold")
    exp_cfg, _ = _clam_fixture(tmp_path, max_epochs=1)
    benchmark = tmp_path / "benchmark"
    splits = benchmark / "splits" / "standard" / "brca"
    (splits / "splits_1.csv").write_text((splits / "splits_0.csv").read_text())
    exp_cfg.n_folds = 2

    clam_runner.run_experiment(exp_cfg, str(benchmark), torch.device("cpu"))
    assert handed == [(exp_cfg.train.seed, 0), (exp_cfg.train.seed, 1)]


def test_the_clam_survival_branch_hands_off_the_same_way(tmp_path, monkeypatch):
    from autobench.pipeline.clam import survival_train

    handed = _hand_offs(
        monkeypatch, survival_train, "train_survival_fold",
        stub=lambda exp_cfg, benchmark_dir, fold, results_dir, device, **kwargs: _canned_fold(
            metric="c_index", fold=fold,
        ),
    )
    exp = ExperimentConfig(
        task=TaskConfig(
            name="os", label_col="status", label_dict={}, n_classes=2, task_type="survival",
        ),
        encoder_key="conch_v15", embed_dim=8, model=ModelConfig(model_type="clam_sb"),
        train=TrainConfig(seed=7), n_folds=2, framework=Framework.CLAM, strategy="standard",
        survival_loss="cox",
    )
    clam_runner.run_experiment(exp, str(tmp_path), torch.device("cpu"))
    assert handed == [(7, 0), (7, 1)]
