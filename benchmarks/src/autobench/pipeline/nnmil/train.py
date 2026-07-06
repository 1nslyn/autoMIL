"""Single-fold training wrapper around nnMIL's trainers.

Selects between classification and survival trainers based on the plan's
``task_type`` (and, for survival, the ``survival_loss``).
"""

from __future__ import annotations

import json
import os
import time

from autobench.pipeline.config import ExperimentConfig, get_nnmil_runtime_overrides
from autobench.pipeline.nnmil.evaluate import normalize_nnmil_metrics


def select_nnmil_trainer(task_type: str, survival_loss: str | None) -> str:
    """Return the trainer kind for an nnMIL experiment.

    One of ``"classification"``, ``"survival"`` (cox/mse/mae), or
    ``"survival_porpoise"`` (nllsurv). Pure and import-free so it is
    unit-testable without nnMIL or a GPU. Note the divergence from nnMIL's
    reference, which picks the porpoise trainer by ``batch_size==1``; here
    selection is strictly by ``survival_loss=="nllsurv"``.
    """
    if task_type != "survival":
        return "classification"
    if survival_loss == "nllsurv":
        return "survival_porpoise"
    return "survival"


def train_nnmil_fold(
    exp_cfg: ExperimentConfig,
    plan_path: str,
    fold: int,
    results_dir: str,
    device: str = "cuda:0",
) -> dict:
    """Train one fold of an nnMIL experiment.

    Instantiates ``ClassificationTrainer`` with the plan file, runs
    ``train()`` and ``evaluate('test')``, then normalizes metrics to the
    shared benchmark format.

    Returns a dict with ``test_metrics`` and ``val_metrics`` keys,
    compatible with the CLAM fold result format.
    """
    fold_dir = os.path.join(results_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    metrics_path = os.path.join(fold_dir, "metrics.json")

    # Resume: skip if already completed
    if os.path.exists(metrics_path):
        print(f"\n    [fold {fold}] Already completed, loading from disk")
        with open(metrics_path) as f:
            return json.load(f)

    # Inspect the plan to decide which trainer to drive.
    with open(plan_path) as f:
        plan = json.load(f)
    task_type = plan.get("task_type", "classification")
    survival_loss = plan.get("survival_loss")
    kind = select_nnmil_trainer(task_type, survival_loss)

    # Set CUDA device for nnMIL
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", device.replace("cuda:", ""))

    # Pull fixed model-policy overrides from benchmark config.
    # This keeps fairness choices centralized and explicit.
    extra_kwargs: dict = get_nnmil_runtime_overrides(exp_cfg.model.model_type)
    if extra_kwargs:
        print(
            f"[nnMIL policy] {exp_cfg.model.model_type}: "
            f"{', '.join(f'{k}={v}' for k, v in sorted(extra_kwargs.items()))}"
        )

    common = dict(
        plan_path=plan_path,
        model_type=exp_cfg.model.model_type,
        fold=fold,
        save_dir=fold_dir,
        seed=exp_cfg.train.seed + fold,
    )

    # Deferred imports — nnMIL must be imported after CUDA_VISIBLE_DEVICES is set
    if kind == "survival_porpoise":
        from autobench.pipeline.nnmil._imports import SurvivalPorpoiseTrainer
        trainer = SurvivalPorpoiseTrainer(
            **common,
            survival_loss="nllsurv",
            nll_bins=plan.get("nll_bins", 4),
            **extra_kwargs,
        )
    elif kind == "survival":
        from autobench.pipeline.nnmil._imports import SurvivalTrainer
        trainer = SurvivalTrainer(
            **common,
            survival_loss=survival_loss,
            **extra_kwargs,
        )
    else:
        from autobench.pipeline.nnmil._imports import ClassificationTrainer
        trainer = ClassificationTrainer(**common, **extra_kwargs)

    _timer_start = time.perf_counter()

    trainer.create_model()
    trainer.create_data_loaders()
    trainer.train()

    # Evaluate test split
    test_raw = trainer.evaluate("test")
    test_metrics = normalize_nnmil_metrics(test_raw, split="test", task_type=task_type)

    # Evaluate val split
    val_raw = trainer.evaluate("val")
    val_metrics = normalize_nnmil_metrics(val_raw, split="val", task_type=task_type)

    elapsed_seconds = time.perf_counter() - _timer_start

    fold_result = {
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "fold": fold,
        "elapsed_seconds": elapsed_seconds,
    }

    with open(metrics_path, "w") as f:
        json.dump(fold_result, f, indent=2)

    # Clean up trainer to free GPU memory
    del trainer

    return fold_result
