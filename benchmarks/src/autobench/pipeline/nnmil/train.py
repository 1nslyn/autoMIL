"""Single-fold training wrapper around nnMIL's ClassificationTrainer."""

from __future__ import annotations

import json
import os
import time

from autobench.pipeline.config import ExperimentConfig, get_nnmil_runtime_overrides
from autobench.pipeline.nnmil.evaluate import normalize_nnmil_metrics


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

    # Deferred import — nnMIL must be imported after CUDA_VISIBLE_DEVICES is set
    from autobench.pipeline.nnmil._imports import ClassificationTrainer

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

    trainer = ClassificationTrainer(
        plan_path=plan_path,
        model_type=exp_cfg.model.model_type,
        fold=fold,
        save_dir=fold_dir,
        seed=exp_cfg.train.seed + fold,
        **extra_kwargs,
    )

    _timer_start = time.perf_counter()

    trainer.create_model()
    trainer.create_data_loaders()
    trainer.train()

    # Evaluate test split
    test_raw = trainer.evaluate("test")
    test_metrics = normalize_nnmil_metrics(test_raw, split="test")

    # Evaluate val split
    val_raw = trainer.evaluate("val")
    val_metrics = normalize_nnmil_metrics(val_raw, split="val")

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
