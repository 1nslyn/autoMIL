#!/usr/bin/env python
"""Run a single benchmark experiment (one task × encoder × model).

Designed for use with autoMIL: runs one experiment and writes result.json
to the current working directory.

Examples
--------
# CLAM experiment
python benchmarks/scripts/run_experiment.py \
    --dataset ccrcc --task high_grade --encoder uni_v2 \
    --model clam_mb --framework clam

# nnMIL experiment
python benchmarks/scripts/run_experiment.py \
    --dataset ccrcc --task pbrm1 --encoder uni_v2 \
    --model ab_mil --framework nnmil
"""

from __future__ import annotations

import argparse
import json
import os
import time

from dotenv import load_dotenv
import torch

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# autoMIL overlay fix (ISSUE-010 / ISSUE-021): prepend THIS checkout's benchmarks/src so the
# co-located autobench (and its LIB_ROOT -> lib/CLAM) wins over the editable install. Without
# this, a worktree run imports autobench + CLAM from the MAIN repo and every overlay to
# benchmarks/src or benchmarks/lib is silently ignored (the orchestrator stopped injecting
# the worktree PYTHONPATH in D-199).
import sys as _sys
_sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from automil.runtime_helpers import register_sigterm_flush
from autobench.config import load_dataset_config
from autobench.pipeline.config import (
    ExperimentConfig,
    Framework,
    ModelConfig,
    TrainConfig,
    build_registries,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a single benchmark experiment")
    p.add_argument("--dataset", required=True, help="Dataset config name or YAML path")
    p.add_argument("--task", required=True, help="Task name (e.g., high_grade, pbrm1)")
    p.add_argument("--encoder", required=True, help="Encoder key (e.g., uni_v2)")
    p.add_argument("--model", required=True, help="Model type (e.g., clam_mb, ab_mil)")
    p.add_argument("--framework", required=True, choices=["clam", "nnmil"])
    p.add_argument("--strategy", default="standard", help="Split strategy")
    p.add_argument("--gpu", type=int, default=None,
                   help="GPU index (default: AUTOMIL_GPU or 0)")

    # Training overrides
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--stop_epoch", type=int, default=50)
    p.add_argument("--no_wandb", action="store_true")

    return p.parse_args()


def summary_to_result_json(summary: dict, elapsed: float) -> dict:
    """Convert autobench summary dict to autoMIL result.json format."""
    test = summary.get("test", {})
    val = summary.get("val", {})

    test_auc = test.get("auc_roc", {}).get("mean", 0.0)
    test_bacc = test.get("balanced_accuracy", {}).get("mean", 0.0)
    val_auc = val.get("auc_roc", {}).get("mean", 0.0)
    val_bacc = val.get("balanced_accuracy", {}).get("mean", 0.0)

    composite = (test_auc + test_bacc) / 2

    # Try to get peak VRAM
    peak_vram_mb = 0
    try:
        if torch.cuda.is_available():
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass

    return {
        "status": "completed",
        "metrics": {
            "val_auc": round(val_auc, 4),
            "val_bacc": round(val_bacc, 4),
            "test_auc": round(test_auc, 4),
            "test_bacc": round(test_bacc, 4),
        },
        "composite": round(composite, 4),
        "elapsed_seconds": round(elapsed, 1),
        "peak_vram_mb": round(peak_vram_mb),
        "summary": summary,
    }


def main() -> None:
    # CAP-03 / D-121: install SIGTERM handler BEFORE any DataLoader/multiprocessing setup.
    # The handler aggregates archive/<node>/fold_*_result.json into a partial result.json
    # on cap-driven cancel, then sys.exit(0) — letting the orchestrator reconcile to
    # status='executed' with metadata.budget_killed=True.
    register_sigterm_flush()

    args = parse_args()
    start_time = time.time()

    # Overlay activation diagnostic: prove which autobench + CLAM are actually
    # loaded. If these paths don't match the current worktree, the overlay is
    # being shadowed by the main-repo editable install (fix: set AUTOBENCH_ROOT
    # and prepend the worktree src to PYTHONPATH in the orchestrator).
    import autobench
    print(f"[automil] autobench.__file__ = {autobench.__file__}")
    print(f"[automil] autobench.LIB_ROOT = {autobench.LIB_ROOT}")
    print(f"[automil] cwd                = {os.getcwd()}")
    print(f"[automil] AUTOBENCH_ROOT env = {os.environ.get('AUTOBENCH_ROOT', '<unset>')}")
    print(f"[automil] AUTOMIL_NODE_ID    = {os.environ.get('AUTOMIL_NODE_ID', '<unset>')}")
    print(f"[automil] AUTOMIL_RESULTS_DIR= {os.environ.get('AUTOMIL_RESULTS_DIR', '<unset>')}")

    # Determine GPU
    gpu = args.gpu
    if gpu is None:
        gpu = int(os.environ.get("AUTOMIL_GPU", os.environ.get("CUDA_VISIBLE_DEVICES", "0")))

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    # Load dataset config
    ds = load_dataset_config(args.dataset)
    registries = build_registries(ds)

    print(f"Running single experiment: {args.framework}/{args.task}/{args.encoder}/{args.model}")
    print(f"  Dataset: {ds.name} — {ds.description}")
    print(f"  Device: {device}")

    # Build experiment config
    task_cfg = registries.task_registry[args.task]
    embed_dim = registries.encoder_dims[args.encoder]
    framework = Framework.CLAM if args.framework == "clam" else Framework.NNMIL

    model_cfg = registries.model_registry.get(
        args.model, ModelConfig(model_type=args.model)
    )

    train_cfg = TrainConfig(
        max_epochs=args.max_epochs,
        lr=args.lr,
        seed=args.seed,
        patience=args.patience,
        stop_epoch=args.stop_epoch,
    )

    exp_cfg = ExperimentConfig(
        task=task_cfg,
        encoder_key=args.encoder,
        embed_dim=embed_dim,
        model=model_cfg,
        train=train_cfg,
        n_folds=args.n_folds,
        framework=framework,
        strategy=args.strategy,
    )

    benchmark_dir = ds.benchmark_dir

    # When running under autoMIL, write per-fold checkpoints/metrics into
    # this experiment's archive dir (set by the orchestrator) so that:
    #   1. Each experiment is isolated (no cross-experiment cache hits)
    #   2. Results are co-located with run.log/spec.json/result.json for
    #      easy inspection in automil/orchestrator/archive/<node_id>/results/
    # Data preparation (splits, CSVs) still uses the shared benchmark_dir.
    automil_results_dir = os.environ.get("AUTOMIL_RESULTS_DIR")
    if automil_results_dir:
        automil_results_dir = os.path.join(automil_results_dir, "results")
        os.makedirs(automil_results_dir, exist_ok=True)

    # Ensure data is prepared
    from autobench.pipeline.prepare import prepare_all
    prepare_all(
        benchmark_dir=benchmark_dir,
        mapping_csv=ds.mapping_csv,
        features_base_dir=ds.features_base_dir,
        encoder_keys=[args.encoder],
        ds=ds,
        seed=train_cfg.seed,
        n_splits=args.n_folds,
    )

    # Run the experiment
    if framework == Framework.CLAM:
        from autobench.pipeline.clam.runner import run_experiment
        summary = run_experiment(
            exp_cfg, benchmark_dir, device,
            wandb_project=None if args.no_wandb else f"{ds.name}-automil",
            results_dir=automil_results_dir,
        )
    else:
        from autobench.pipeline.nnmil.runner import run_nnmil_experiment
        from autobench.pipeline.nnmil.prepare import prepare_nnmil_experiment
        prepare_nnmil_experiment(
            benchmark_dir=benchmark_dir,
            task_name=args.task,
            encoder_key=args.encoder,
            strategy=args.strategy,
            label_col=task_cfg.label_col,
            label_dict=task_cfg.label_dict,
            embed_dim=embed_dim,
            features_base_dir=ds.features_base_dir,
            dataset_name=ds.name,
            seed=train_cfg.seed,
            n_splits=args.n_folds,
        )
        summary = run_nnmil_experiment(
            exp_cfg, benchmark_dir, device=str(device),
        )

    elapsed = time.time() - start_time

    # Write result.json (autoMIL contract)
    result = summary_to_result_json(summary, elapsed)
    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nExperiment complete in {elapsed:.0f}s")
    print(f"  test_auc={result['metrics']['test_auc']:.4f}  "
          f"test_bacc={result['metrics']['test_bacc']:.4f}  "
          f"composite={result['composite']:.4f}")
    print(f"  result.json written to {os.path.abspath('result.json')}")


if __name__ == "__main__":
    main()
