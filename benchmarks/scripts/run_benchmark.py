#!/usr/bin/env python
"""CLI for running the WSI classification benchmark.

Examples
--------
# Full benchmark with ovarian dataset
uv run python benchmarks/scripts/run_benchmark.py --dataset ovarian --gpu 0

# Multi-GPU (auto-detect)
uv run python benchmarks/scripts/run_benchmark.py --dataset ovarian --all_gpus

# CLWD dataset
uv run python benchmarks/scripts/run_benchmark.py --dataset clwd --gpu 0

# Subset
uv run python benchmarks/scripts/run_benchmark.py --dataset ovarian --encoders conch_v15 --models clam_sb --tasks brca

# Data prep only
uv run python benchmarks/scripts/run_benchmark.py --dataset ovarian --prep_only

# nnMIL with specific strategies
uv run python benchmarks/scripts/run_benchmark.py --dataset ovarian --frameworks nnmil --strategies standard --all_gpus
"""

from __future__ import annotations

# Pin cuBLAS workspace BEFORE torch imports so CUDA contexts pick it up at
# creation time. nnMIL's set_random_seeds sets this same value mid-process,
# but cuBLAS only reads CUBLAS_WORKSPACE_CONFIG at context init — so a CLAM
# experiment that runs before any nnMIL one in the same process otherwise
# gets a different workspace and picks different (non-deterministic) kernels.
# Setting it here makes every experiment in this process see the same value
# from epoch 0, regardless of framework execution order.
import os as _os
_os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import os
import sys

from dotenv import load_dotenv
import torch

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from autobench.config import load_dataset_config
from autobench.pipeline.config import (
    BenchmarkConfig,
    Framework,
    TrainConfig,
    build_registries,
    generate_all_experiments,
)


_FRAMEWORK_MAP: dict[str, Framework] = {
    "clam": Framework.CLAM,
    "nnmil": Framework.NNMIL,
    "dtfd": Framework.DTFD,
    "titan": Framework.TITAN,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WSI Classification Benchmark")

    # Dataset (required)
    p.add_argument("--dataset", type=str, required=True,
                   help="Dataset config name (e.g., 'ovarian', 'clwd') or path to YAML")

    # GPU selection (mutually exclusive)
    gpu_group = p.add_mutually_exclusive_group()
    gpu_group.add_argument("--gpu", type=int, default=0, help="Single GPU index (default: 0)")
    gpu_group.add_argument("--all_gpus", action="store_true", help="Use all available GPUs")
    gpu_group.add_argument("--gpus", type=int, nargs="+", help="Specific GPU indices")

    # Path overrides (override dataset YAML defaults)
    p.add_argument("--benchmark_dir", type=str, default=None)
    p.add_argument("--mapping_csv", type=str, default=None)
    p.add_argument("--features_base_dir", type=str, default=None)

    # Experiment subset (defaults loaded from dataset config)
    p.add_argument("--encoders", nargs="+", default=None)
    p.add_argument("--models", nargs="+", default=None, help="CLAM model types")
    p.add_argument("--tasks", nargs="+", default=None)
    p.add_argument("--strategies", nargs="+", default=None, help="Split strategies")
    p.add_argument("--frameworks", nargs="+", default=["clam"],
                   choices=list(_FRAMEWORK_MAP.keys()),
                   help="Model frameworks (default: clam)")
    p.add_argument("--nnmil_models", nargs="+", default=None,
                   help="nnMIL model types (default: all from dataset config)")
    p.add_argument("--dtfd_models", nargs="+", default=None,
                   help="DTFD model types (default: all from dataset config)")

    # Training
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_folds", type=int, default=10)
    p.add_argument("--no_early_stopping", action="store_true")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--stop_epoch", type=int, default=50)
    p.add_argument("--no_weighted_sample", action="store_true")

    # Logging
    p.add_argument("--wandb_project", type=str, default=None,
                   help="Wandb project name (default: {dataset}-benchmark)")
    p.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")

    # Concurrency
    p.add_argument("--experiments_per_gpu", type=int, default=None,
                   help="Concurrent experiments per GPU (default: auto-detect)")

    # Modes
    p.add_argument("--prep_only", action="store_true", help="Only run data preparation")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load dataset configuration
    ds = load_dataset_config(args.dataset)
    registries = build_registries(ds)
    print(f"Loaded dataset config: {ds.name} ({ds.description})")
    print(f"  Tasks: {list(ds.tasks.keys())}")
    print(f"  Strategies: {list(ds.split_strategies.keys())}")
    print(f"  Encoders: {list(ds.encoder_dims.keys())}")

    # Resolve defaults from dataset config
    encoders = args.encoders or list(ds.encoder_dims.keys())
    models = args.models or ds.clam_models
    tasks = args.tasks or list(ds.tasks.keys())
    strategies = args.strategies or [list(ds.split_strategies.keys())[0]]
    nnmil_models = args.nnmil_models or ds.nnmil_models
    dtfd_models = args.dtfd_models or ds.dtfd_models
    frameworks = [_FRAMEWORK_MAP[f] for f in args.frameworks]

    # Validate encoder keys
    for e in encoders:
        if e not in ds.encoder_dims:
            print(f"Error: unknown encoder '{e}'. Valid: {list(ds.encoder_dims.keys())}")
            sys.exit(1)

    # Validate strategies
    for s in strategies:
        if s not in ds.split_strategies:
            print(f"Error: unknown strategy '{s}'. Valid: {list(ds.split_strategies.keys())}")
            sys.exit(1)

    train_cfg = TrainConfig(
        max_epochs=args.max_epochs,
        lr=args.lr,
        seed=args.seed,
        early_stopping=not args.no_early_stopping,
        patience=args.patience,
        stop_epoch=args.stop_epoch,
        weighted_sample=not args.no_weighted_sample,
    )

    wandb_project = args.wandb_project or f"{ds.name}-benchmark"

    cfg = BenchmarkConfig(
        benchmark_dir=args.benchmark_dir or ds.benchmark_dir,
        mapping_csv=args.mapping_csv or ds.mapping_csv,
        features_base_dir=args.features_base_dir or ds.features_base_dir,
        encoder_keys=encoders,
        model_types=models,
        tasks=tasks,
        train=train_cfg,
        n_folds=args.n_folds,
        gpu=args.gpu,
        wandb_project=None if args.no_wandb else wandb_project,
        experiments_per_gpu=args.experiments_per_gpu,
        strategies=strategies,
        frameworks=frameworks,
        nnmil_model_types=nnmil_models,
        dtfd_model_types=dtfd_models,
    )

    if args.prep_only:
        from autobench.pipeline.prepare import prepare_all

        prepare_all(
            benchmark_dir=cfg.benchmark_dir,
            mapping_csv=cfg.mapping_csv,
            features_base_dir=cfg.features_base_dir,
            encoder_keys=cfg.encoder_keys,
            ds=ds,
            seed=cfg.train.seed,
            n_splits=cfg.n_folds,
        )
        print("Data preparation complete.")
        return

    # Determine GPU mode
    if args.all_gpus:
        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            print(f"Only {n_gpus} GPU(s) detected, falling back to single-GPU mode")
            from autobench.pipeline.orchestrator import run_benchmark
            run_benchmark(cfg, ds=ds, registries=registries)
        else:
            gpu_ids = list(range(n_gpus))
            print(f"Multi-GPU mode: {gpu_ids}")
            from autobench.pipeline.orchestrator import run_benchmark_multigpu
            run_benchmark_multigpu(cfg, gpu_ids, ds=ds, registries=registries)
    elif args.gpus:
        print(f"Multi-GPU mode: {args.gpus}")
        from autobench.pipeline.orchestrator import run_benchmark_multigpu
        run_benchmark_multigpu(cfg, args.gpus, ds=ds, registries=registries)
    else:
        from autobench.pipeline.orchestrator import run_benchmark
        run_benchmark(cfg, ds=ds, registries=registries)


if __name__ == "__main__":
    main()
