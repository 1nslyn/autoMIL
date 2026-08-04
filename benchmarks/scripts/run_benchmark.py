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
    "abmil": Framework.ABMIL,
}

# L-1: source name shown in the empty-roster error, per framework. TITAN is
# deliberately absent -- it has no model-type axis (generate_all_experiments
# pins the "titan" pseudo-model unconditionally), so an empty roster is not
# reachable for it.
_ROSTER_SOURCE_NAMES: dict[Framework, str] = {
    Framework.CLAM: "--models / clam_models",
    Framework.DTFD: "--dtfd_models / dtfd_models",
    Framework.ABMIL: "--abmil_models / abmil_models",
    Framework.NNMIL: "--nnmil_models / nnmil_models",
}


def _empty_rosters(
    frameworks: list[Framework],
    *,
    models: list[str],
    dtfd_models: list[str],
    abmil_models: list[str],
    nnmil_models: list[str],
) -> list[tuple[Framework, str]]:
    """Requested frameworks whose resolved model roster is empty (L-1).

    ``generate_all_experiments``'s ``for model_type in model_types:`` loop
    simply doesn't iterate when a framework's roster is empty -- e.g. a
    dataset YAML that never set ``nnmil_models`` (defaults to ``[]``). The
    run then "succeeds" having launched zero experiments for that framework,
    indistinguishable from "there was nothing to do". Returns the offending
    (framework, source-name) pairs so the caller can fail loudly instead.
    """
    rosters = {
        Framework.CLAM: models,
        Framework.DTFD: dtfd_models,
        Framework.ABMIL: abmil_models,
        Framework.NNMIL: nnmil_models,
    }
    return [
        (fw, _ROSTER_SOURCE_NAMES[fw])
        for fw in frameworks
        if fw in rosters and not rosters[fw]
    ]


def _build_parser() -> argparse.ArgumentParser:
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
    p.add_argument("--abmil_models", nargs="+", default=None,
                   help="ABMIL model types (default: all from dataset config)")

    # Training. CFG-3 (audit 2026-07-28): default=None, NOT a CLI literal.
    # These used to duplicate every TrainConfig default here and pass them in
    # unconditionally, which made the dataclass defaults DEAD on the static-grid
    # path -- the grid's hyperparameters came from this file, so correcting
    # TrainConfig.lr to CLAM's upstream 1e-4 would have been silently reset to
    # 2e-4 on every launch. Same CFG-01 / D-01 pattern run_experiment.py uses.
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n_folds", type=int, default=None)
    p.add_argument("--no_early_stopping", action="store_true")
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--stop_epoch", type=int, default=None)
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

    return p


def parse_args() -> argparse.Namespace:
    return _build_parser().parse_args()


def _train_config_from_args(args: argparse.Namespace) -> TrainConfig:
    """Build a TrainConfig from only the flags that were explicitly supplied.

    CFG-3: unset flags parse as ``None`` and are dropped, so each arm's dataclass
    default stays the single source of truth. The two ``store_true`` switches are
    inverted only when actually passed, for the same reason.
    """
    explicit = {
        k: v for k, v in {
            "max_epochs": args.max_epochs,
            "lr": args.lr,
            "seed": args.seed,
            "patience": args.patience,
            "stop_epoch": args.stop_epoch,
        }.items() if v is not None
    }
    if args.no_early_stopping:
        explicit["early_stopping"] = False
    if args.no_weighted_sample:
        explicit["weighted_sample"] = False
    return TrainConfig(**explicit)


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
    abmil_models = args.abmil_models or ds.abmil_models
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

    # L-1: validate every requested framework's model roster is non-empty
    # BEFORE any experiment generation -- otherwise the grid generator's
    # per-framework loop silently contributes zero experiments and the run
    # exits 0 having done nothing.
    empty = _empty_rosters(
        frameworks,
        models=models, dtfd_models=dtfd_models,
        abmil_models=abmil_models, nnmil_models=nnmil_models,
    )
    if empty:
        for fw, source in empty:
            print(
                f"Error: framework '{fw.value}' has an empty model roster "
                f"({source} resolved to []). Configure it in the dataset "
                "YAML or pass it explicitly on the CLI."
            )
        sys.exit(1)

    train_cfg = _train_config_from_args(args)

    wandb_project = args.wandb_project or f"{ds.name}-benchmark"

    cfg = BenchmarkConfig(
        benchmark_dir=args.benchmark_dir or ds.benchmark_dir,
        mapping_csv=args.mapping_csv or ds.mapping_csv,
        features_base_dir=args.features_base_dir or ds.features_base_dir,
        encoder_keys=encoders,
        model_types=models,
        tasks=tasks,
        train=train_cfg,
        # CFG-3: None means "unset" -> BenchmarkConfig's own default (5) applies.
        n_folds=args.n_folds if args.n_folds is not None else 5,
        gpu=args.gpu,
        wandb_project=None if args.no_wandb else wandb_project,
        experiments_per_gpu=args.experiments_per_gpu,
        strategies=strategies,
        frameworks=frameworks,
        nnmil_model_types=nnmil_models,
        dtfd_model_types=dtfd_models,
        abmil_model_types=abmil_models,
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
