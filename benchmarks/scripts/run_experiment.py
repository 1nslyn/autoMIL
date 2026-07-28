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

# ABMIL experiment (reuses nnMIL's H5-bag prep)
python benchmarks/scripts/run_experiment.py \
    --dataset ccrcc --task pbrm1 --encoder uni_v2 \
    --model abmil --framework abmil

# DTFD-MIL experiment (reuses nnMIL's H5-bag prep)
python benchmarks/scripts/run_experiment.py \
    --dataset ccrcc --task pbrm1 --encoder uni_v2 \
    --model dtfd_mil --framework dtfd

# TITAN experiment (frozen slide embedding -- --encoder must be "titan")
python benchmarks/scripts/run_experiment.py \
    --dataset ccrcc --task pbrm1 --encoder titan \
    --model titan --framework titan
"""

from __future__ import annotations

import argparse
import json
import math
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
    p.add_argument("--framework", required=True, choices=["clam", "nnmil", "abmil", "dtfd", "titan"])
    p.add_argument("--strategy", default="standard", help="Split strategy")
    p.add_argument(
        "--survival_loss", default=None,
        choices=["cox", "mse", "mae", "nllsurv"],
        help="Survival loss variant (survival tasks only; defaults to the "
             "task's first configured survival_losses entry)",
    )
    p.add_argument("--gpu", type=int, default=None,
                   help="GPU index (default: AUTOMIL_GPU or 0)")

    # Training overrides — default=None so dataclass defaults are honored when not supplied (CFG-01 / D-01)
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n_folds", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--stop_epoch", type=int, default=None)
    # H-3b: these two had no flag at all, so they were reachable only through a
    # registered variant's CLAM_ARGS.
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--early_stopping", dest="early_stopping",
                   action="store_const", const=True, default=None)
    p.add_argument("--no_early_stopping", dest="early_stopping",
                   action="store_const", const=False)
    # H-3b: the opaque per-arm channel. The shared transport is CLAM-shaped, so
    # DTFD's numGroup, ABMIL's M/L and nnMIL's warmup_epochs have no flag to
    # travel in — and adding a flag per arm-specific knob would be both endless
    # and asymmetric. A JSON object keeps the channel arm-agnostic; each name is
    # checked against that arm's DECLARED search space (search_space.py), so an
    # undeclared knob fails loudly rather than being silently dropped.
    #
    # Note this is a *value*, not a bare flag: `--override "--numGroup 8"` would
    # reach argparse as an unrecognised flag and SystemExit(2) the run, which is
    # how this asymmetry stayed invisible.
    p.add_argument(
        "--hparams", type=str, default=None,
        help='JSON object of arm-specific hyperparameter overrides, '
             'e.g. \'{"numGroup": 8, "grad_clip": 1.0}\'',
    )
    p.add_argument("--no_wandb", action="store_true")

    return p.parse_args()


def _parse_hparams(raw: str | None) -> dict:
    """Parse --hparams, failing loudly on anything that is not a flat JSON object."""
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--hparams is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(
            f"--hparams must be a JSON object mapping knob -> value, got "
            f"{type(parsed).__name__}"
        )
    bad = [k for k, v in parsed.items() if isinstance(v, (dict, list))]
    if bad:
        raise SystemExit(
            f"--hparams values must be scalars; nested value(s) for {sorted(bad)}"
        )
    return parsed


def summary_to_result_json(summary: dict, elapsed: float) -> dict:
    """Convert autobench summary dict to autoMIL result.json format.

    The composite is the VALIDATION selection signal (autoMIL keep/discard and
    UCB select on it): survival summaries (``c_index`` entry) use the validation
    concordance index; classification uses ``(val_auc + val_bacc) / 2``. Test
    metrics stay in ``metrics`` for now (quarantined in a later step) and are
    never the selection signal.
    """
    test = summary.get("test", {})
    val = summary.get("val", {})

    # Try to get peak VRAM
    peak_vram_mb = 0
    try:
        if torch.cuda.is_available():
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass

    if "c_index" in test:
        test_ci = test.get("c_index", {}).get("mean", 0.0)
        # CR-3 (audit 2026-07-23): prefer the POOLED cross-fold val concordance as
        # the selection signal. The per-fold val c-index is computed on ~2 events —
        # the survival trainers themselves refuse to select checkpoints on it
        # ("near-random"), yet autoMIL's search was selecting recipes on the mean
        # of five such values. Pooling scores concordance once over every fold's
        # val risks (~5x the events); it stays a concordance, so it remains
        # comparable across recipes and across cox/nllsurv. Falls back to the
        # fold-mean when a runner has not yet exported val_records.
        pooled = (summary.get("val_pooled") or {}).get("c_index")
        fold_mean_ci = val.get("c_index", {}).get("mean", 0.0)
        val_ci = pooled if isinstance(pooled, float) and math.isfinite(pooled) else fold_mean_ci
        composite = val_ci
        metrics = {"val_c_index": round(val_ci, 4)}
        held_out = {"test_c_index": round(test_ci, 4)}
    else:
        test_auc = test.get("auc_roc", {}).get("mean", 0.0)
        test_bacc = test.get("balanced_accuracy", {}).get("mean", 0.0)
        val_auc = val.get("auc_roc", {}).get("mean", 0.0)
        val_bacc = val.get("balanced_accuracy", {}).get("mean", 0.0)
        composite = (val_auc + val_bacc) / 2
        metrics = {
            "val_auc": round(val_auc, 4),
            "val_bacc": round(val_bacc, 4),
        }
        held_out = {
            "test_auc": round(test_auc, 4),
            "test_bacc": round(test_bacc, 4),
        }

    # H-8 (audit 2026-07-23): count folds that produced a finite primary val
    # metric. compute_confidence_intervals silently drops NaN folds and, with <2
    # valid, reports a zero-variance point estimate — so a degenerate 1-fold run
    # would otherwise masquerade as a complete K-fold "completed" result and get
    # selected as if robust. Surface the count and quarantine it (status=partial,
    # which autoMIL keeps out of keep/discard) when the CV is too degenerate.
    primary_val_key = "c_index" if "c_index" in test else "auc_roc"
    per_fold_val = summary.get("per_fold_val", []) or []
    n_folds_total = summary.get("n_folds", len(per_fold_val))
    n_valid_folds = sum(
        1 for fm in per_fold_val
        if isinstance(fm, dict)
        and isinstance(fm.get(primary_val_key), (int, float))
        and math.isfinite(float(fm.get(primary_val_key, float("nan"))))
    )
    status = "completed" if n_valid_folds >= 2 else "partial"

    # ``metrics`` is agent-facing (val only); ``held_out`` (test) + ``summary``
    # are sealed into certify.json by terminal_writer — never seen during search.
    return {
        "status": status,
        "metrics": metrics,
        "held_out": held_out,
        "composite": round(composite, 4),
        "elapsed_seconds": round(elapsed, 1),
        "peak_vram_mb": round(peak_vram_mb),
        "n_valid_folds": n_valid_folds,
        "n_folds": n_folds_total,
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
    _FRAMEWORK_MAP = {
        "clam": Framework.CLAM,
        "nnmil": Framework.NNMIL,
        "abmil": Framework.ABMIL,
        "dtfd": Framework.DTFD,
        "titan": Framework.TITAN,
    }
    framework = _FRAMEWORK_MAP[args.framework]
    # TITAN has no tile-encoder axis: "titan" isn't in encoder_dims, so use a
    # 768 placeholder -- run_titan_experiment overwrites it from the manifest dim.
    embed_dim = 768 if framework == Framework.TITAN else registries.encoder_dims[args.encoder]

    model_cfg = registries.model_registry.get(
        args.model, ModelConfig(model_type=args.model)
    )

    # CFG-01 / D-01: only pass training-override args that were explicitly supplied on the CLI.
    # When a flag is absent, args.<flag> is None and the TrainConfig dataclass default is honored.
    _train_overrides = {k: v for k, v in {
        "max_epochs": args.max_epochs,
        "lr": args.lr,
        "seed": args.seed,
        "patience": args.patience,
        "stop_epoch": args.stop_epoch,
        "weight_decay": args.weight_decay,
        "early_stopping": args.early_stopping,
    }.items() if v is not None}
    train_cfg = TrainConfig(**_train_overrides)

    # Resolve survival loss for survival tasks (CLI override, else first
    # configured variant). Stays None for classification.
    survival_loss = None
    if task_cfg.task_type == "survival":
        survival_loss = args.survival_loss or (
            task_cfg.survival_losses[0] if task_cfg.survival_losses else "cox"
        )

    # CFG-01 / D-01: pass n_folds only when explicitly supplied; otherwise ExperimentConfig.n_folds applies.
    _exp_kwargs = {}
    if args.n_folds is not None:
        _exp_kwargs["n_folds"] = args.n_folds
    exp_cfg = ExperimentConfig(
        task=task_cfg,
        encoder_key=args.encoder,
        embed_dim=embed_dim,
        model=model_cfg,
        train=train_cfg,
        framework=framework,
        strategy=args.strategy,
        survival_loss=survival_loss,
        hparam_overrides=_parse_hparams(args.hparams),
        **_exp_kwargs,
    )

    # APL-02: apply registered model variant (if any) to exp_cfg before training.
    # Reads automil/applied_variant.json (written by `automil apply` and propagated
    # into the worktree by apply_overlay). No-op when no variant is selected or
    # when running outside autoMIL (applied_variant.json absent).
    from pathlib import Path as _Path
    from autobench.pipeline.variant_dispatch import apply_model_variant_to_exp_cfg
    _automil_dir = _Path("automil")
    # WR-03: warn when automil/ is absent so the operator knows variant dispatch
    # is skipped.  This happens on manual invocations from any directory that is
    # not the worktree root; under the orchestrator the cwd is always the worktree
    # root so automil/ is always present.
    if not _automil_dir.exists():
        print(
            f"[automil] WARNING: automil/ directory not found in cwd "
            f"({os.getcwd()}); variant dispatch skipped (running baseline).",
            flush=True,
        )
    apply_model_variant_to_exp_cfg(exp_cfg, _automil_dir)

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

    # Ensure data is prepared. TITAN skips the tile-encoder H5->PT step (no
    # "features" key), so pass no encoder_keys.
    from autobench.pipeline.prepare import prepare_all
    prepare_all(
        benchmark_dir=benchmark_dir,
        mapping_csv=ds.mapping_csv,
        features_base_dir=ds.features_base_dir,
        encoder_keys=[] if framework == Framework.TITAN else [args.encoder],
        ds=ds,
        seed=train_cfg.seed,
        n_splits=exp_cfg.n_folds,  # CFG-01: use resolved value, not args.n_folds (may be None)
    )

    # Run the experiment
    if framework == Framework.CLAM:
        from autobench.pipeline.clam.runner import run_experiment
        summary = run_experiment(
            exp_cfg, benchmark_dir, device,
            wandb_project=None if args.no_wandb else f"{ds.name}-automil",
            results_dir=automil_results_dir,
        )
    elif framework == Framework.TITAN:
        # TITAN is a frozen per-slide embedding -- its own prep/runner, no H5-bag step.
        from autobench.pipeline.titan.prepare import prepare_titan_experiment
        from autobench.pipeline.titan.runner import run_titan_experiment
        prepare_titan_experiment(
            benchmark_dir=benchmark_dir,
            task_name=args.task,
            features_base_dir=ds.features_base_dir,
        )
        summary = run_titan_experiment(
            exp_cfg, benchmark_dir, device=str(device),
            results_dir=automil_results_dir,  # CR-5: isolate per-experiment results
        )
    else:
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
            n_splits=exp_cfg.n_folds,  # CFG-01: use resolved value, not args.n_folds (may be None)
            task_type=task_cfg.task_type,
            event_col=task_cfg.event_col,
            time_col=task_cfg.time_col,
            survival_loss=survival_loss,
            nll_bins=task_cfg.nll_bins,
        )
        if framework == Framework.ABMIL:
            from autobench.pipeline.abmil.runner import run_abmil_experiment
            summary = run_abmil_experiment(exp_cfg, benchmark_dir, device=str(device),
                                           results_dir=automil_results_dir)  # CR-5
        elif framework == Framework.DTFD:
            from autobench.pipeline.dtfd import run_dtfd_experiment
            summary = run_dtfd_experiment(exp_cfg, benchmark_dir, device=str(device),
                                          results_dir=automil_results_dir)  # CR-5
        else:
            from autobench.pipeline.nnmil.runner import run_nnmil_experiment
            summary = run_nnmil_experiment(exp_cfg, benchmark_dir, device=str(device),
                                           results_dir=automil_results_dir)  # CR-5

    elapsed = time.time() - start_time

    # Write result.json (autoMIL contract)
    result = summary_to_result_json(summary, elapsed)
    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nExperiment complete in {elapsed:.0f}s")
    # val-firewall: surface only the validation selection signal to stdout/run.log;
    # test lives in the sealed held_out block (result['held_out']).
    if "val_c_index" in result["metrics"]:
        print(f"  val_c_index={result['metrics']['val_c_index']:.4f}  "
              f"composite={result['composite']:.4f}")
    else:
        print(f"  val_auc={result['metrics']['val_auc']:.4f}  "
              f"val_bacc={result['metrics']['val_bacc']:.4f}  "
              f"composite={result['composite']:.4f}")
    print(f"  result.json written to {os.path.abspath('result.json')}")


if __name__ == "__main__":
    main()
