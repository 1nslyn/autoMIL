#!/usr/bin/env python
"""Run a single benchmark experiment (one task × encoder × model).

Designed for use with autoMIL: runs one experiment and writes result.json
to the current working directory.

Examples
--------
# CLAM experiment
uv run --package autobench python benchmarks/scripts/run_experiment.py \
    --dataset ccrcc --task high_grade --encoder uni_v2 \
    --model clam_mb --framework clam

# nnMIL experiment
uv run --package autobench python benchmarks/scripts/run_experiment.py \
    --dataset ccrcc --task pbrm1 --encoder uni_v2 \
    --model ab_mil --framework nnmil

# ABMIL experiment (reuses nnMIL's H5-bag prep)
uv run --package autobench python benchmarks/scripts/run_experiment.py \
    --dataset ccrcc --task pbrm1 --encoder uni_v2 \
    --model abmil --framework abmil

# DTFD-MIL experiment (reuses nnMIL's H5-bag prep)
uv run --package autobench python benchmarks/scripts/run_experiment.py \
    --dataset ccrcc --task pbrm1 --encoder uni_v2 \
    --model dtfd_mil --framework dtfd

# TITAN experiment (frozen slide embedding -- --encoder must be "titan")
uv run --package autobench python benchmarks/scripts/run_experiment.py \
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
        "--benchmark-dir", "--benchmark_dir", dest="benchmark_dir", default=None,
        help=(
            "Explicit benchmark root. Results are written below its results/ "
            "directory instead of the dataset YAML's default benchmark_dir."
        ),
    )
    p.add_argument(
        "--skip-prep", "--skip_prep", dest="skip_prep", action="store_true",
        help=(
            "Use an already prepared benchmark root without creating or updating "
            "dataset_csv, splits, features, nnmil, or titan artifacts."
        ),
    )
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
    p.add_argument(
        "--folds", default=None,
        help="Comma-separated subset of the prepared fold indices to train "
             "(for staged campaigns; n_folds still defines the split set).",
    )
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
    p.add_argument(
        "--policy-variant", default=None,
        help="Registered train-only PolicyVariant under automil/variants/_policies/.",
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


def _parse_folds(raw: str | None, n_folds: int) -> tuple[int, ...] | None:
    """Parse one immutable subset of an already prepared split set."""
    if raw is None:
        return None
    try:
        values = tuple(int(part.strip()) for part in raw.split(","))
    except ValueError as exc:
        raise SystemExit("--folds must be comma-separated integer indices") from exc
    if not values or any(not part.strip() for part in raw.split(",")):
        raise SystemExit("--folds must contain at least one integer index")
    if len(set(values)) != len(values):
        raise SystemExit("--folds must not contain duplicate indices")
    if any(value < 0 or value >= n_folds for value in values):
        raise SystemExit(
            f"--folds indices must lie in [0, {n_folds}), got {values}"
        )
    return values


def _per_fold_composites(per_fold_val: list, is_survival: bool) -> list[float]:
    """The composite recomputed per fold — the input to its cross-fold SE (CR-4).

    The composite reported at the top of ``summary_to_result_json`` is a mean of
    fold MEANS, so its own spread is not recoverable from that number alone. Here
    the same formula is applied fold by fold, which is what makes the noise
    measurable at all.

    A fold missing ANY component of the composite is dropped whole rather than
    contributing a half-composite: averaging a fold's AUC with a missing balanced
    accuracy would report a value on a different scale from every other fold and
    inflate the spread.
    """
    out: list[float] = []
    for fm in per_fold_val or []:
        if not isinstance(fm, dict):
            continue
        keys = ("c_index",) if is_survival else ("auc_roc", "balanced_accuracy")
        vals = []
        for k in keys:
            v = fm.get(k)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                break
            f = float(v)
            if not math.isfinite(f):
                break
            vals.append(f)
        else:
            out.append(sum(vals) / len(vals))
    return out


def _finite_or_none(value: object) -> float | None:
    """Return ``value`` as a finite float, or ``None`` when it is not estimable.

    ``None`` serializes as JSON ``null``; a NaN would serialize as a bare ``NaN``
    token, which makes result.json invalid JSON and gets the whole file rejected
    at ingestion (see ``automil.runtime_helpers.json_safe``). Every number that
    can legitimately be unestimable passes through here on its way to disk.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _validation_fold_evidence(summary: dict) -> list[dict]:
    """Return the fold-indexed, validation-only evidence used by campaigns.

    The raw per-fold artifacts are born-sealed because each file also contains
    held-out metrics.  A stage controller must never open those files merely to
    recover validation values.  This deliberately narrow projection is safe to
    leave in the agent-facing ``result.json`` and is sufficient to prove exact
    fold coverage and recompute an equal-weight cross-stage mean.
    """
    per_fold = summary.get("per_fold_val", []) or []
    indices = summary.get("fold_indices")
    if not isinstance(indices, list) or len(indices) != len(per_fold):
        indices = list(range(len(per_fold)))
    is_survival = "c_index" in (summary.get("test") or {})
    evidence: list[dict] = []
    for fold_index, raw_metrics in zip(indices, per_fold):
        metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
        # Non-finite fold values are nulled here, not passed through raw: this
        # block ships in the AGENT-FACING copy of result.json, so a NaN AUC from
        # a fold that happened to miss a class would get the file rejected at
        # ingestion and the node recorded as a crash.
        if is_survival:
            value = _finite_or_none(metrics.get("c_index"))
            public_metrics = {"val_c_index": value}
            values = (value,)
        else:
            auc = _finite_or_none(metrics.get("auc_roc"))
            bacc = _finite_or_none(metrics.get("balanced_accuracy"))
            public_metrics = {"val_auc": auc, "val_bacc": bacc}
            values = (auc, bacc)
        finite = all(value is not None for value in values)
        evidence.append({
            "fold_index": fold_index,
            "metrics": public_metrics,
            "composite": (
                sum(values) / len(values) if finite else None
            ),
        })
    return evidence


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
    validation_folds = _validation_fold_evidence(summary)

    # Try to get peak VRAM
    peak_vram_mb = 0
    try:
        if torch.cuda.is_available():
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        pass

    # An unestimable metric is DROPPED from its block rather than written as NaN.
    # `metrics` and `held_out` are schema-constrained to numbers, and CR-1b
    # recomputes the composite as the mean of `metrics` — so the composite below
    # is likewise the mean of what survived, keeping reported and recomputed in
    # agreement. Which names went missing is reported via `unestimable`.
    if "c_index" in test:
        test_ci = _finite_or_none(test.get("c_index", {}).get("mean"))
        # The campaign ranks discovery, promotion, and the final winner by the
        # equal-weight mean of the same fold composites. Keep the graph-facing
        # result on that exact scale as well; ``val_pooled`` remains a useful
        # sealed diagnostic but must not silently change the search estimand.
        fold_values = [
            value
            for value in (
                _finite_or_none(fold.get("composite")) for fold in validation_folds
            )
            if value is not None
        ]
        val_ci = (
            math.fsum(fold_values) / len(fold_values)
            if fold_values
            else _finite_or_none(val.get("c_index", {}).get("mean"))
        )
        metrics = {"val_c_index": val_ci} if val_ci is not None else {}
        held_out = {"test_c_index": round(test_ci, 4)} if test_ci is not None else {}
        unestimable = [] if val_ci is not None else ["val_c_index"]
        composite = val_ci if val_ci is not None else 0.0
    else:
        candidates = {
            "val_auc": _finite_or_none(val.get("auc_roc", {}).get("mean")),
            "val_bacc": _finite_or_none(val.get("balanced_accuracy", {}).get("mean")),
        }
        held_out_candidates = {
            "test_auc": _finite_or_none(test.get("auc_roc", {}).get("mean")),
            "test_bacc": _finite_or_none(test.get("balanced_accuracy", {}).get("mean")),
        }
        metrics = {
            name: round(value, 4)
            for name, value in candidates.items() if value is not None
        }
        held_out = {
            name: round(value, 4)
            for name, value in held_out_candidates.items() if value is not None
        }
        unestimable = [name for name, value in candidates.items() if value is None]
        # ALL-OR-NOTHING, deliberately. An earlier revision reported the mean of
        # whichever components survived, so a node missing val_auc was scored on
        # val_bacc alone -- a different estimand, on a different scale, from
        # every sibling scored on (auc+bacc)/2. The composite formula is
        # pre-registered (`meta.scoring`) and the Ladder margin is declared
        # against it; silently swapping the estimand per node at runtime is the
        # same class of move the val-firewall and the Ladder exist to prevent.
        # It also leaked: `status: partial` keeps the node itself out of
        # KEEP_CLASS, but nothing stops it being a PARENT, and terminal_writer
        # gates a child against `parent["composite"]` with no partial check --
        # so a half-scale bar silently decided a completed child's keep/discard.
        # NOTE this does not close that leak, it only stops feeding it a
        # wrong-scale number: the parent bar becomes the 0.0 sentinel, which
        # auto-keeps every child. That is no worse than before (such a node was
        # a crash at composite 0.0), but the real fix is a parent-status gate in
        # terminal_writer/graph, which is deliberately out of scope here.
        # If a cell genuinely cannot estimate AUC, the honest fix is to declare a
        # bacc-only metric set for that cell up front, so every node in it is on
        # one scale.
        if unestimable:
            metrics = {}
            composite = 0.0
        else:
            composite = math.fsum(candidates.values()) / len(candidates)

    # A stage is complete only when every fold it declared has a finite
    # selection composite.  The old global ``>= 2`` threshold let a 2/3-fold
    # discovery attempt enter keep/UCB even though freeze later rejected it.
    # Promotion's declared 2/2 subset remains complete; a full run requires 5/5.
    per_fold_val = summary.get("per_fold_val", []) or []
    n_folds_total = summary.get("n_folds", len(per_fold_val))
    valid_fold_composites = _per_fold_composites(
        per_fold_val, is_survival="c_index" in test,
    )
    n_valid_folds = len(valid_fold_composites)
    selected = summary.get("fold_indices")
    if selected is None:
        required_folds = n_folds_total
        declared_coverage_valid = (
            type(required_folds) is int
            and required_folds > 0
            and len(per_fold_val) == required_folds
        )
    else:
        declared_coverage_valid = (
            isinstance(selected, list)
            and len(selected) == len(per_fold_val)
            and all(type(fold) is int for fold in selected)
            and len(set(selected)) == len(selected)
        )
        required_folds = len(selected) if declared_coverage_valid else -1
    status = (
        "completed"
        if declared_coverage_valid and n_valid_folds == required_folds
        else "partial"
    )
    # A selection signal missing a component is not a completed run. Say so as a
    # quarantined `partial` with a readable cause (D-01), rather than letting a
    # NaN reach disk and get the node written off as a phantom crash.
    if unestimable:
        status = "partial"

    # CR-4: measure the noise the Ladder keep-margin is supposed to exceed.
    # `composite_se` is TOP-LEVEL, deliberately: CR-1b recomputes the composite as
    # the mean of `metrics`, so an extra key in there would corrupt the very
    # selection signal this is meant to protect. None (not 0.0) when fewer than
    # two folds are estimable — 0.0 would read as "measured, noise-free".
    from automil.scoring import cross_fold_se

    composite_se = cross_fold_se(valid_fold_composites)

    # ``metrics`` is agent-facing (val only); ``held_out`` (test) + ``summary``
    # are sealed into certify.json by terminal_writer — never seen during search.
    result = {
        "status": status,
        "metrics": metrics,
        "held_out": held_out,
        "composite": composite if "c_index" in test else round(composite, 4),
        "composite_se": composite_se,
        "elapsed_seconds": round(elapsed, 1),
        "peak_vram_mb": round(peak_vram_mb),
        "n_valid_folds": n_valid_folds,
        "n_folds": n_folds_total,
        "validation_folds": validation_folds,
        "summary": summary,
    }
    if unestimable:
        # Describes the all-or-nothing rule above. An earlier draft said the
        # composite was "the mean of the N metric(s) that were estimable",
        # which was left over from the partial-mean semantics this replaced:
        # `metrics` is now always {} here, so N was always 0, and the composite
        # is a sentinel rather than any mean. The trigger is the pooled
        # cross-fold mean being non-finite -- which happens only when NO fold
        # was estimable, since compute_confidence_intervals already drops
        # non-finite folds per metric. This string is agent-facing (`error` is
        # not in _SEALED_RESULT_KEYS), so it has to be true.
        result["error"] = (
            f"composite not reported: {', '.join(unestimable)} was unestimable "
            "across every fold, and the composite is only defined over its full "
            "declared metric set. composite=0.0 is a sentinel, not a score; the "
            "node is quarantined as partial."
        )
    return result


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
    resolved_n_folds = args.n_folds if args.n_folds is not None else 5
    if args.n_folds is not None:
        _exp_kwargs["n_folds"] = args.n_folds
    fold_indices = _parse_folds(args.folds, resolved_n_folds)
    if fold_indices is not None:
        _exp_kwargs["fold_indices"] = fold_indices
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
        policy_variant=args.policy_variant,
        dataset=ds.name,  # DATA-ID: prefer the resolved DatasetConfig name over args.dataset
        **_exp_kwargs,
    )

    # APL-02: apply registered model variant (if any) to exp_cfg before training.
    # Reads automil/applied_variant.json (written by `automil apply` and propagated
    # into the worktree by apply_overlay). No-op when no variant is selected or
    # when running outside autoMIL (applied_variant.json absent).
    from autobench.pipeline.variant_dispatch import apply_model_variant_to_exp_cfg
    from autobench.pipeline.policy_dispatch import (
        resolve_policy_name,
        runtime_automil_dir,
    )
    _automil_dir = runtime_automil_dir()
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
    # Resolve the train-only policy before any runner fingerprints or writes the
    # ExperimentConfig. An archived selection must be provenance-equivalent to
    # an explicit --policy-variant, never an invisible runtime side channel.
    exp_cfg.policy_variant = resolve_policy_name(exp_cfg, _automil_dir)

    benchmark_dir = os.path.abspath(
        os.path.expanduser(args.benchmark_dir or ds.benchmark_dir)
    )
    print(f"  Benchmark root: {benchmark_dir}")
    if args.skip_prep:
        print("  Data preparation: skipped by explicit --skip-prep")

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
    if not args.skip_prep:
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
        if not args.skip_prep:
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
        if not args.skip_prep:
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

    # Write result.json (autoMIL contract).
    #
    # L-3: split-written across the val-firewall boundary. The FULL payload —
    # `held_out` included — goes to the sealed AUTOMIL_RESULTS_DIR; the copy
    # that lands in the worktree is stripped. Writing it here directly, as this
    # did, left the test metrics sitting in `.automil_worktrees/<node>/result.json`
    # for the entire run, in a directory with no access control of its own —
    # readable by anything that can read the project tree, including the agent
    # driving the search, without waiting for `automil certify`.
    result = summary_to_result_json(summary, elapsed)
    from automil.runtime_helpers import write_result_json

    write_result_json(result)

    print(f"\nExperiment complete in {elapsed:.0f}s")
    # val-firewall: surface only the validation selection signal to stdout/run.log;
    # test lives in the sealed held_out block (result['held_out']).
    # Format whatever `metrics` actually holds. It used to always carry both
    # names (NaN-valued at worst, which formats fine), so indexing them was safe;
    # an unestimable metric is now DROPPED, and `metrics` can be empty. Indexing
    # raised KeyError here -- after result.json was already written, so the
    # damage was not a lost result but a non-zero exit: the campaign's native
    # baseline stage refuses to archive a result whose process exited non-zero
    # and aborts the stage, discarding exactly the run this whole change exists
    # to rescue.
    reported = "  ".join(
        f"{name}={value:.4f}" for name, value in sorted(result["metrics"].items())
    )
    summary_line = f"composite={result['composite']:.4f}"
    if reported:
        summary_line = f"{reported}  {summary_line}"
    print(f"  {summary_line}")
    if result.get("error"):
        print(f"  {result['error']}")
    print(f"  result.json written to {os.path.abspath('result.json')}")


if __name__ == "__main__":
    main()
