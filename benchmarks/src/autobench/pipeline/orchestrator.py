"""Benchmark orchestrator: single-GPU and multi-GPU experiment grid runner."""

from __future__ import annotations

import contextlib
import fcntl
import gc
import glob
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field

import pandas as pd
from tqdm import tqdm

from autobench.config import DatasetConfig
from autobench.pipeline.config import (
    BenchmarkConfig,
    ExperimentConfig,
    Framework,
    Registries,
    StrategyConfig,
    generate_all_experiments,
)
from autobench.pipeline.prepare import prepare_all


# ---------------------------------------------------------------------------
# Thread-safe completion tracking
# ---------------------------------------------------------------------------


def _completed_path(benchmark_dir: str) -> str:
    return os.path.join(benchmark_dir, "results", "_completed.json")


def load_completed(benchmark_dir: str) -> set[str]:
    path = _completed_path(benchmark_dir)
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            return set(json.load(f))
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def mark_completed(benchmark_dir: str, experiment_id: str) -> None:
    os.makedirs(os.path.join(benchmark_dir, "results"), exist_ok=True)
    path = _completed_path(benchmark_dir)
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            content = f.read().strip()
            completed = set(json.loads(content)) if content else set()
            completed.add(experiment_id)
            f.seek(0)
            f.truncate()
            f.write(json.dumps(sorted(completed), indent=2))
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Thread-safe failure tracking
# ---------------------------------------------------------------------------


def _failed_path(benchmark_dir: str) -> str:
    return os.path.join(benchmark_dir, "results", "_failed.json")


def load_failed(benchmark_dir: str) -> dict[str, dict]:
    """Load persistent failure records.

    Returns ``{experiment_id: {reason, detail, timestamp, gpu_id, estimated_vram_gb}}``.
    """
    path = _failed_path(benchmark_dir)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            return json.load(f)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def mark_failed(
    benchmark_dir: str,
    experiment_id: str,
    reason: str,
    detail: str = "",
    gpu_id: int | None = None,
    estimated_vram_gb: float | None = None,
) -> None:
    """Persist a failure record for auditing/debugging."""
    os.makedirs(os.path.join(benchmark_dir, "results"), exist_ok=True)
    path = _failed_path(benchmark_dir)
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            content = f.read().strip()
            failed = json.loads(content) if content else {}
            failed[experiment_id] = {
                "reason": reason,
                "detail": detail,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "gpu_id": gpu_id,
                "estimated_vram_gb": estimated_vram_gb,
            }
            f.seek(0)
            f.truncate()
            f.write(json.dumps(failed, indent=2))
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def clear_failed(benchmark_dir: str, experiment_id: str) -> None:
    """Remove stale failure records once an experiment eventually succeeds."""
    path = _failed_path(benchmark_dir)
    if not os.path.exists(path):
        return
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            content = f.read().strip()
            failed = json.loads(content) if content else {}
            if experiment_id not in failed:
                return
            failed.pop(experiment_id, None)
            f.seek(0)
            f.truncate()
            f.write(json.dumps(failed, indent=2))
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# VRAM estimation
# ---------------------------------------------------------------------------

_MODEL_BASE_VRAM: dict[str, float] = {  # GB at embed_dim=768
    "simple_mil": 2.5,
    "ab_mil": 3.0,
    "ds_mil": 4.0,
    "dtfd_mil": 4.0,
    "wikg_mil": 4.0,
    "clam_sb": 3.0,
    "clam_mb": 3.5,
    "mil": 2.5,
    "ilra_mil": 10.0,
    "rrt": 8.0,
    "trans_mil": 12.0,
    "vision_transformer": 16.0,
}

_CUDA_CONTEXT_GB = 1.8

_ATTENTION_MODELS = {"trans_mil", "vision_transformer", "rrt", "ilra_mil"}

# Retry policy: OOM experiments get re-queued with a bumped estimate so
# they run with fewer concurrent jobs.  If the estimate reaches the full
# GPU capacity (solo run) and it still OOMs, it's permanently failed.
_MAX_OOM_RETRIES = 3
_OOM_ESTIMATE_MULTIPLIER = 1.5
_RETRIABLE_FAILURES = {"OOM", "TRANSIENT_NO_VRAM", "SKIP"}


def estimate_vram_gb(exp: ExperimentConfig) -> float:
    """Estimate peak VRAM usage for an experiment in GiB.

    Uses a lookup table by model_type, scaled by embed_dim relative to
    the 768-dim baseline.  Attention-heavy models scale super-linearly.
    """
    base = _MODEL_BASE_VRAM.get(exp.model.model_type, 6.0)
    dim_ratio = exp.embed_dim / 768.0
    if exp.model.model_type in _ATTENTION_MODELS:
        dim_factor = dim_ratio ** 1.5
    else:
        dim_factor = dim_ratio
    return round(base * dim_factor + _CUDA_CONTEXT_GB, 1)


# ---------------------------------------------------------------------------
# Results aggregation
# ---------------------------------------------------------------------------


def aggregate_results(summaries: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for s in summaries:
        row = {
            "framework": s.get("framework", "clam"),
            "strategy": s["strategy"],
            "task": s["task"],
            "encoder": s["encoder"],
            "model_type": s["model_type"],
            "embed_dim": s["embed_dim"],
            "n_folds": s["n_folds"],
            "seed": s["seed"],
        }
        for split_name in ("test", "val"):
            for metric_name, metric_data in s[split_name].items():
                for stat in ("mean", "std", "ci_low", "ci_high"):
                    row[f"{split_name}_{metric_name}_{stat}"] = metric_data[stat]
        rows.append(row)
    return pd.DataFrame(rows)


def _load_or_collect_summary(benchmark_dir: str, exp_cfg_or_id) -> dict | None:
    """Load summary.json for an experiment.

    Accepts either an ExperimentConfig or an experiment_id string.
    """
    if isinstance(exp_cfg_or_id, ExperimentConfig):
        # New path: results/{framework}/{strategy}/{task}/{encoder}/{model}/
        path = os.path.join(
            benchmark_dir, "results", exp_cfg_or_id.results_subdir, "summary.json",
        )
        if not os.path.exists(path):
            # Legacy path fallback: results/{task}/{encoder}/{model}/
            path = os.path.join(
                benchmark_dir, "results",
                exp_cfg_or_id.task.name, exp_cfg_or_id.encoder_key,
                exp_cfg_or_id.model.model_type, "summary.json",
            )
    else:
        # Legacy flat path fallback
        path = os.path.join(benchmark_dir, "results", exp_cfg_or_id, "summary.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Data preparation dispatcher
# ---------------------------------------------------------------------------


def _prepare_data(cfg: BenchmarkConfig, ds: DatasetConfig) -> None:
    """Run data preparation."""
    prepare_all(
        benchmark_dir=cfg.benchmark_dir,
        mapping_csv=cfg.mapping_csv,
        features_base_dir=cfg.features_base_dir,
        encoder_keys=cfg.encoder_keys,
        ds=ds,
        seed=cfg.train.seed,
        n_splits=cfg.n_folds,
        holdout_frac=cfg.holdout_frac,
    )


def _prepare_nnmil_plans(
    cfg: BenchmarkConfig,
    experiments: list[ExperimentConfig],
    registries: Registries | None = None,
    dataset_name: str = "dataset",
) -> None:
    """Generate nnMIL plan files for all unique (task, encoder, strategy) combos."""
    from autobench.pipeline.nnmil.prepare import prepare_nnmil_experiment

    seen: set[str] = set()
    for exp in experiments:
        if exp.framework != Framework.NNMIL:
            continue
        key = f"{exp.task.name}__{exp.encoder_key}__{exp.strategy}"
        if key in seen:
            continue
        seen.add(key)

        prepare_nnmil_experiment(
            benchmark_dir=cfg.benchmark_dir,
            task_name=exp.task.name,
            encoder_key=exp.encoder_key,
            strategy=exp.strategy,
            label_col=exp.task.label_col,
            label_dict=exp.task.label_dict,
            embed_dim=exp.embed_dim,
            features_base_dir=cfg.features_base_dir,
            dataset_name=dataset_name,
            seed=cfg.train.seed,
            n_splits=cfg.n_folds,
            max_seq_multiplier=cfg.nnmil_max_seq_multiplier,
            use_original_length=cfg.nnmil_use_original_length,
        )


# ---------------------------------------------------------------------------
# Experiment dispatch
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _isolated_torch_state():
    """Isolate each experiment from process-global torch state set by the previous one.

    The single-GPU runner — and multi-GPU workers recycled via
    ``max_tasks_per_child`` — execute heterogeneous third-party trainers
    back-to-back in one process.  Several relevant torch switches are
    *thread-local globals*, not per-tensor/per-model state, so one trainer can
    silently change the next one's behaviour:

    * ``torch.set_grad_enabled`` — nnMIL's ``ClassificationTrainer.evaluate``
      leaves grad **disabled** on return; the next CLAM experiment's first
      ``backward()`` then raises "element 0 of tensors does not require grad
      and does not have a grad_fn" (the original crash).
    * ``torch.use_deterministic_algorithms`` — nnMIL's ``set_random_seeds``
      enables it globally; CLAM never sets it, so a CLAM experiment that ran
      *after* an nnMIL one would select different (deterministic) kernels than
      one that ran first.  That makes a model's metrics depend on scheduling
      order, which breaks benchmark fairness.

    Restoring the canonical pristine state (grad on, deterministic-algorithms
    off — the process defaults) on both entry and exit makes every experiment
    behave as if it were the only one in a fresh process.  nnMIL re-applies its
    own determinism inside its trainer ``__init__``, so this changes neither
    nnMIL's metrics nor a first-run CLAM experiment's metrics — it only removes
    the order-dependence.

    Residual: ``CUBLAS_WORKSPACE_CONFIG`` (also set by nnMIL's
    ``set_random_seeds``) is read by cuBLAS at context-creation time and cannot
    be reliably reset mid-process; the multi-GPU path (one fresh process per
    experiment) is the way to isolate that fully.
    """
    import torch  # local: keep torch out of the module-import path (see _gpu_worker)

    def _pristine() -> None:
        torch.set_grad_enabled(True)
        torch.use_deterministic_algorithms(False)

    _pristine()
    try:
        yield
    finally:
        _pristine()


def _run_single_experiment_dispatch(
    exp_cfg: ExperimentConfig,
    benchmark_dir: str,
    device: torch.device,
    wandb_project: str | None = None,
) -> dict:
    """Dispatch to CLAM or nnMIL runner based on framework."""
    with _isolated_torch_state():
        if exp_cfg.framework == Framework.NNMIL:
            from autobench.pipeline.nnmil.runner import run_nnmil_experiment
            return run_nnmil_experiment(exp_cfg, benchmark_dir, device=str(device))
        else:
            from autobench.pipeline.clam.runner import run_experiment
            return run_experiment(exp_cfg, benchmark_dir, device, wandb_project)


# ---------------------------------------------------------------------------
# Single-GPU benchmark
# ---------------------------------------------------------------------------


def run_benchmark(
    cfg: BenchmarkConfig,
    ds: DatasetConfig,
    registries: Registries | None = None,
) -> pd.DataFrame:
    """Run all experiments sequentially on a single GPU."""
    import torch
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
    _prepare_data(cfg, ds=ds)

    experiments = generate_all_experiments(cfg, registries) if registries else []

    # Prepare nnMIL plans if needed
    nnmil_experiments = [e for e in experiments if e.framework == Framework.NNMIL]
    if nnmil_experiments:
        print("\n" + "=" * 60)
        print("NNMIL PLAN GENERATION")
        print("=" * 60)
        dataset_name = ds.name if ds else "dataset"
        _prepare_nnmil_plans(cfg, nnmil_experiments, registries=registries, dataset_name=dataset_name)

    experiments.sort(key=lambda e: (e.encoder_key, e.task.name, e.model.model_type))
    completed = load_completed(cfg.benchmark_dir)
    print(f"\nTotal experiments: {len(experiments)}  "
          f"(already completed: {len(completed)})")

    current_encoder: str | None = None
    n_skipped = sum(1 for e in experiments if e.experiment_id in completed)

    pbar = tqdm(total=len(experiments), initial=n_skipped, desc="Benchmark",
                unit="exp", dynamic_ncols=True)
    for i, exp_cfg in enumerate(experiments):
        exp_id = exp_cfg.experiment_id

        # Already-completed experiments produced summary.json on a previous run
        # and will be picked up by the on-disk scan in _finalize.
        if exp_id in completed:
            continue

        # GPU cleanup on encoder switch
        if current_encoder is not None and current_encoder != exp_cfg.encoder_key:
            gc.collect()
            torch.cuda.empty_cache()  # torch imported at top of run_benchmark
        current_encoder = exp_cfg.encoder_key

        pbar.set_postfix_str(exp_id)
        _run_single_experiment_dispatch(
            exp_cfg, cfg.benchmark_dir, device, cfg.wandb_project,
        )
        mark_completed(cfg.benchmark_dir, exp_id)
        pbar.update(1)
    pbar.close()

    return _finalize(cfg.benchmark_dir, collect_all_summaries_on_disk(cfg.benchmark_dir))


# ---------------------------------------------------------------------------
# Multi-GPU benchmark
# ---------------------------------------------------------------------------


@dataclass
class _GpuState:
    """Per-GPU scheduling state for budget-based concurrency."""

    gpu_id: int
    total_gb: float
    reserved_gb: float = 0.0
    active: dict = field(default_factory=dict)     # {future: (exp_id, vram_est)}

    @property
    def budget_free(self) -> float:
        return self.total_gb - self.reserved_gb


def _validate_expected_completed(
    benchmark_dir: str, expected_experiment_ids: set[str],
) -> set[str]:
    """Validate strict completeness and return the final completed set."""
    completed_final = load_completed(benchmark_dir)
    missing = sorted(expected_experiment_ids - completed_final)
    if missing:
        preview = ", ".join(missing[:5])
        suffix = f" ... +{len(missing) - 5}" if len(missing) > 5 else ""
        raise RuntimeError(
            f"Strict-completion check failed: {len(missing)} experiments are not "
            f"completed ({preview}{suffix})"
        )
    return completed_final


def _collect_all_summaries_or_raise(
    benchmark_dir: str,
    experiments: list[ExperimentConfig],
) -> list[dict]:
    """Collect all expected summaries and fail if any are missing."""
    all_summaries: list[dict] = []
    missing_summary_ids: list[str] = []
    for exp in experiments:
        summary = _load_or_collect_summary(benchmark_dir, exp)
        if summary is None:
            missing_summary_ids.append(exp.experiment_id)
            continue
        all_summaries.append(summary)

    if missing_summary_ids:
        preview = ", ".join(missing_summary_ids[:5])
        suffix = (
            f" ... +{len(missing_summary_ids) - 5}"
            if len(missing_summary_ids) > 5 else ""
        )
        raise RuntimeError(
            f"Missing summary.json for {len(missing_summary_ids)} completed "
            f"experiments ({preview}{suffix})"
        )
    return all_summaries


def collect_all_summaries_on_disk(benchmark_dir: str) -> list[dict]:
    """Collect every ``summary.json`` under ``results/<fw>/<strategy>/.../``.

    Used by ``_finalize`` so the rolled-up CSVs reflect every experiment that
    has ever completed for this benchmark dir, not just the subset selected by
    the current run's ``--encoders``/``--models``/``--frameworks`` filters.
    Without this, re-running with a narrower filter silently truncates the
    aggregated CSV (and downstream consumers go stale).
    """
    results_root = os.path.join(benchmark_dir, "results")
    if not os.path.isdir(results_root):
        return []

    summaries: list[dict] = []
    seen_keys: set[tuple] = set()
    # Layout: results/<fw>/<strategy>/<task>/<encoder>/<model>/summary.json
    pattern = os.path.join(results_root, "*", "*", "*", "*", "*", "summary.json")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path) as f:
                s = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        # Dedupe by (framework, strategy, task, encoder, model_type) — defensive
        # in case both the new and legacy result paths happen to coexist.
        key = (
            s.get("framework", "clam"),
            s.get("strategy"),
            s.get("task"),
            s.get("encoder"),
            s.get("model_type"),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        summaries.append(s)
    return summaries


def run_benchmark_multigpu(
    cfg: BenchmarkConfig,
    gpu_ids: list[int],
    ds: DatasetConfig,
    registries: Registries | None = None,
) -> pd.DataFrame:
    """Distribute experiments across GPUs using memory-budget scheduling.

    Strict completion semantics:
    - A phase succeeds only if every expected experiment is completed.
    - Historical failures in ``_failed.json`` are audit-only and never skipped.
    - Non-OOM failures are fatal immediately.
    - OOM/transient-no-VRAM failures are retried by reducing concurrency only.
    """
    import sys

    print("=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
    _prepare_data(cfg, ds=ds)

    experiments = generate_all_experiments(cfg, registries) if registries else []

    # Prepare nnMIL plans if needed
    nnmil_experiments = [e for e in experiments if e.framework == Framework.NNMIL]
    if nnmil_experiments:
        print("\n" + "=" * 60)
        print("NNMIL PLAN GENERATION")
        print("=" * 60)
        dataset_name = ds.name if ds else "dataset"
        _prepare_nnmil_plans(cfg, nnmil_experiments, registries=registries, dataset_name=dataset_name)

    expected_experiment_ids = {e.experiment_id for e in experiments}
    completed = load_completed(cfg.benchmark_dir)
    pending = [e for e in experiments if e.experiment_id not in completed]
    failed_records = load_failed(cfg.benchmark_dir)
    n_failed_history = sum(1 for eid in expected_experiment_ids if eid in failed_records)
    if n_failed_history:
        print(
            f"\nFound {n_failed_history} historical failure records in _failed.json "
            "(audit-only; experiments will still be scheduled)"
        )

    # Query real GPU VRAM
    from autobench.pipeline._gpu_worker import _is_cuda_oom, query_gpu_vram, gpu_init, run_single_experiment

    # Budget against FREE VRAM, not total: a co-tenant or stale allocation on a
    # shared H100 would otherwise make the packer over-commit and OOM (job
    # 44355372). On a clean GPU free ≈ total, so this is a no-op there.
    gpu_vram = query_gpu_vram(gpu_ids, metric="free")
    reserve_gb = 2.0  # reserve for OS / display server

    gpu_states: dict[int, _GpuState] = {
        g: _GpuState(gpu_id=g, total_gb=max(0.0, gpu_vram[g] - reserve_gb))
        for g in gpu_ids
    }
    if not gpu_states:
        raise ValueError("No GPUs provided for multi-GPU benchmark.")

    max_gpu_usable_gb = max(gs.total_gb for gs in gpu_states.values())

    # Pre-estimate VRAM for all pending experiments.
    # Oversized estimates are capped at max usable GPU memory and retried
    # with solo scheduling if needed.
    exp_vram: dict[str, float] = {}
    exp_configs: dict[str, ExperimentConfig] = {}
    for exp in pending:
        est = estimate_vram_gb(exp)
        est = min(est, max_gpu_usable_gb) if max_gpu_usable_gb > 0 else est
        exp_vram[exp.experiment_id] = est
        exp_configs[exp.experiment_id] = exp

    print(f"\nTotal: {len(experiments)}  Completed: {len(completed)}  "
          f"Pending: {len(pending)}  GPUs: {gpu_ids}")
    for g in gpu_ids:
        print(f"  GPU {g}: {gpu_vram[g]:.1f} GB free, "
              f"{gpu_vram[g] - reserve_gb:.1f} GB usable (budgeting against free VRAM)")
    log_dir = os.path.join(cfg.benchmark_dir, "logs")
    print(f"Per-experiment logs: {log_dir}/<framework>/<strategy>/"
          f"<task>__<encoder>__<model>__s<seed>.log")

    if not pending:
        _validate_expected_completed(cfg.benchmark_dir, expected_experiment_ids)
        _collect_all_summaries_or_raise(cfg.benchmark_dir, experiments)
        return _finalize(cfg.benchmark_dir, collect_all_summaries_on_disk(cfg.benchmark_dir))

    pending_queue: list[ExperimentConfig] = sorted(
        pending,
        key=lambda e: exp_vram[e.experiment_id],
        reverse=True,
    )

    queue_preview = ", ".join(
        f"{e.model.model_type}({exp_vram[e.experiment_id]:.1f}G)"
        for e in pending_queue[:8]
    )
    queue_suffix = f"... +{len(pending_queue)-8}" if len(pending_queue) > 8 else ""
    print(f"Global pending queue: {len(pending_queue)} exps [{queue_preview}{queue_suffix}]")

    # ---- Budget-based scheduling loop ----
    ctx = multiprocessing.get_context("spawn")
    pools: dict[int, ProcessPoolExecutor] = {}
    all_futures: dict = {}  # future -> (gpu_id, exp_id, vram_est)
    n_done = 0
    oom_retries: dict[str, int] = {}  # exp_id -> number of retries so far
    had_error = False

    pbar = tqdm(total=len(pending_queue), desc="Multi-GPU benchmark",
                unit="exp", dynamic_ncols=True)

    def _insert_pending(exp: ExperimentConfig) -> None:
        pending_queue.append(exp)
        pending_queue.sort(key=lambda e: exp_vram[e.experiment_id], reverse=True)

    def _pop_largest_that_fits(free_gb: float) -> ExperimentConfig | None:
        for idx, exp in enumerate(pending_queue):
            if exp_vram[exp.experiment_id] <= free_gb:
                return pending_queue.pop(idx)
        return None

    def _try_submit(gs: _GpuState) -> int:
        """Submit as many largest-fit jobs as possible for one GPU."""
        submitted = 0
        while gs.gpu_id in pools:
            exp = _pop_largest_that_fits(gs.budget_free)
            if exp is None:
                break
            est = exp_vram[exp.experiment_id]
            future = pools[gs.gpu_id].submit(
                run_single_experiment,
                exp,
                cfg.benchmark_dir,
                cfg.wandb_project,
            )
            gs.active[future] = (exp.experiment_id, est)
            gs.reserved_gb += est
            all_futures[future] = (gs.gpu_id, exp.experiment_id, est)
            submitted += 1
        return submitted

    try:
        # Create one pool per GPU (high max_workers; budget controls concurrency)
        for g in gpu_ids:
            pools[g] = ProcessPoolExecutor(
                max_workers=8,
                mp_context=ctx,
                initializer=gpu_init,
                initargs=(g,),
                max_tasks_per_child=cfg.max_tasks_per_child,
            )

        while pending_queue or all_futures:
            for gs in gpu_states.values():
                _try_submit(gs)

            # Pending exists but nothing is running: schedule deadlock
            if pending_queue and not all_futures:
                largest_pending_gb = max(
                    exp_vram[e.experiment_id] for e in pending_queue
                )
                max_gpu_budget_gb = max(gs.total_gb for gs in gpu_states.values())
                raise RuntimeError(
                    "Scheduler stalled: pending experiments remain but none can be "
                    f"submitted (largest pending {largest_pending_gb:.1f} GB, "
                    f"max GPU budget {max_gpu_budget_gb:.1f} GB)"
                )

            if not all_futures:
                break

            done, _ = wait(all_futures, return_when=FIRST_COMPLETED, timeout=5.0)
            if not done:
                continue

            for future in done:
                gpu_id, exp_id, vram_est = all_futures.pop(future)
                gs = gpu_states[gpu_id]
                gs.active.pop(future, None)
                gs.reserved_gb = max(0.0, gs.reserved_gb - vram_est)

                try:
                    result = future.result()
                except Exception as exc:
                    reason = type(exc).__name__
                    detail = str(exc).split("\n")[0][:200]
                    mark_failed(
                        cfg.benchmark_dir, exp_id,
                        reason=reason, detail=detail,
                        gpu_id=gpu_id, estimated_vram_gb=vram_est,
                    )
                    pbar.clear()
                    print(f"  GPU {gpu_id} / {exp_id}: [{reason}] {detail}",
                          file=sys.stderr, flush=True)
                    pbar.refresh()
                    raise RuntimeError(
                        f"Non-OOM worker failure for {exp_id}: [{reason}] {detail}"
                    ) from exc

                if isinstance(result, dict) and result.get("_failed"):
                    reason = result["reason"]
                    detail = result.get("detail", "")
                    # Defensive reclassification: any failure whose detail
                    # mentions "out of memory" is treated as OOM regardless
                    # of the original exception type (e.g. AcceleratorError).
                    if reason not in _RETRIABLE_FAILURES and _is_cuda_oom(detail):
                        reason = "OOM"
                    retries = oom_retries.get(exp_id, 0)
                    already_solo = vram_est >= max_gpu_usable_gb
                    can_retry = (
                        reason in _RETRIABLE_FAILURES
                        and retries < _MAX_OOM_RETRIES
                        and not already_solo
                    )

                    if can_retry:
                        new_est = round(vram_est * _OOM_ESTIMATE_MULTIPLIER, 1)
                        new_est = min(new_est, max_gpu_usable_gb)
                        oom_retries[exp_id] = retries + 1
                        exp_vram[exp_id] = new_est
                        _insert_pending(exp_configs[exp_id])
                        pbar.clear()
                        print(
                            f"  GPU {gpu_id} / {exp_id}: [{reason}] retrying "
                            f"(attempt {retries + 2}/{_MAX_OOM_RETRIES + 1}, "
                            f"est {vram_est:.1f}->{new_est:.1f} GB)",
                            file=sys.stderr, flush=True,
                        )
                        pbar.refresh()
                        continue

                    mark_failed(
                        cfg.benchmark_dir, exp_id,
                        reason=reason, detail=detail,
                        gpu_id=gpu_id, estimated_vram_gb=vram_est,
                    )
                    pbar.clear()
                    print(f"  GPU {gpu_id} / {exp_id}: [{reason}] {detail}",
                          file=sys.stderr, flush=True)
                    pbar.refresh()
                    if reason in _RETRIABLE_FAILURES:
                        raise RuntimeError(
                            f"Experiment {exp_id} failed after retry budget: "
                            f"[{reason}] {detail}"
                        )
                    raise RuntimeError(
                        f"Non-OOM experiment failure for {exp_id}: [{reason}] {detail}"
                    )

                if result is None:
                    mark_failed(
                        cfg.benchmark_dir, exp_id,
                        reason="UNKNOWN", detail="returned None",
                        gpu_id=gpu_id, estimated_vram_gb=vram_est,
                    )
                    pbar.clear()
                    print(f"  GPU {gpu_id} / {exp_id}: [UNKNOWN] returned None",
                          file=sys.stderr, flush=True)
                    pbar.refresh()
                    raise RuntimeError(f"Experiment {exp_id} returned None")

                clear_failed(cfg.benchmark_dir, exp_id)
                n_done += 1
                pbar.update(1)
                pbar.set_postfix_str(f"ok={n_done}")
                pbar.clear()
                print(f"  GPU {gpu_id} / {exp_id}: [OK]",
                      file=sys.stderr, flush=True)
                pbar.refresh()

    except KeyboardInterrupt:
        had_error = True
        print("\n\nInterrupted -- killing workers...")
        raise SystemExit(130)
    except Exception:
        had_error = True
        raise
    finally:
        for pool in pools.values():
            pool.shutdown(wait=not had_error, cancel_futures=had_error)
        pbar.close()

    _validate_expected_completed(cfg.benchmark_dir, expected_experiment_ids)
    _collect_all_summaries_or_raise(cfg.benchmark_dir, experiments)
    return _finalize(cfg.benchmark_dir, collect_all_summaries_on_disk(cfg.benchmark_dir))


# ---------------------------------------------------------------------------
# Finalization
# ---------------------------------------------------------------------------


def _finalize(benchmark_dir: str, all_summaries: list[dict]) -> pd.DataFrame:
    results_df = aggregate_results(all_summaries)

    # Write one CSV/JSON per (framework, strategy) under aggregated/{framework}/.
    agg_dir = os.path.join(benchmark_dir, "aggregated")
    written: list[str] = []

    for (fw, strat), group_df in results_df.groupby(["framework", "strategy"]):
        fw_dir = os.path.join(agg_dir, fw)
        os.makedirs(fw_dir, exist_ok=True)

        csv_path = os.path.join(fw_dir, f"{strat}.csv")
        group_df.to_csv(csv_path, index=False)

        group_summaries = [
            s for s in all_summaries
            if s.get("framework", "clam") == fw
            and s.get("strategy") == strat
        ]
        json_path = os.path.join(fw_dir, f"{strat}.json")
        with open(json_path, "w") as f:
            json.dump(group_summaries, f, indent=2)

        written.append(csv_path)

    print(f"\nBenchmark complete: {len(all_summaries)} experiments")
    print(f"Results ({len(written)} files) in {agg_dir}/")
    for p in written:
        print(f"  {os.path.relpath(p, benchmark_dir)}")
    # Print a compact summary table
    if not results_df.empty:
        cols = ["framework", "strategy", "task", "encoder", "model_type",
                "test_auc_roc_mean", "test_auc_roc_ci_low",
                "test_auc_roc_ci_high", "test_balanced_accuracy_mean"]
        avail = [c for c in cols if c in results_df.columns]
        print(results_df[avail].to_string(index=False, float_format="%.3f"))
    return results_df
