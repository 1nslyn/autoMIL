"""GPU worker entry point for multi-GPU benchmark.

This module intentionally avoids importing torch or any CUDA-dependent
module at the top level.  When ``ProcessPoolExecutor`` (spawn context)
unpickles worker functions, it only imports *this* lightweight module.
``CUDA_VISIBLE_DEVICES`` is therefore set **before** ``torch`` is first
imported in the subprocess, guaranteeing the CUDA runtime binds to the
correct physical GPU.
"""

from __future__ import annotations

import os
import signal
import subprocess


def _is_cuda_oom(text: str) -> bool:
    """Return True if *text* describes a CUDA out-of-memory error.

    Requires both 'cuda' and 'out of memory' to avoid false-positives
    from CPU/RAM MemoryError messages.
    """
    lower = text.lower()
    return "out of memory" in lower and "cuda" in lower


def gpu_init(gpu_id: int) -> None:
    """Pool initializer — pins this worker process to a single GPU.

    Called once per worker at startup, before any task function runs.
    Sets ``CUDA_VISIBLE_DEVICES`` so that subsequent ``import torch``
    sees only the target GPU as ``cuda:0``.

    Also restores default SIGINT handling so that Ctrl+C kills workers
    immediately (``ProcessPoolExecutor`` with spawn context ignores
    SIGINT in children by default).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    signal.signal(signal.SIGINT, signal.SIG_DFL)


def run_single_experiment(
    experiment,
    benchmark_dir: str,
    wandb_project: str | None = None,
) -> dict | None:
    """Run one experiment on the GPU bound by ``gpu_init``.

    Returns the summary dict on success, or a structured ``{"_failed": ...}``
    payload for retryable/fatal failures.

    Dispatches to CLAM or nnMIL runner based on the experiment's framework.
    """
    # ---- Deferred imports (torch must load AFTER gpu_init set the env) ----
    import gc
    import sys

    import torch

    from autobench.pipeline.config import Framework
    from autobench.pipeline.orchestrator import (
        _isolated_torch_state,
        _load_or_collect_summary,
        load_completed,
        mark_completed,
    )

    exp_id = experiment.experiment_id

    # ---- Race-condition guard: already completed? ----
    if exp_id in load_completed(benchmark_dir):
        return _load_or_collect_summary(benchmark_dir, experiment)

    # ---- Per-experiment log file (organized: logs/{framework}/{strategy}/) ----
    log_dir = os.path.join(
        benchmark_dir, "logs",
        experiment.framework.value, experiment.strategy,
    )
    os.makedirs(log_dir, exist_ok=True)
    # Filename: {task}__{encoder}__{model}__s{seed}.log
    log_name = f"{experiment.task.name}__{experiment.encoder_key}__{experiment.model.model_type}__s{experiment.train.seed}.log"
    log_path = os.path.join(log_dir, log_name)
    log_file = open(log_path, "w")
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = log_file
    sys.stderr = log_file

    device = torch.device("cuda:0")

    def _make_failure(reason: str, detail: str = "") -> dict:
        """Return a structured failure dict (distinguishable from a success summary)."""
        return {"_failed": True, "experiment_id": exp_id, "reason": reason, "detail": detail}

    try:
        # ---- Pre-flight VRAM check ----
        free_bytes, _ = torch.cuda.mem_get_info(device)
        free_gb = free_bytes / (1024 ** 3)
        if free_gb < 1.5:
            print(
                f"[TRANSIENT_NO_VRAM] {exp_id}: only {free_gb:.1f} GB free "
                "(need >= 1.5 GB)"
            )
            return _make_failure("TRANSIENT_NO_VRAM", f"only {free_gb:.1f} GB free")

        print(f"[START] {exp_id} (free VRAM: {free_gb:.1f} GB)")

        # ---- Framework dispatch ----
        # Recycled workers (max_tasks_per_child > 1) reuse one process across
        # experiments, so isolate process-global torch state the same way the
        # single-GPU path does (see orchestrator._isolated_torch_state).
        with _isolated_torch_state():
            if experiment.framework == Framework.NNMIL:
                from autobench.pipeline.nnmil.runner import run_nnmil_experiment
                summary = run_nnmil_experiment(experiment, benchmark_dir, device=str(device))
            elif experiment.framework == Framework.DTFD:
                from autobench.pipeline.dtfd import run_dtfd_experiment
                summary = run_dtfd_experiment(experiment, benchmark_dir, device=str(device))
            elif experiment.framework == Framework.TITAN:
                from autobench.pipeline.titan import run_titan_experiment
                summary = run_titan_experiment(experiment, benchmark_dir, device=str(device))
            elif experiment.framework == Framework.ABMIL:
                from autobench.pipeline.abmil import run_abmil_experiment
                summary = run_abmil_experiment(experiment, benchmark_dir, device=str(device))
            else:
                from autobench.pipeline.clam.runner import run_experiment
                summary = run_experiment(experiment, benchmark_dir, device, wandb_project)

        mark_completed(benchmark_dir, exp_id)
        return summary

    except torch.cuda.OutOfMemoryError:
        print(f"[OOM] {exp_id}: cleaning up, will retry on next run")
        gc.collect()
        torch.cuda.empty_cache()
        return _make_failure("OOM", "CUDA out of memory")

    except Exception as exc:
        import traceback
        exc_str = str(exc)
        detail = exc_str.split("\n")[0][:200]
        if _is_cuda_oom(exc_str):
            print(f"[OOM] {exp_id}: {type(exc).__name__} (CUDA OOM): cleaning up")
            gc.collect()
            torch.cuda.empty_cache()
            return _make_failure("OOM", detail)
        print(f"[ERROR] {exp_id}: {exc}")
        traceback.print_exc()
        return _make_failure(type(exc).__name__, detail)

    finally:
        gc.collect()
        torch.cuda.empty_cache()
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        log_file.close()


MAX_WORKERS_PER_GPU = 12


def query_gpu_vram(gpu_ids: list[int]) -> dict[int, float]:
    """Query total VRAM per GPU in GiB via nvidia-smi.

    Returns a mapping ``{gpu_id: total_vram_gib}``.
    Falls back to 48.0 GiB for GPUs not found in nvidia-smi output.
    """
    fallback_gb = 48.0
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {g: fallback_gb for g in gpu_ids}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {g: fallback_gb for g in gpu_ids}

    total_by_gpu: dict[int, float] = {}
    for line in result.stdout.strip().splitlines():
        parts = line.split(",")
        if len(parts) != 2:
            continue
        idx = int(parts[0].strip())
        total_mib = float(parts[1].strip())
        total_by_gpu[idx] = total_mib / 1024  # MiB → GiB

    return {g: total_by_gpu.get(g, fallback_gb) for g in gpu_ids}


def detect_experiments_per_gpu(
    gpu_ids: list[int],
    mem_per_exp_gb: float = 10.0,
    reserve_gb: float = 4.0,
    cuda_context_gb: float = 0.5,
) -> int:
    """Auto-detect how many experiments can run concurrently per GPU.

    Queries ``nvidia-smi`` for free memory on each GPU, calculates
    ``(free - reserve) / (per_exp + cuda_context)``, and returns the
    minimum across all GPUs.  Each worker process pays ~0.5 GB for its
    CUDA context on top of the experiment memory.

    The result is capped at ``MAX_WORKERS_PER_GPU`` (12) to avoid
    excessive process overhead.  Falls back to 1 if ``nvidia-smi`` is
    unavailable.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return 1
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 1

    # Parse: each line is "index, free_MiB"
    free_by_gpu: dict[int, float] = {}
    for line in result.stdout.strip().splitlines():
        parts = line.split(",")
        if len(parts) != 2:
            continue
        idx = int(parts[0].strip())
        free_mib = float(parts[1].strip())
        free_by_gpu[idx] = free_mib / 1024  # MiB → GiB

    if not free_by_gpu:
        return 1

    cost_per_worker = mem_per_exp_gb + cuda_context_gb
    per_gpu_counts: list[int] = []
    for gpu_id in gpu_ids:
        free_gb = free_by_gpu.get(gpu_id, 0.0)
        n = int((free_gb - reserve_gb) / cost_per_worker)
        per_gpu_counts.append(max(min(n, MAX_WORKERS_PER_GPU), 1))

    return min(per_gpu_counts) if per_gpu_counts else 1
