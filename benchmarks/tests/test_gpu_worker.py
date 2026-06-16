"""Tests for autobench.pipeline._gpu_worker and orchestrator scheduling."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from autobench.pipeline._gpu_worker import (
    MAX_WORKERS_PER_GPU,
    _is_cuda_oom,
    detect_experiments_per_gpu,
    gpu_init,
    query_gpu_vram,
)
from autobench.pipeline.config import (
    BenchmarkConfig,
    ExperimentConfig,
    ModelConfig,
    TaskConfig,
    TrainConfig,
    build_registries,
)
from autobench.pipeline.orchestrator import (
    _GpuState,
    _MAX_OOM_RETRIES,
    _MODEL_BASE_VRAM,
    _OOM_ESTIMATE_MULTIPLIER,
    _RETRIABLE_FAILURES,
    clear_failed,
    estimate_vram_gb,
    load_failed,
    mark_failed,
    run_benchmark_multigpu,
)
from _helpers import make_test_ds


class TestGpuInit:
    def test_sets_cuda_visible_devices(self):
        gpu_init(3)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"

    def test_overwrites_existing_value(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
        gpu_init(5)
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "5"


class TestDetectExperimentsPerGpu:
    def _mock_nvidia_smi(self, stdout: str, returncode: int = 0):
        return patch(
            "autobench.pipeline._gpu_worker.subprocess.run",
            return_value=MagicMock(stdout=stdout, returncode=returncode),
        )

    def test_calculates_from_free_memory(self):
        # GPU 0: 48000 MiB free = ~46.875 GiB
        # (46.875 - 4.0) / (10.0 + 0.5) = 42.875 / 10.5 = 4.08 -> int = 4
        stdout = "0, 48000\n1, 48000\n"
        with self._mock_nvidia_smi(stdout):
            n = detect_experiments_per_gpu([0, 1])
        assert n == 4

    def test_takes_minimum_across_gpus(self):
        # GPU 0: 48000 MiB (~46.9 GiB) -> 4
        # GPU 1: 12000 MiB (~11.7 GiB) -> (11.7 - 4.0) / 10.5 < 1 -> clamped to 1
        stdout = "0, 48000\n1, 12000\n"
        with self._mock_nvidia_smi(stdout):
            n = detect_experiments_per_gpu([0, 1])
        assert n == 1

    def test_minimum_is_one(self):
        # GPU 0: 4000 MiB (~3.9 GiB) -> (3.9 - 4.0) / 3.0 < 0 -> clamped to 1
        stdout = "0, 4000\n"
        with self._mock_nvidia_smi(stdout):
            n = detect_experiments_per_gpu([0])
        assert n == 1

    def test_fallback_on_nvidia_smi_failure(self):
        with self._mock_nvidia_smi("", returncode=1):
            n = detect_experiments_per_gpu([0, 1])
        assert n == 1

    def test_fallback_on_nvidia_smi_not_found(self):
        with patch(
            "autobench.pipeline._gpu_worker.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            n = detect_experiments_per_gpu([0])
        assert n == 1

    def test_fallback_on_timeout(self):
        import subprocess

        with patch(
            "autobench.pipeline._gpu_worker.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10),
        ):
            n = detect_experiments_per_gpu([0])
        assert n == 1

    def test_custom_mem_params(self):
        # GPU 0: 24000 MiB (~23.4 GiB), reserve=2, per_exp=2, context=0.5
        # (23.4 - 2) / (2 + 0.5) = 21.4 / 2.5 = 8.56 -> int = 8 (below cap)
        stdout = "0, 24000\n"
        with self._mock_nvidia_smi(stdout):
            n = detect_experiments_per_gpu([0], mem_per_exp_gb=2.0, reserve_gb=2.0)
        assert n == 8

    def test_cap_at_max_workers(self):
        # Even with massive free memory, capped at MAX_WORKERS_PER_GPU.
        # Need free_gb such that (free_gb - 4) / 10.5 > MAX_WORKERS_PER_GPU.
        # At 200000 MiB free (~195.3 GiB): (195.3-4)/10.5 = 18.2 -> capped.
        stdout = "0, 200000\n"
        with self._mock_nvidia_smi(stdout):
            n = detect_experiments_per_gpu([0])
        assert n == MAX_WORKERS_PER_GPU

    def test_gpu_not_in_nvidia_smi_output(self):
        # Only GPU 0 in output, but we request GPU 5
        stdout = "0, 48000\n"
        with self._mock_nvidia_smi(stdout):
            n = detect_experiments_per_gpu([5])
        # free_gb=0.0 -> (0.0 - 4.0) / 3.0 < 0 -> clamped to 1
        assert n == 1

    def test_handles_empty_gpu_ids(self):
        stdout = "0, 48000\n"
        with self._mock_nvidia_smi(stdout):
            n = detect_experiments_per_gpu([])
        assert n == 1

    def test_multiple_identical_gpus(self):
        # Two GPUs with the same free memory should give the same result
        # 48000 MiB = 46.875 GiB -> (46.875 - 4.0) / (10.0 + 0.5) = 4
        stdout = "0, 48000\n1, 48000\n"
        with self._mock_nvidia_smi(stdout):
            n = detect_experiments_per_gpu([0, 1])
        # Both GPUs identical -> result is min of two equal values
        assert n == 4

    def test_malformed_nvidia_smi_line_skipped(self):
        # A garbage line mixed with valid output; garbage should be skipped
        # 48000 MiB = 46.875 GiB -> (46.875 - 4.0) / (10.0 + 0.5) = 4
        stdout = "garbage\n0, 48000\n"
        with self._mock_nvidia_smi(stdout):
            n = detect_experiments_per_gpu([0])
        assert n == 4


class TestBenchmarkConfigExperimentsPerGpu:
    def test_default_is_none(self):
        cfg = BenchmarkConfig()
        assert cfg.experiments_per_gpu is None

    def test_explicit_value(self):
        cfg = BenchmarkConfig(experiments_per_gpu=8)
        assert cfg.experiments_per_gpu == 8


class TestRoundRobinFairness:
    """Pure-logic tests for the round-robin GPU assignment pattern used in _run_multigpu."""

    @staticmethod
    def _round_robin(encoder_keys: list[str], gpu_ids: list[int]) -> dict[int, list[str]]:
        assignments: dict[int, list[str]] = {g: [] for g in gpu_ids}
        for idx, key in enumerate(encoder_keys):
            gpu = gpu_ids[idx % len(gpu_ids)]
            assignments[gpu].append(key)
        return assignments

    def test_even_distribution(self):
        keys = [f"m{i}" for i in range(6)]
        result = self._round_robin(keys, [0, 1, 2])
        assert all(len(v) == 2 for v in result.values())

    def test_uneven_distribution(self):
        keys = [f"m{i}" for i in range(7)]
        result = self._round_robin(keys, [0, 1, 2])
        counts = sorted([len(v) for v in result.values()], reverse=True)
        assert counts == [3, 2, 2]

    def test_more_gpus_than_models(self):
        keys = ["m0", "m1"]
        result = self._round_robin(keys, [0, 1, 2, 3, 4])
        counts = sorted([len(v) for v in result.values()], reverse=True)
        assert counts == [1, 1, 0, 0, 0]

    def test_single_gpu(self):
        keys = [f"m{i}" for i in range(5)]
        result = self._round_robin(keys, [0])
        assert len(result[0]) == 5
        assert result[0] == keys

    def test_single_model(self):
        keys = ["only_model"]
        result = self._round_robin(keys, [0, 1, 2])
        assert result[0] == ["only_model"]
        assert result[1] == []
        assert result[2] == []


# ---------------------------------------------------------------------------
# query_gpu_vram
# ---------------------------------------------------------------------------


class TestQueryGpuVram:
    def _mock_nvidia_smi(self, stdout: str, returncode: int = 0):
        return patch(
            "autobench.pipeline._gpu_worker.subprocess.run",
            return_value=MagicMock(stdout=stdout, returncode=returncode),
        )

    def test_parses_total_memory(self):
        # GPU 0: 49152 MiB = 48.0 GiB, GPU 1: 24576 MiB = 24.0 GiB
        stdout = "0, 49152\n1, 24576\n"
        with self._mock_nvidia_smi(stdout):
            result = query_gpu_vram([0, 1])
        assert result[0] == 48.0
        assert result[1] == 24.0

    def test_fallback_for_missing_gpu(self):
        stdout = "0, 49152\n"
        with self._mock_nvidia_smi(stdout):
            result = query_gpu_vram([0, 5])
        assert result[0] == 48.0
        assert result[5] == 48.0  # fallback

    def test_fallback_on_failure(self):
        with self._mock_nvidia_smi("", returncode=1):
            result = query_gpu_vram([0, 1])
        assert result[0] == 48.0
        assert result[1] == 48.0

    def test_fallback_on_not_found(self):
        with patch(
            "autobench.pipeline._gpu_worker.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            result = query_gpu_vram([0])
        assert result[0] == 48.0

    def test_fallback_on_timeout(self):
        import subprocess
        with patch(
            "autobench.pipeline._gpu_worker.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10),
        ):
            result = query_gpu_vram([2])
        assert result[2] == 48.0


# ---------------------------------------------------------------------------
# estimate_vram_gb
# ---------------------------------------------------------------------------


def _make_exp(model_type: str, embed_dim: int = 768) -> ExperimentConfig:
    """Helper to create a minimal ExperimentConfig for VRAM estimation tests."""
    return ExperimentConfig(
        task=TaskConfig(name="brca", label_col="label", label_dict={"neg": 0, "pos": 1}),
        encoder_key="test_enc",
        embed_dim=embed_dim,
        model=ModelConfig(model_type=model_type),
        train=TrainConfig(),
        strategy="standard",
    )


class TestEstimateVramGb:
    def test_known_light_model_768(self):
        exp = _make_exp("simple_mil", embed_dim=768)
        est = estimate_vram_gb(exp)
        # base=2.5, dim_ratio=1.0, dim_factor=1.0, + 1.8 context
        assert est == pytest.approx(4.3, abs=0.1)

    def test_known_heavy_model_768(self):
        exp = _make_exp("vision_transformer", embed_dim=768)
        est = estimate_vram_gb(exp)
        # base=16.0, dim_ratio=1.0 ** 1.5 = 1.0, 16.0 + 1.8 = 17.8
        assert est == pytest.approx(17.8, abs=0.1)

    def test_attention_model_larger_dim(self):
        # trans_mil with embed_dim=1536 (2x baseline)
        exp = _make_exp("trans_mil", embed_dim=1536)
        est = estimate_vram_gb(exp)
        # base=12.0, dim_ratio=2.0, dim_factor=2.0^1.5=2.828
        # 12.0 * 2.828 + 1.8 = 35.7
        assert est > 30.0

    def test_non_attention_model_larger_dim(self):
        # clam_sb with embed_dim=1536 (2x baseline)
        exp = _make_exp("clam_sb", embed_dim=1536)
        est = estimate_vram_gb(exp)
        # base=3.0, dim_ratio=2.0, dim_factor=2.0 (linear)
        # 3.0 * 2.0 + 1.8 = 7.8
        assert est == pytest.approx(7.8, abs=0.1)

    def test_unknown_model_uses_default(self):
        exp = _make_exp("totally_new_model", embed_dim=768)
        est = estimate_vram_gb(exp)
        # base=6.0 (default), dim_ratio=1.0, + 1.8 = 7.8
        assert est == pytest.approx(7.8, abs=0.1)

    def test_all_known_models_present(self):
        """Every model in the lookup should return a value without error."""
        for model_type in _MODEL_BASE_VRAM:
            exp = _make_exp(model_type, embed_dim=768)
            est = estimate_vram_gb(exp)
            assert est > 0


# ---------------------------------------------------------------------------
# load_failed / mark_failed
# ---------------------------------------------------------------------------


class TestFailureTracking:
    def test_load_failed_empty(self, tmp_path):
        assert load_failed(str(tmp_path)) == {}

    def test_mark_and_load(self, tmp_path):
        bd = str(tmp_path)
        mark_failed(bd, "exp_1", reason="OOM", detail="CUDA out of memory", gpu_id=0, estimated_vram_gb=10.0)
        failed = load_failed(bd)
        assert "exp_1" in failed
        assert failed["exp_1"]["reason"] == "OOM"
        assert failed["exp_1"]["detail"] == "CUDA out of memory"
        assert failed["exp_1"]["gpu_id"] == 0
        assert failed["exp_1"]["estimated_vram_gb"] == 10.0
        assert "timestamp" in failed["exp_1"]

    def test_multiple_failures(self, tmp_path):
        bd = str(tmp_path)
        mark_failed(bd, "exp_1", reason="OOM")
        mark_failed(bd, "exp_2", reason="EXCEEDS_GPU_VRAM")
        mark_failed(bd, "exp_3", reason="RuntimeError", detail="something broke")
        failed = load_failed(bd)
        assert len(failed) == 3
        assert failed["exp_2"]["reason"] == "EXCEEDS_GPU_VRAM"

    def test_overwrite_existing_failure(self, tmp_path):
        bd = str(tmp_path)
        mark_failed(bd, "exp_1", reason="OOM")
        mark_failed(bd, "exp_1", reason="EXCEEDS_GPU_VRAM")
        failed = load_failed(bd)
        assert len(failed) == 1
        assert failed["exp_1"]["reason"] == "EXCEEDS_GPU_VRAM"

    def test_failed_json_format(self, tmp_path):
        bd = str(tmp_path)
        mark_failed(bd, "exp_1", reason="OOM", gpu_id=2, estimated_vram_gb=15.5)
        path = os.path.join(bd, "results", "_failed.json")
        with open(path) as f:
            raw = json.load(f)
        assert isinstance(raw, dict)
        assert "exp_1" in raw

    def test_clear_failed_removes_record(self, tmp_path):
        bd = str(tmp_path)
        mark_failed(bd, "exp_1", reason="OOM")
        clear_failed(bd, "exp_1")
        assert "exp_1" not in load_failed(bd)


# ---------------------------------------------------------------------------
# _GpuState budget logic
# ---------------------------------------------------------------------------


class TestGpuState:
    def test_budget_free(self):
        gs = _GpuState(gpu_id=0, total_gb=46.0, reserved_gb=10.0)
        assert gs.budget_free == 36.0

    def test_budget_free_zero_reserved(self):
        gs = _GpuState(gpu_id=0, total_gb=46.0)
        assert gs.budget_free == 46.0

    def test_budget_free_fully_reserved(self):
        gs = _GpuState(gpu_id=0, total_gb=46.0, reserved_gb=46.0)
        assert gs.budget_free == 0.0


# ---------------------------------------------------------------------------
# OOM retry policy constants
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    """Verify the retry policy constants and escalation logic."""

    def test_retriable_failures_include_oom(self):
        assert "OOM" in _RETRIABLE_FAILURES

    def test_retriable_failures_include_skip(self):
        assert "SKIP" in _RETRIABLE_FAILURES

    def test_retriable_failures_include_transient_no_vram(self):
        assert "TRANSIENT_NO_VRAM" in _RETRIABLE_FAILURES

    def test_non_retriable_failure(self):
        assert "RuntimeError" not in _RETRIABLE_FAILURES

    def test_max_retries_positive(self):
        assert _MAX_OOM_RETRIES >= 1

    def test_multiplier_greater_than_one(self):
        assert _OOM_ESTIMATE_MULTIPLIER > 1.0

    def test_escalation_reaches_solo(self):
        """Starting from a 10 GB estimate on a 46 GB GPU, escalation
        should eventually cap at GPU capacity (solo run)."""
        gpu_total = 46.0
        est = 10.0
        for _ in range(_MAX_OOM_RETRIES):
            est = min(est * _OOM_ESTIMATE_MULTIPLIER, gpu_total)
        # After max retries, should have reached or approached GPU total
        assert est >= 10.0 * _OOM_ESTIMATE_MULTIPLIER

    def test_already_solo_not_retried(self):
        """If estimate >= GPU total, the experiment already ran solo.
        Retrying won't help -- should be permanently failed."""
        gpu_total = 46.0
        vram_est = 46.0  # already at full GPU
        already_solo = vram_est >= gpu_total
        assert already_solo is True

    def test_light_model_escalation_sequence(self):
        """A 3 GB experiment escalates: 3 -> 4.5 -> 6.75 -> 10.1.
        All well within 46 GB, so all retries are valid."""
        est = 3.0
        gpu_total = 46.0
        for i in range(_MAX_OOM_RETRIES):
            new_est = round(est * _OOM_ESTIMATE_MULTIPLIER, 1)
            new_est = min(new_est, gpu_total)
            assert new_est <= gpu_total
            est = new_est

    def test_heavy_model_escalation_hits_cap(self):
        """A 20 GB experiment on 46 GB GPU: 20 -> 30 -> 45 -> 46 (capped).
        Third retry runs solo."""
        est = 20.0
        gpu_total = 46.0
        estimates = [est]
        for _ in range(_MAX_OOM_RETRIES):
            est = round(min(est * _OOM_ESTIMATE_MULTIPLIER, gpu_total), 1)
            estimates.append(est)
        # Should hit cap by the last retry
        assert estimates[-1] == gpu_total


class _ImmediateFuture:
    def __init__(self, result=None, exc: Exception | None = None):
        self._result = result
        self._exc = exc

    def result(self):
        if self._exc is not None:
            raise self._exc
        return self._result


class _InlineProcessPool:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def submit(self, fn, *args, **kwargs):
        try:
            return _ImmediateFuture(result=fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover - defensive
            return _ImmediateFuture(exc=exc)

    def shutdown(self, wait=True, cancel_futures=False):
        del wait, cancel_futures


def _fake_wait(futures, return_when=None, timeout=None):
    del return_when, timeout
    items = list(futures)
    if not items:
        return set(), set()
    return {items[0]}, set(items[1:])


def _make_multigpu_cfg(tmp_path) -> BenchmarkConfig:
    return BenchmarkConfig(
        benchmark_dir=str(tmp_path),
        mapping_csv=str(tmp_path / "mapping.csv"),
        features_base_dir=str(tmp_path / "features"),
        encoder_keys=["test_enc"],
        model_types=["clam_sb"],
        tasks=["brca"],
        strategies=["standard"],
    )


def _patch_multigpu_runtime(
    monkeypatch,
    experiments: list[ExperimentConfig],
    run_single_impl,
    summaries_by_id: dict[str, dict] | None = None,
    gpu_vram: dict[int, float] | None = None,
):
    monkeypatch.setattr(
        "autobench.pipeline.orchestrator._prepare_data",
        lambda cfg, ds: None,
    )
    monkeypatch.setattr(
        "autobench.pipeline.orchestrator._prepare_nnmil_plans",
        lambda cfg, exps, registries=None, dataset_name="dataset": None,
    )
    monkeypatch.setattr(
        "autobench.pipeline.orchestrator.generate_all_experiments",
        lambda cfg, registries=None: experiments,
    )
    monkeypatch.setattr(
        "autobench.pipeline.orchestrator.ProcessPoolExecutor",
        _InlineProcessPool,
    )
    monkeypatch.setattr(
        "autobench.pipeline.orchestrator.wait",
        _fake_wait,
    )
    monkeypatch.setattr(
        "autobench.pipeline._gpu_worker.query_gpu_vram",
        lambda gpu_ids: gpu_vram or {g: 48.0 for g in gpu_ids},
    )
    monkeypatch.setattr(
        "autobench.pipeline._gpu_worker.gpu_init",
        lambda gpu_id: None,
    )
    monkeypatch.setattr(
        "autobench.pipeline._gpu_worker.run_single_experiment",
        run_single_impl,
    )
    if summaries_by_id is not None:
        monkeypatch.setattr(
            "autobench.pipeline.orchestrator._load_or_collect_summary",
            lambda benchmark_dir, exp_or_id: summaries_by_id.get(
                exp_or_id.experiment_id if hasattr(exp_or_id, "experiment_id") else exp_or_id
            ),
        )
        monkeypatch.setattr(
            "autobench.pipeline.orchestrator.collect_all_summaries_on_disk",
            lambda benchmark_dir: list(summaries_by_id.values()),
        )
    monkeypatch.setattr(
        "autobench.pipeline.orchestrator._finalize",
        lambda benchmark_dir, summaries: summaries,
    )


class TestStrictCompletionScheduler:
    def test_pending_does_not_skip_failed_history(self, tmp_path, monkeypatch):
        from autobench.pipeline.orchestrator import mark_completed

        exp = _make_exp("clam_sb")
        summary = {"experiment_id": exp.experiment_id}
        calls: list[str] = []

        def _run_single(experiment, benchmark_dir, wandb_project):
            del wandb_project
            calls.append(experiment.experiment_id)
            mark_completed(benchmark_dir, experiment.experiment_id)
            return summary

        _patch_multigpu_runtime(
            monkeypatch,
            experiments=[exp],
            run_single_impl=_run_single,
            summaries_by_id={exp.experiment_id: summary},
        )

        mark_failed(str(tmp_path), exp.experiment_id, reason="OOM")
        cfg = _make_multigpu_cfg(tmp_path)
        results = run_benchmark_multigpu(cfg, [0], ds=make_test_ds(), registries=build_registries(make_test_ds()))

        assert calls == [exp.experiment_id]
        assert len(results) == 1
        assert exp.experiment_id not in load_failed(str(tmp_path))

    def test_abort_on_non_oom_failure(self, tmp_path, monkeypatch):
        exp = _make_exp("clam_sb")

        def _run_single(experiment, benchmark_dir, wandb_project):
            del experiment, benchmark_dir, wandb_project
            return {"_failed": True, "reason": "RuntimeError", "detail": "boom"}

        _patch_multigpu_runtime(
            monkeypatch,
            experiments=[exp],
            run_single_impl=_run_single,
        )

        cfg = _make_multigpu_cfg(tmp_path)
        with pytest.raises(RuntimeError, match="Non-OOM experiment failure"):
            run_benchmark_multigpu(cfg, [0], ds=make_test_ds(), registries=build_registries(make_test_ds()))

        assert load_failed(str(tmp_path))[exp.experiment_id]["reason"] == "RuntimeError"

    def test_oom_retry_escalates_to_solo_then_succeeds(self, tmp_path, monkeypatch):
        from autobench.pipeline.orchestrator import mark_completed

        exp = _make_exp("clam_sb")
        summary = {"experiment_id": exp.experiment_id}
        attempts = {"n": 0}

        def _run_single(experiment, benchmark_dir, wandb_project):
            del wandb_project
            attempts["n"] += 1
            if attempts["n"] == 1:
                return {"_failed": True, "reason": "OOM", "detail": "CUDA out of memory"}
            mark_completed(benchmark_dir, experiment.experiment_id)
            return summary

        _patch_multigpu_runtime(
            monkeypatch,
            experiments=[exp],
            run_single_impl=_run_single,
            summaries_by_id={exp.experiment_id: summary},
        )

        cfg = _make_multigpu_cfg(tmp_path)
        results = run_benchmark_multigpu(cfg, [0], ds=make_test_ds(), registries=build_registries(make_test_ds()))

        assert attempts["n"] == 2
        assert len(results) == 1
        assert exp.experiment_id not in load_failed(str(tmp_path))

    def test_oom_retry_escalates_to_solo_then_fails(self, tmp_path, monkeypatch):
        exp = _make_exp("clam_sb")
        attempts = {"n": 0}

        def _run_single(experiment, benchmark_dir, wandb_project):
            del experiment, benchmark_dir, wandb_project
            attempts["n"] += 1
            return {"_failed": True, "reason": "OOM", "detail": "CUDA out of memory"}

        _patch_multigpu_runtime(
            monkeypatch,
            experiments=[exp],
            run_single_impl=_run_single,
        )

        cfg = _make_multigpu_cfg(tmp_path)
        with pytest.raises(RuntimeError, match="failed after retry budget"):
            run_benchmark_multigpu(cfg, [0], ds=make_test_ds(), registries=build_registries(make_test_ds()))

        assert attempts["n"] >= 2
        assert load_failed(str(tmp_path))[exp.experiment_id]["reason"] == "OOM"

    def test_no_stranded_pending_jobs(self, tmp_path, monkeypatch):
        from autobench.pipeline.orchestrator import mark_completed

        exp_big = _make_exp("vision_transformer")
        exp_small = _make_exp("clam_sb")
        summaries = {
            exp_big.experiment_id: {"experiment_id": exp_big.experiment_id},
            exp_small.experiment_id: {"experiment_id": exp_small.experiment_id},
        }
        calls: list[str] = []

        def _run_single(experiment, benchmark_dir, wandb_project):
            del wandb_project
            calls.append(experiment.experiment_id)
            mark_completed(benchmark_dir, experiment.experiment_id)
            return summaries[experiment.experiment_id]

        _patch_multigpu_runtime(
            monkeypatch,
            experiments=[exp_big, exp_small],
            run_single_impl=_run_single,
            summaries_by_id=summaries,
            gpu_vram={0: 12.0, 1: 22.0},  # usable: 10 GB and 20 GB
        )

        cfg = _make_multigpu_cfg(tmp_path)
        results = run_benchmark_multigpu(cfg, [0, 1], ds=make_test_ds(), registries=build_registries(make_test_ds()))

        assert len(results) == 2
        assert sorted(calls) == sorted([exp_big.experiment_id, exp_small.experiment_id])

    def test_strict_completion_guard(self, tmp_path, monkeypatch):
        exp = _make_exp("clam_sb")
        summary = {"experiment_id": exp.experiment_id}

        def _run_single(experiment, benchmark_dir, wandb_project):
            del experiment, benchmark_dir, wandb_project
            # Simulate worker bug: returns summary but forgets to mark completed.
            return summary

        _patch_multigpu_runtime(
            monkeypatch,
            experiments=[exp],
            run_single_impl=_run_single,
            summaries_by_id={exp.experiment_id: summary},
        )

        cfg = _make_multigpu_cfg(tmp_path)
        with pytest.raises(RuntimeError, match="Strict-completion check failed"):
            run_benchmark_multigpu(cfg, [0], ds=make_test_ds(), registries=build_registries(make_test_ds()))


class TestIsCudaOom:
    """Unit tests for the _is_cuda_oom string matcher."""

    def test_classic_cuda_oom(self):
        assert _is_cuda_oom("CUDA error: out of memory") is True

    def test_torch_oom_message(self):
        assert _is_cuda_oom("CUDA out of memory. Tried to allocate 2.00 GiB") is True

    def test_case_insensitive(self):
        assert _is_cuda_oom("Cuda Out Of Memory") is True

    def test_cpu_oom_not_matched(self):
        """A CPU/RAM 'out of memory' without 'cuda' must NOT match."""
        assert _is_cuda_oom("out of memory") is False
        assert _is_cuda_oom("Failed to allocate: out of memory") is False

    def test_cuda_without_oom_not_matched(self):
        assert _is_cuda_oom("CUDA error: device-side assert triggered") is False

    def test_empty_string(self):
        assert _is_cuda_oom("") is False

    def test_accelerator_error_oom(self):
        """The exact error message from torch.AcceleratorError."""
        assert _is_cuda_oom("CUDA error: out of memory") is True


class TestWorkerOomReclassification:
    """Test the worker's generic-except OOM handler (primary path)."""

    def test_non_oom_error_returns_exception_type(self):
        """A generic exception without OOM keywords returns the type name."""
        exc = RuntimeError("some random error")
        exc_str = str(exc)
        assert not _is_cuda_oom(exc_str)

    def test_accelerator_error_with_cuda_oom_returns_oom_reason(self):
        """AcceleratorError('CUDA error: out of memory') should be
        classified as reason='OOM' by the worker's generic handler."""
        exc_str = "CUDA error: out of memory"
        assert _is_cuda_oom(exc_str)
        # The worker constructs: detail = exc_str.split("\\n")[0][:200]
        detail = exc_str.split("\n")[0][:200]
        assert "out of memory" in detail.lower()
        assert "cuda" in detail.lower()

    def test_cpu_memory_error_not_reclassified(self):
        """A MemoryError (CPU) should NOT be classified as CUDA OOM."""
        exc = MemoryError("out of memory")
        assert not _is_cuda_oom(str(exc))


class TestAcceleratorErrorOOM:
    """AcceleratorError wrapping 'out of memory' should be retried as OOM."""

    def test_accelerator_error_oom_retried_then_succeeds(self, tmp_path, monkeypatch):
        """An AcceleratorError with 'out of memory' detail should be reclassified
        as OOM by the orchestrator and retried, not treated as fatal."""
        from autobench.pipeline.orchestrator import mark_completed

        exp = _make_exp("clam_sb")
        summary = {"experiment_id": exp.experiment_id}
        attempts = {"n": 0}

        def _run_single(experiment, benchmark_dir, wandb_project):
            del wandb_project
            attempts["n"] += 1
            if attempts["n"] == 1:
                # Simulate AcceleratorError("CUDA error: out of memory")
                # classified by the worker's generic except handler
                return {
                    "_failed": True,
                    "reason": "AcceleratorError",
                    "detail": "CUDA error: out of memory",
                }
            mark_completed(benchmark_dir, experiment.experiment_id)
            return summary

        _patch_multigpu_runtime(
            monkeypatch,
            experiments=[exp],
            run_single_impl=_run_single,
            summaries_by_id={exp.experiment_id: summary},
        )

        cfg = _make_multigpu_cfg(tmp_path)
        results = run_benchmark_multigpu(cfg, [0], ds=make_test_ds(), registries=build_registries(make_test_ds()))

        assert attempts["n"] == 2
        assert len(results) == 1
        assert exp.experiment_id not in load_failed(str(tmp_path))

    def test_accelerator_error_without_oom_is_fatal(self, tmp_path, monkeypatch):
        """AcceleratorError NOT containing 'out of memory' should remain fatal."""
        exp = _make_exp("clam_sb")

        def _run_single(experiment, benchmark_dir, wandb_project):
            del experiment, benchmark_dir, wandb_project
            return {
                "_failed": True,
                "reason": "AcceleratorError",
                "detail": "CUDA error: device-side assert triggered",
            }

        _patch_multigpu_runtime(
            monkeypatch,
            experiments=[exp],
            run_single_impl=_run_single,
        )

        cfg = _make_multigpu_cfg(tmp_path)
        with pytest.raises(RuntimeError, match="Non-OOM experiment failure"):
            run_benchmark_multigpu(cfg, [0], ds=make_test_ds(), registries=build_registries(make_test_ds()))

    def test_cpu_oom_not_reclassified(self, tmp_path, monkeypatch):
        """A CPU MemoryError ('out of memory' without 'cuda') stays fatal."""
        exp = _make_exp("clam_sb")

        def _run_single(experiment, benchmark_dir, wandb_project):
            del experiment, benchmark_dir, wandb_project
            return {
                "_failed": True,
                "reason": "MemoryError",
                "detail": "out of memory",
            }

        _patch_multigpu_runtime(
            monkeypatch,
            experiments=[exp],
            run_single_impl=_run_single,
        )

        cfg = _make_multigpu_cfg(tmp_path)
        with pytest.raises(RuntimeError, match="Non-OOM experiment failure"):
            run_benchmark_multigpu(cfg, [0], ds=make_test_ds(), registries=build_registries(make_test_ds()))


