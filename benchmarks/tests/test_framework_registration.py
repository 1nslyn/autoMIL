"""Tests for DTFD + TITAN framework registration (design spec §4, §6-8).

Covers the infrastructure that lets the two new arms participate in the grid and
the multi-GPU dispatch, while their trainers are still stubs (NotImplementedError).
"""

import pytest
import torch

from autobench.pipeline.config import (
    BenchmarkConfig,
    ExperimentConfig,
    Framework,
    ModelConfig,
    TrainConfig,
    build_registries,
    generate_all_experiments,
)
from autobench.pipeline.orchestrator import (
    _MODEL_BASE_VRAM,
    _prepare_titan_plans,
    _run_single_experiment_dispatch,
)
from autobench.pipeline.dtfd import run_dtfd_experiment
from autobench.pipeline.titan import run_titan_experiment
from _helpers import make_test_ds


@pytest.fixture
def ds():
    return make_test_ds()


@pytest.fixture
def registries(ds):
    return build_registries(ds)


# ---------------------------------------------------------------------------
# Framework enum
# ---------------------------------------------------------------------------

class TestFrameworkEnum:
    def test_dtfd_and_titan_registered(self):
        assert Framework.DTFD.value == "dtfd"
        assert Framework.TITAN.value == "titan"

    def test_exactly_four_frameworks(self):
        assert {f.value for f in Framework} == {"clam", "nnmil", "dtfd", "titan"}


# ---------------------------------------------------------------------------
# DTFD grid generation (sweeps encoders + dtfd_model_types, like nnMIL)
# ---------------------------------------------------------------------------

class TestDtfdGrid:
    def _cfg(self, ds, **kw):
        params = dict(
            frameworks=[Framework.DTFD], strategies=["standard"],
            tasks=["brca"], encoder_keys=["conch_v15"],
        )
        params.update(kw)
        return BenchmarkConfig.from_dataset_config(ds, **params)

    def test_uses_dtfd_models(self, ds, registries):
        # make_test_ds sets dtfd_models=["dtfd_mil"] -> 1 task x 1 enc x 1 model.
        exps = generate_all_experiments(self._cfg(ds), registries)
        assert len(exps) == 1
        assert exps[0].framework == Framework.DTFD
        assert exps[0].model.model_type == "dtfd_mil"
        assert exps[0].encoder_key == "conch_v15"

    def test_experiment_id_format(self, ds, registries):
        exps = generate_all_experiments(self._cfg(ds), registries)
        assert exps[0].experiment_id == "dtfd__standard__brca__conch_v15__dtfd_mil__s42"

    def test_results_subdir(self, ds, registries):
        exps = generate_all_experiments(self._cfg(ds), registries)
        assert exps[0].results_subdir == "dtfd/standard/brca/conch_v15/dtfd_mil"

    def test_sweeps_encoders(self, ds, registries):
        cfg = self._cfg(ds, encoder_keys=["conch_v15", "uni_v2"])
        exps = generate_all_experiments(cfg, registries)
        assert len(exps) == 2  # 2 encoders x 1 dtfd model
        assert {e.encoder_key for e in exps} == {"conch_v15", "uni_v2"}


# ---------------------------------------------------------------------------
# TITAN grid generation (single arm per task; TITAN *is* the encoder)
# ---------------------------------------------------------------------------

class TestTitanGrid:
    def test_single_arm_per_task_no_sweep(self, ds, registries):
        cfg = BenchmarkConfig.from_dataset_config(
            ds, frameworks=[Framework.TITAN], strategies=["standard"],
        )
        exps = generate_all_experiments(cfg, registries)
        # One per task, NOT multiplied by the 7 encoders or any model list.
        assert len(exps) == len(cfg.tasks)
        for e in exps:
            assert e.framework == Framework.TITAN
            assert e.encoder_key == "titan"
            assert e.model.model_type == "titan"

    def test_embed_dim_is_titan_default_768(self, ds, registries):
        cfg = BenchmarkConfig.from_dataset_config(
            ds, frameworks=[Framework.TITAN], strategies=["standard"], tasks=["brca"],
        )
        exps = generate_all_experiments(cfg, registries)
        assert exps[0].embed_dim == 768

    def test_experiment_id_format(self, ds, registries):
        cfg = BenchmarkConfig.from_dataset_config(
            ds, frameworks=[Framework.TITAN], strategies=["standard"], tasks=["brca"],
        )
        exps = generate_all_experiments(cfg, registries)
        assert exps[0].experiment_id == "titan__standard__brca__titan__titan__s42"


# ---------------------------------------------------------------------------
# Mixed grid + unknown-framework guard
# ---------------------------------------------------------------------------

class TestMultiFramework:
    def test_all_four_frameworks_in_one_grid(self, ds, registries):
        cfg = BenchmarkConfig.from_dataset_config(
            ds,
            frameworks=[Framework.CLAM, Framework.NNMIL, Framework.DTFD, Framework.TITAN],
            strategies=["standard"], tasks=["brca"], encoder_keys=["conch_v15"],
        )
        exps = generate_all_experiments(cfg, registries)

        def n(fw):
            return len([e for e in exps if e.framework == fw])

        assert n(Framework.CLAM) == 3    # 3 CLAM heads
        assert n(Framework.NNMIL) == 9   # 9 nnMIL heads
        assert n(Framework.DTFD) == 1    # 1 dtfd head
        assert n(Framework.TITAN) == 1   # single arm

    def test_unknown_framework_raises(self, ds, registries):
        cfg = BenchmarkConfig.from_dataset_config(
            ds, strategies=["standard"], tasks=["brca"], encoder_keys=["conch_v15"],
            frameworks=["bogus_framework"],  # not a Framework member -> hits the guard
        )
        with pytest.raises(ValueError, match="Unknown framework"):
            generate_all_experiments(cfg, registries)


# ---------------------------------------------------------------------------
# Dispatch routing (stubs raise; proves the wiring reaches them)
# ---------------------------------------------------------------------------

class TestDispatchRouting:
    def _exp(self, registries, framework):
        is_titan = framework == Framework.TITAN
        return ExperimentConfig(
            task=registries.task_registry["brca"],
            encoder_key="titan" if is_titan else "conch_v15",
            embed_dim=768,
            model=ModelConfig(model_type="titan" if is_titan else "dtfd_mil"),
            train=TrainConfig(seed=42),
            framework=framework,
            strategy="standard",
        )

    def test_stub_runners_raise_not_implemented(self):
        with pytest.raises(NotImplementedError):
            run_dtfd_experiment(None, "/tmp")
        with pytest.raises(NotImplementedError):
            run_titan_experiment(None, "/tmp")

    def test_dispatch_routes_dtfd_to_stub(self, registries, tmp_path):
        exp = self._exp(registries, Framework.DTFD)
        with pytest.raises(NotImplementedError):
            _run_single_experiment_dispatch(exp, str(tmp_path), torch.device("cpu"))

    def test_dispatch_routes_titan_to_stub(self, registries, tmp_path):
        exp = self._exp(registries, Framework.TITAN)
        with pytest.raises(NotImplementedError):
            _run_single_experiment_dispatch(exp, str(tmp_path), torch.device("cpu"))


# ---------------------------------------------------------------------------
# VRAM table + TITAN prepare stub
# ---------------------------------------------------------------------------

class TestVramAndPrepStubs:
    def test_titan_vram_registered(self):
        assert _MODEL_BASE_VRAM["titan"] == 2.0

    def test_dtfd_vram_present(self):
        assert "dtfd_mil" in _MODEL_BASE_VRAM

    def test_titan_prepare_is_failfast_stub(self):
        with pytest.raises(NotImplementedError):
            _prepare_titan_plans(None, [])
