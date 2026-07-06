"""Tests for autobench.pipeline.config module (dynamic registries)."""

import json

import pytest

from autobench.pipeline.config import (
    BenchmarkConfig,
    ExperimentConfig,
    Framework,
    ModelConfig,
    Registries,
    StrategyConfig,
    TaskConfig,
    TrainConfig,
    build_registries,
    generate_all_experiments,
    get_nnmil_runtime_overrides,
    NNMIL_MODEL_RUNTIME_OVERRIDES,
    NNMIL_RUNTIME_DEFAULTS,
)
from _helpers import make_test_ds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def ds():
    return make_test_ds()


@pytest.fixture
def registries(ds):
    return build_registries(ds)


# ---------------------------------------------------------------------------
# TaskConfig
# ---------------------------------------------------------------------------

class TestTaskConfig:
    def test_brca_registered(self, registries):
        assert "brca" in registries.task_registry

    def test_hrd_registered(self, registries):
        assert "hrd" in registries.task_registry

    def test_label_dict_maps_string_to_int(self, registries):
        for task in registries.task_registry.values():
            for k, v in task.label_dict.items():
                assert isinstance(k, str)
                assert isinstance(v, int)

    def test_binary_classification(self, registries):
        for task in registries.task_registry.values():
            assert task.n_classes == 2


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------

class TestModelConfig:
    def test_three_models_registered(self, registries):
        assert set(registries.model_registry.keys()) == {"clam_sb", "clam_mb", "mil"}

    def test_model_types_match_keys(self, registries):
        for key, cfg in registries.model_registry.items():
            assert cfg.model_type == key

    def test_default_hyperparams(self):
        m = ModelConfig(model_type="clam_sb")
        assert m.model_size == "small"
        assert m.dropout == 0.25
        assert m.bag_weight == 0.7
        assert m.B == 8


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------

class TestTrainConfig:
    def test_defaults(self):
        t = TrainConfig()
        assert t.max_epochs == 200
        assert t.lr == 2e-4
        assert t.early_stopping is True
        assert t.weighted_sample is True
        assert t.seed == 42


# ---------------------------------------------------------------------------
# EncoderDims
# ---------------------------------------------------------------------------

class TestEncoderDims:
    def test_seven_encoders(self, registries):
        assert len(registries.encoder_dims) == 7

    def test_all_dims_positive(self, registries):
        for key, dim in registries.encoder_dims.items():
            assert dim > 0, f"{key} has non-positive dim"

    def test_h0_mini_after_cls_extraction(self, registries):
        assert registries.encoder_dims["h0_mini"] == 768

    def test_known_dims(self, registries):
        assert registries.encoder_dims["conch_v15"] == 768
        assert registries.encoder_dims["hibou_l"] == 1024
        assert registries.encoder_dims["virchow2"] == 2560


# ---------------------------------------------------------------------------
# SplitStrategy (now plain strings + strategy_registry)
# ---------------------------------------------------------------------------

class TestSplitStrategy:
    def test_all_strategies_registered(self, ds, registries):
        for name in ds.split_strategies:
            assert name in registries.strategy_registry

    def test_standard_no_fixed_test(self, registries):
        cfg = registries.strategy_registry["standard"]
        assert cfg.train_cohorts == []
        assert cfg.test_cohorts == []


# ---------------------------------------------------------------------------
# Framework
# ---------------------------------------------------------------------------

class TestFramework:
    def test_enum_values(self):
        assert Framework.CLAM.value == "clam"
        assert Framework.NNMIL.value == "nnmil"


# ---------------------------------------------------------------------------
# nnMIL models
# ---------------------------------------------------------------------------

class TestNnmilModels:
    def test_eight_models(self, registries):
        """ab_mil moved to its own Framework.ABMIL arm; 9 - 1 = 8 remain."""
        assert len(registries.nnmil_models) == 8

    def test_known_models(self, registries):
        assert "trans_mil" in registries.nnmil_models
        assert "simple_mil" in registries.nnmil_models
        assert "ab_mil" not in registries.nnmil_models


# ---------------------------------------------------------------------------
# nnMIL runtime overrides
# ---------------------------------------------------------------------------

class TestNnmilRuntimeOverrides:
    def test_default_overrides_include_num_workers(self):
        assert NNMIL_RUNTIME_DEFAULTS["num_workers"] == 8

    def test_any_model_gets_defaults_only(self):
        # NNMIL_MODEL_RUNTIME_OVERRIDES is empty post-Level-D revert; every
        # model_type returns just NNMIL_RUNTIME_DEFAULTS.
        cfg = get_nnmil_runtime_overrides("trans_mil")
        assert cfg == {"num_workers": 8}
        cfg = get_nnmil_runtime_overrides("vision_transformer")
        assert cfg == {"num_workers": 8}

    def test_registry_keys_are_valid_nnmil_models(self, registries):
        for model_type in NNMIL_MODEL_RUNTIME_OVERRIDES:
            assert model_type in registries.nnmil_models


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------

class TestExperimentConfig:
    @pytest.fixture
    def exp(self, registries):
        return ExperimentConfig(
            task=registries.task_registry["brca"],
            encoder_key="conch_v15",
            embed_dim=768,
            model=registries.model_registry["clam_sb"],
            train=TrainConfig(seed=42),
            strategy="standard",
        )

    def test_experiment_id_format(self, exp):
        assert exp.experiment_id == "clam__standard__brca__conch_v15__clam_sb__s42"

    def test_experiment_id_changes_with_seed(self, exp):
        exp2 = ExperimentConfig(
            task=exp.task, encoder_key=exp.encoder_key,
            embed_dim=exp.embed_dim, model=exp.model,
            train=TrainConfig(seed=99),
            strategy="standard",
        )
        assert exp2.experiment_id == "clam__standard__brca__conch_v15__clam_sb__s99"

    def test_experiment_id_includes_framework_strategy(self, registries):
        exp = ExperimentConfig(
            task=registries.task_registry["brca"],
            encoder_key="conch_v15",
            embed_dim=768,
            model=ModelConfig(model_type="trans_mil"),
            train=TrainConfig(seed=42),
            framework=Framework.NNMIL,
            strategy="standard",
        )
        assert exp.experiment_id == "nnmil__standard__brca__conch_v15__trans_mil__s42"

    def test_results_subdir(self, registries):
        exp = ExperimentConfig(
            task=registries.task_registry["brca"],
            encoder_key="conch_v15",
            embed_dim=768,
            model=registries.model_registry["clam_sb"],
            train=TrainConfig(seed=42),
            framework=Framework.CLAM,
            strategy="standard",
        )
        assert exp.results_subdir == "clam/standard/brca/conch_v15/clam_sb"

    def test_to_dict(self, exp):
        d = exp.to_dict()
        assert d["encoder_key"] == "conch_v15"
        assert d["embed_dim"] == 768
        assert d["task"]["name"] == "brca"
        assert d["framework"] == "clam"
        assert d["strategy"] == "standard"

    def test_save_creates_file(self, exp, tmp_path):
        path = str(tmp_path / "config.json")
        exp.save(path)
        with open(path) as f:
            data = json.load(f)
        assert data["encoder_key"] == "conch_v15"
        assert data["task"]["name"] == "brca"

    def test_save_creates_parent_dirs(self, exp, tmp_path):
        path = str(tmp_path / "a" / "b" / "config.json")
        exp.save(path)
        assert (tmp_path / "a" / "b" / "config.json").exists()


# ---------------------------------------------------------------------------
# generate_all_experiments (now takes registries)
# ---------------------------------------------------------------------------

class TestGenerateAllExperiments:
    def test_full_grid_count(self, ds, registries):
        """Default config: CLAM + standard = 2 tasks x 7 encoders x 3 models = 42."""
        cfg = BenchmarkConfig.from_dataset_config(ds, strategies=["standard"])
        exps = generate_all_experiments(cfg, registries)
        assert len(exps) == 2 * 7 * 3  # 42

    def test_unique_ids(self, ds, registries):
        cfg = BenchmarkConfig.from_dataset_config(ds, strategies=["standard"])
        exps = generate_all_experiments(cfg, registries)
        ids = [e.experiment_id for e in exps]
        assert len(set(ids)) == len(ids)

    def test_subset_encoders(self, ds, registries):
        cfg = BenchmarkConfig.from_dataset_config(
            ds, strategies=["standard"], encoder_keys=["conch_v15", "uni_v2"],
        )
        exps = generate_all_experiments(cfg, registries)
        assert len(exps) == 2 * 2 * 3  # 12

    def test_subset_tasks(self, ds, registries):
        cfg = BenchmarkConfig.from_dataset_config(
            ds, strategies=["standard"], tasks=["brca"],
        )
        exps = generate_all_experiments(cfg, registries)
        assert len(exps) == 1 * 7 * 3  # 21
        assert all(e.task.name == "brca" for e in exps)

    def test_embed_dim_matches_encoder(self, ds, registries):
        cfg = BenchmarkConfig.from_dataset_config(ds, strategies=["standard"])
        for exp in generate_all_experiments(cfg, registries):
            assert exp.embed_dim == registries.encoder_dims[exp.encoder_key]

    def test_nnmil_framework_uses_nnmil_models(self, ds, registries):
        cfg = BenchmarkConfig.from_dataset_config(
            ds,
            frameworks=[Framework.NNMIL],
            strategies=["standard"],
            tasks=["brca"],
            encoder_keys=["conch_v15"],
        )
        exps = generate_all_experiments(cfg, registries)
        assert len(exps) == 8  # 1 task x 1 encoder x 8 nnmil models
        for exp in exps:
            assert exp.framework == Framework.NNMIL

    def test_multi_framework(self, ds, registries):
        cfg = BenchmarkConfig.from_dataset_config(
            ds,
            frameworks=[Framework.CLAM, Framework.NNMIL],
            strategies=["standard"],
            tasks=["brca"],
            encoder_keys=["conch_v15"],
        )
        exps = generate_all_experiments(cfg, registries)
        clam_exps = [e for e in exps if e.framework == Framework.CLAM]
        nnmil_exps = [e for e in exps if e.framework == Framework.NNMIL]
        assert len(clam_exps) == 3  # 3 CLAM models
        assert len(nnmil_exps) == 8  # 8 nnMIL models
