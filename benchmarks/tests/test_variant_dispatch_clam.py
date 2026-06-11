"""APL-02: variant fields thread through apply_model_variant_to_exp_cfg → _make_clam_args.

Tests in this file are RED until Plan 10-03 implements the body of
apply_model_variant_to_exp_cfg in variant_dispatch.py.

CI-safe: no GPU, no external data, no real CLAM run required.

Registry pollution guard: the autouse clear_registry_after_each fixture
calls _clear_registry() before and after every test (T-10-01 mitigation,
RESEARCH.md Pitfall 2).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pytest
import yaml

from automil.registry._state import MODEL_VARIANTS, LOSS_VARIANTS, _clear_registry
from automil.registry.spec import VariantSpec
from automil.registry.variants.model import ModelVariant
from autobench.pipeline.config import ExperimentConfig, ModelConfig, TaskConfig, TrainConfig
from autobench.pipeline.clam.train import _make_clam_args
from autobench.pipeline.variant_dispatch import apply_model_variant_to_exp_cfg


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_registry_after_each():
    """Ensure registry is clean before AND after each test (Pitfall 2)."""
    _clear_registry()
    yield
    _clear_registry()


def _minimal_exp_cfg(model_type: str = "clam_sb") -> ExperimentConfig:
    """Build a minimal ExperimentConfig for testing (no dataset required)."""
    return ExperimentConfig(
        task=TaskConfig(
            name="test_task",
            label_col="label",
            label_dict={"neg": 0, "pos": 1},
            n_classes=2,
        ),
        encoder_key="hibou_l",
        embed_dim=1024,
        model=ModelConfig(model_type=model_type),
        train=TrainConfig(),
    )


def _register_test_variant(
    clam_args: dict,
    name: str = "test_variant_v0",
    parent: str = "clam_mb",
) -> tuple[str, str]:
    """Register a concrete ModelVariant with given CLAM_ARGS and return (parent, name)."""

    class _TestVariant(ModelVariant):
        CLAM_ARGS = clam_args

        def forward(self, features: Any, coords: Optional[Any] = None) -> Any:
            raise NotImplementedError("test variant — not for real use")

    MODEL_VARIANTS[(parent, name)] = _TestVariant
    return parent, name


def _write_automil_config(automil_dir: Path, variant_name: str, parent: str | None = None) -> None:
    """Write a minimal automil/config.yaml with model.variant set."""
    config: dict = {"model": {"variant": variant_name}}
    if parent is not None:
        config["model"]["parent"] = parent
    (automil_dir / "config.yaml").write_text(yaml.safe_dump(config))


# ---------------------------------------------------------------------------
# Test 1: variant patches ModelConfig fields from CLAM_ARGS
# RED until Plan 10-03 implements apply_model_variant_to_exp_cfg.
# ---------------------------------------------------------------------------


def test_apply_variant_patches_model_config(tmp_path):
    """apply_model_variant_to_exp_cfg patches exp_cfg.model fields from CLAM_ARGS.

    After calling apply, exp_cfg.model.model_size must be "big" and
    exp_cfg.model.dropout must be 0.5 (from the registered variant's CLAM_ARGS).
    """
    automil_dir = tmp_path / "automil"
    automil_dir.mkdir()
    (automil_dir / "variants").mkdir()

    parent, name = _register_test_variant(
        clam_args={"model_size": "big", "dropout": 0.5},
        name="test_variant_v0",
        parent="clam_mb",
    )
    _write_automil_config(automil_dir, variant_name=name, parent=parent)

    exp_cfg = _minimal_exp_cfg()
    assert exp_cfg.model.model_size == "small"  # baseline default
    assert exp_cfg.model.dropout == 0.25         # baseline default

    apply_model_variant_to_exp_cfg(exp_cfg, automil_dir)

    assert exp_cfg.model.model_size == "big", (
        f"Expected model_size='big' after variant dispatch, got '{exp_cfg.model.model_size}'"
    )
    assert exp_cfg.model.dropout == 0.5, (
        f"Expected dropout=0.5 after variant dispatch, got {exp_cfg.model.dropout}"
    )


# ---------------------------------------------------------------------------
# Test 2: variant fields flow through _make_clam_args into args namespace
# RED until Plan 10-03 implements apply_model_variant_to_exp_cfg.
# ---------------------------------------------------------------------------


def test_variant_fields_flow_through_make_clam_args(tmp_path):
    """After apply, _make_clam_args returns args with variant overrides applied.

    This proves the full APL-02 chain:
      apply_model_variant_to_exp_cfg → exp_cfg.model patched
      _make_clam_args(exp_cfg, ...) → args.model_size == "big", args.drop_out == 0.5
    """
    automil_dir = tmp_path / "automil"
    automil_dir.mkdir()
    (automil_dir / "variants").mkdir()

    parent, name = _register_test_variant(
        clam_args={"model_size": "big", "dropout": 0.5},
        name="test_chain_v0",
        parent="clam_mb",
    )
    _write_automil_config(automil_dir, variant_name=name, parent=parent)

    exp_cfg = _minimal_exp_cfg()
    apply_model_variant_to_exp_cfg(exp_cfg, automil_dir)

    args = _make_clam_args(exp_cfg, "/tmp")
    assert args.model_size == "big", (
        f"Expected args.model_size='big', got '{args.model_size}'"
    )
    assert args.drop_out == 0.5, (
        f"Expected args.drop_out=0.5, got {args.drop_out}"
    )


# ---------------------------------------------------------------------------
# Test 3: no variant in config is a no-op (ModelConfig unchanged)
# RED until Plan 10-03 implements apply_model_variant_to_exp_cfg.
# ---------------------------------------------------------------------------


def test_no_variant_is_noop(tmp_path):
    """apply_model_variant_to_exp_cfg is a no-op when no model.variant in config."""
    automil_dir = tmp_path / "automil"
    automil_dir.mkdir()

    # Config with no model.variant key.
    (automil_dir / "config.yaml").write_text(yaml.safe_dump({"data": {"seed": 42}}))

    exp_cfg = _minimal_exp_cfg()
    original_model_type = exp_cfg.model.model_type
    original_dropout = exp_cfg.model.dropout

    apply_model_variant_to_exp_cfg(exp_cfg, automil_dir)

    assert exp_cfg.model.model_type == original_model_type
    assert exp_cfg.model.dropout == original_dropout


# ---------------------------------------------------------------------------
# Test 4: unknown variant name raises ValueError
# RED until Plan 10-03 implements apply_model_variant_to_exp_cfg.
# ---------------------------------------------------------------------------


def test_unknown_variant_raises(tmp_path):
    """apply_model_variant_to_exp_cfg raises ValueError for unregistered variant name."""
    automil_dir = tmp_path / "automil"
    automil_dir.mkdir()
    (automil_dir / "variants").mkdir()

    # Set a variant name that is NOT registered in MODEL_VARIANTS.
    _write_automil_config(automil_dir, variant_name="nonexistent_v9999", parent="clam_mb")

    exp_cfg = _minimal_exp_cfg()

    with pytest.raises(ValueError, match="not found in registry"):
        apply_model_variant_to_exp_cfg(exp_cfg, automil_dir)


# ---------------------------------------------------------------------------
# Test 5: TrainConfig field override via CLAM_ARGS "lr" key
# RED until Plan 10-03 implements apply_model_variant_to_exp_cfg.
# ---------------------------------------------------------------------------


def test_train_field_override_via_train_config(tmp_path):
    """CLAM_ARGS with "lr" key patches exp_cfg.train.lr (not exp_cfg.model)."""
    automil_dir = tmp_path / "automil"
    automil_dir.mkdir()
    (automil_dir / "variants").mkdir()

    parent, name = _register_test_variant(
        clam_args={"lr": 1e-3},
        name="test_lr_override_v0",
        parent="clam_mb",
    )
    _write_automil_config(automil_dir, variant_name=name, parent=parent)

    exp_cfg = _minimal_exp_cfg()
    assert exp_cfg.train.lr == 2e-4  # default

    apply_model_variant_to_exp_cfg(exp_cfg, automil_dir)

    assert exp_cfg.train.lr == pytest.approx(1e-3), (
        f"Expected train.lr=1e-3 after variant dispatch, got {exp_cfg.train.lr}"
    )
