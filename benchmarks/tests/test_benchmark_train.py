"""Tests for autobench.pipeline.clam.train module."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from autobench.pipeline.config import build_registries
from autobench.pipeline.clam.train import seed_everything
from autobench.pipeline.clam._imports import EarlyStopping, initiate_model
from _helpers import make_test_ds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ds():
    return make_test_ds()


@pytest.fixture
def registries(ds):
    return build_registries(ds)


# ---------------------------------------------------------------------------
# seed_everything
# ---------------------------------------------------------------------------


class TestSeedEverything:
    def test_torch_reproducible(self):
        seed_everything(42)
        a = torch.randn(10)
        seed_everything(42)
        b = torch.randn(10)
        assert torch.allclose(a, b)

    def test_numpy_reproducible(self):
        seed_everything(42)
        a = np.random.randn(10)
        seed_everything(42)
        b = np.random.randn(10)
        assert np.allclose(a, b)

    def test_different_seeds_differ(self):
        seed_everything(42)
        a = torch.randn(10)
        seed_everything(99)
        b = torch.randn(10)
        assert not torch.allclose(a, b)


# ---------------------------------------------------------------------------
# initiate_model (CLAM's checkpoint loading, used for val extended metrics)
# ---------------------------------------------------------------------------


def _make_clam_args(model_type: str, encoder_key: str = "conch_v15") -> SimpleNamespace:
    """Build a minimal CLAM-compatible args namespace for initiate_model."""
    ds = make_test_ds()
    return SimpleNamespace(
        model_type=model_type,
        model_size="small",
        n_classes=2,
        embed_dim=ds.encoder_dims[encoder_key],
        drop_out=0.25,
    )


class TestInitiateModel:
    """Test CLAM's initiate_model with our encoder dims and model types."""

    @pytest.mark.parametrize("model_type", ["clam_sb", "clam_mb", "mil"])
    def test_roundtrip_save_load(self, model_type, tmp_path, registries):
        """Save a model, load it back via initiate_model, verify forward pass."""
        from autobench.pipeline.clam._imports import CLAM_SB, CLAM_MB, MIL_fc

        embed_dim = registries.encoder_dims["conch_v15"]

        # Create model directly (same logic as CLAM's train())
        if model_type in ("clam_sb", "clam_mb"):
            cls = CLAM_SB if model_type == "clam_sb" else CLAM_MB
            model = cls(
                n_classes=2, embed_dim=embed_dim,
                dropout=0.25, size_arg="small", k_sample=8,
                instance_loss_fn=nn.CrossEntropyLoss(), gate=True, subtyping=False,
            )
        else:
            model = MIL_fc(
                size_arg="small", dropout=0.25, n_classes=2,
                embed_dim=embed_dim,
            )

        ckpt_path = str(tmp_path / "ckpt.pt")
        torch.save(model.state_dict(), ckpt_path)

        args = _make_clam_args(model_type)
        loaded = initiate_model(args, ckpt_path, torch.device("cpu"))
        x = torch.randn(50, embed_dim)
        logits, Y_prob, Y_hat, _, _ = loaded(x)
        assert Y_prob.shape == (1, 2)

    @pytest.mark.parametrize("encoder_key", list(make_test_ds().encoder_dims.keys()))
    def test_all_encoder_dims_work(self, encoder_key, tmp_path):
        """Each encoder's dim should produce a valid model via initiate_model."""
        from autobench.pipeline.clam._imports import CLAM_SB

        ds = make_test_ds()
        embed_dim = ds.encoder_dims[encoder_key]

        model = CLAM_SB(
            n_classes=2, embed_dim=embed_dim,
            dropout=0.25, size_arg="small", k_sample=8,
            instance_loss_fn=nn.CrossEntropyLoss(), gate=True, subtyping=False,
        )
        ckpt_path = str(tmp_path / f"ckpt_{encoder_key}.pt")
        torch.save(model.state_dict(), ckpt_path)

        args = _make_clam_args("clam_sb", encoder_key)
        loaded = initiate_model(args, ckpt_path, torch.device("cpu"))
        x = torch.randn(50, embed_dim)
        logits, Y_prob, Y_hat, _, _ = loaded(x)
        assert logits.shape == (1, 2)


# ---------------------------------------------------------------------------
# EarlyStopping (CLAM's implementation)
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    """Test CLAM's EarlyStopping class."""

    def _make_dummy_model(self):
        return nn.Linear(10, 2)

    def test_no_stop_if_improving(self, tmp_path):
        es = EarlyStopping(patience=3, stop_epoch=0)
        model = self._make_dummy_model()
        for i in range(10):  # rising validation AUC
            es(i, 0.5 + i * 0.04, model, ckpt_name=str(tmp_path / "ckpt.pt"))
            assert not es.early_stop

    def test_stops_after_patience(self, tmp_path):
        es = EarlyStopping(patience=3, stop_epoch=0)
        model = self._make_dummy_model()
        ckpt = str(tmp_path / "ckpt.pt")
        es(0, 0.8, model, ckpt)  # best
        es(1, 0.7, model, ckpt)
        es(2, 0.6, model, ckpt)
        es(3, 0.5, model, ckpt)  # 3rd non-improvement
        assert es.early_stop

    def test_respects_stop_epoch(self, tmp_path):
        es = EarlyStopping(patience=2, stop_epoch=10)
        model = self._make_dummy_model()
        ckpt = str(tmp_path / "ckpt.pt")
        es(0, 0.7, model, ckpt)
        es(1, 0.6, model, ckpt)
        es(2, 0.5, model, ckpt)  # patience exhausted but epoch < stop_epoch
        assert not es.early_stop

    def test_saves_checkpoint_on_improvement(self, tmp_path):
        es = EarlyStopping(patience=5, stop_epoch=0)
        model = self._make_dummy_model()
        ckpt = str(tmp_path / "ckpt.pt")
        es(0, 0.5, model, ckpt)
        assert os.path.exists(ckpt)

    def test_resets_counter_on_improvement(self, tmp_path):
        es = EarlyStopping(patience=3, stop_epoch=0)
        model = self._make_dummy_model()
        ckpt = str(tmp_path / "ckpt.pt")
        es(0, 0.7, model, ckpt)
        es(1, 0.6, model, ckpt)
        es(2, 0.5, model, ckpt)
        assert es.counter == 2
        es(3, 0.8, model, ckpt)  # new best
        assert es.counter == 0


class TestSubtypingDynamic:
    """clam/train.py sets ``subtyping=(exp_cfg.task.n_classes > 2)``.

    Binary tasks (label_smoothing-only, mutation status) keep
    ``subtyping=False`` so CLAM routes to the simple bag-loss path; multi-
    class tasks (CLWD subtype_7class etc.) get ``subtyping=True`` so
    CLAM routes to the per-class instance-loss path
    (``lib/CLAM/models/model_clam.py:146-168``). A hardcoded False would
    silently use the wrong path on multi-class. Locking this in.
    """

    def _make_exp_cfg(self, n_classes: int):
        from autobench.pipeline.config import (
            ExperimentConfig,
            Framework,
            ModelConfig,
            TaskConfig,
            TrainConfig,
        )
        return ExperimentConfig(
            task=TaskConfig(
                name="t",
                label_col="label",
                label_dict={f"c{i}": i for i in range(n_classes)},
                n_classes=n_classes,
            ),
            encoder_key="conch_v15",
            embed_dim=512,
            model=ModelConfig(model_type="clam_mb"),
            train=TrainConfig(),
            n_folds=10,
            framework=Framework.CLAM,
        )

    def test_binary_task_is_not_subtyping(self, tmp_path):
        from autobench.pipeline.clam.train import _make_clam_args
        args = _make_clam_args(self._make_exp_cfg(n_classes=2), str(tmp_path))
        assert args.subtyping is False

    def test_three_class_task_enables_subtyping(self, tmp_path):
        from autobench.pipeline.clam.train import _make_clam_args
        args = _make_clam_args(self._make_exp_cfg(n_classes=3), str(tmp_path))
        assert args.subtyping is True

    def test_seven_class_task_enables_subtyping(self, tmp_path):
        """CLWD subtype_7class is the real-world driver for this."""
        from autobench.pipeline.clam.train import _make_clam_args
        args = _make_clam_args(self._make_exp_cfg(n_classes=7), str(tmp_path))
        assert args.subtyping is True
