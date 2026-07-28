"""CR-5b: the per-fold results cache must be keyed by everything that changes the numbers.

The defect: CR-5 gave the *orchestrated* path an isolated ``AUTOMIL_RESULTS_DIR``,
but the static grid (``submit_benchmark.sh`` -> ``run_benchmark.py``) never sets it,
so every runner fell back to
``benchmark_dir/results/{framework}/{strategy}/{task}/{encoder}/{model}[/{loss}]``
-- a path carrying **no seed and no hyperparameter** -- and then every trainer
short-circuits on an existing ``fold_N/metrics.json``.

Two failure modes, both silent:

* ``--seed 43`` after a seed-42 grid returns **seed 42's numbers verbatim**, so a
  multi-seed variance study reports zero variance.
* Re-running after correcting a learning rate returns the **old** numbers.

Two different remedies, because the two cases want different behaviour:

* **seed** goes in the *path* -- seeds are meant to coexist, not to evict each other.
* **every other training-relevant field** is fingerprinted into a sidecar and a
  mismatch **fails loudly** with the purge command, following the same
  deliberately-not-self-healing precedent as the task-CSV cache guard
  (PRELAUNCH_REVIEW B2): concurrent experiments share ``benchmark_dir``, so a
  self-purging cache can delete folds another process is training from.
"""
from __future__ import annotations

import json
import os

import pytest

from autobench.pipeline.config import (
    ExperimentConfig,
    Framework,
    ModelConfig,
    TaskConfig,
    TrainConfig,
)
from autobench.pipeline.results_cache import (
    FINGERPRINT_FILENAME,
    StaleResultsCacheError,
    config_fingerprint,
    resolve_results_dir,
)


def _exp(seed: int = 42, lr: float = 2e-4, **train_kw) -> ExperimentConfig:
    return ExperimentConfig(
        task=TaskConfig(name="kras", label_col="KRAS", label_dict={"WT": 0, "MUT": 1}),
        encoder_key="uni_v2",
        embed_dim=1536,
        model=ModelConfig(model_type="clam_mb"),
        train=TrainConfig(seed=seed, lr=lr, **train_kw),
        framework=Framework.CLAM,
    )


class TestSeedIsInThePath:
    """Seeds must coexist. Without this, multi-seed is impossible to run at all."""

    def test_results_subdir_carries_the_seed(self):
        assert _exp(seed=42).results_subdir.endswith("s42")

    def test_two_seeds_do_not_share_a_directory(self):
        assert _exp(seed=42).results_subdir != _exp(seed=43).results_subdir

    def test_the_rest_of_the_path_is_unchanged(self):
        assert _exp().results_subdir == "clam/standard/kras/uni_v2/clam_mb/s42"

    def test_survival_loss_still_precedes_the_seed(self):
        cfg = _exp()
        cfg.survival_loss = "nllsurv"
        assert cfg.results_subdir == "clam/standard/kras/uni_v2/clam_mb/nllsurv/s42"

    def test_resolve_puts_two_seeds_in_different_dirs(self, tmp_path):
        a = resolve_results_dir(_exp(seed=42), str(tmp_path))
        b = resolve_results_dir(_exp(seed=43), str(tmp_path))
        assert a != b


class TestFingerprintCatchesHyperparameterDrift:
    """A changed knob at the same seed must not silently resume the old folds."""

    def test_identical_config_resumes_cleanly(self, tmp_path):
        first = resolve_results_dir(_exp(), str(tmp_path))
        second = resolve_results_dir(_exp(), str(tmp_path))
        assert first == second  # the 24h-wall self-resubmit case must keep working

    def test_sidecar_is_written(self, tmp_path):
        d = resolve_results_dir(_exp(), str(tmp_path))
        assert os.path.exists(os.path.join(d, FINGERPRINT_FILENAME))

    def test_changed_lr_raises_instead_of_resuming(self, tmp_path):
        resolve_results_dir(_exp(lr=2e-4), str(tmp_path))
        with pytest.raises(StaleResultsCacheError):
            resolve_results_dir(_exp(lr=1e-4), str(tmp_path))

    def test_changed_epochs_raises(self, tmp_path):
        resolve_results_dir(_exp(max_epochs=200), str(tmp_path))
        with pytest.raises(StaleResultsCacheError):
            resolve_results_dir(_exp(max_epochs=20), str(tmp_path))

    def test_error_names_the_directory_and_the_purge_command(self, tmp_path):
        resolve_results_dir(_exp(lr=2e-4), str(tmp_path))
        with pytest.raises(StaleResultsCacheError) as exc:
            resolve_results_dir(_exp(lr=1e-4), str(tmp_path))
        msg = str(exc.value)
        assert "rm -rf" in msg
        assert "lr" in msg  # names the field that actually changed

    def test_arm_specific_config_is_fingerprinted(self, tmp_path):
        """DTFD/ABMIL hold their knobs in their own dataclass, outside exp_cfg —
        a change there must invalidate the cache just the same."""
        from autobench.pipeline.dtfd.config import DTFDConfig
        from dataclasses import replace

        base = DTFDConfig()
        resolve_results_dir(_exp(), str(tmp_path), arm_cfg=base)
        with pytest.raises(StaleResultsCacheError):
            resolve_results_dir(_exp(), str(tmp_path), arm_cfg=replace(base, numGroup=8))

    def test_task_definition_change_invalidates(self, tmp_path):
        a = _exp()
        resolve_results_dir(a, str(tmp_path))
        b = _exp()
        b.task.n_classes = 3
        with pytest.raises(StaleResultsCacheError):
            resolve_results_dir(b, str(tmp_path))


class TestFingerprintIsStable:
    def test_same_config_same_fingerprint(self):
        assert config_fingerprint(_exp()) == config_fingerprint(_exp())

    def test_survival_losses_list_does_not_affect_it(self):
        """Adding a loss to the dataset YAML must not evict unrelated caches —
        only the loss actually being trained matters."""
        a, b = _exp(), _exp()
        b.task.survival_losses = ["cox", "nllsurv", "mse"]
        assert config_fingerprint(a) == config_fingerprint(b)

    def test_fingerprint_is_json_serialisable_and_records_the_fields(self, tmp_path):
        d = resolve_results_dir(_exp(), str(tmp_path))
        with open(os.path.join(d, FINGERPRINT_FILENAME)) as f:
            payload = json.load(f)
        assert "digest" in payload and "config" in payload
        assert payload["config"]["train"]["lr"] == 2e-4


class TestExplicitResultsDirStillWins:
    """The orchestrated path passes its own isolated dir (CR-5); keep that."""

    def test_explicit_dir_is_honoured(self, tmp_path):
        explicit = str(tmp_path / "node-archive" / "results")
        got = resolve_results_dir(_exp(), str(tmp_path), results_dir=explicit)
        assert got == explicit

    def test_explicit_dir_is_still_fingerprinted(self, tmp_path):
        explicit = str(tmp_path / "node-archive" / "results")
        resolve_results_dir(_exp(), str(tmp_path), results_dir=explicit)
        assert os.path.exists(os.path.join(explicit, FINGERPRINT_FILENAME))
