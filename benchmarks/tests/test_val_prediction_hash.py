"""A4': per-fold val predictions are persisted and hashed, arm by arm.

The canary finding this serves (§0.4c): a 47-patient val split cannot even
distinguish a changed model from an unchanged one by metrics — val_auc is an
exact rank fraction. The hash of the persisted per-fold val predictions is the
no-op detector: identical bytes = identical selected model on that fold.

Contract under test:
  * every fold dir persists ``predictions_val.csv`` (classification: shared
    writer schema; survival: ``patient_id,status,time,risk_score``);
  * the fold result carries ``val_predictions_sha256`` = sha256 of that file;
  * the runner summary carries ``per_fold_val_predictions_sha256`` positional
    with ``per_fold_val``;
  * ``summary_to_result_json`` lifts it into each ``validation_folds`` entry
    at ENTRY level — never inside the exact-key-locked ``metrics``.

Every test drives the REAL producers (real training, real assembly) and only
READS the artifacts it asserts — per tasks/lessons.md, nothing here writes
the artifact under test.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import torch

from autobench.pipeline.config import build_registries
from tests.test_abmil_arm import (
    _build_benchmark_fixture,
    _exp_cfg,
    _smoke_cfg,
    make_test_ds,
)

_HEX64 = "0123456789abcdef"


def _load_run_experiment() -> ModuleType:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    script_path = scripts_dir / "run_experiment.py"
    if not script_path.exists():
        pytest.skip(f"run_experiment.py not found at {script_path}")
    mod_name = "run_experiment_a4"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _is_sha256_hex(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in _HEX64 for c in value)
    )


# ---------------------------------------------------------------------------
# ABMIL (cheapest arm): hash flows run -> summary -> validation_folds entries.
# ---------------------------------------------------------------------------


class TestHashFlowsThroughRealAbmilRun:
    @pytest.fixture(scope="class")
    def real_run(self, tmp_path_factory):
        from autobench.pipeline.abmil.runner import run_abmil_experiment

        root = tmp_path_factory.mktemp("abmil-valhash")
        _build_benchmark_fixture(str(root), n_folds=2)
        exp = _exp_cfg(build_registries(make_test_ds()), n_folds=2)
        summary = run_abmil_experiment(exp, str(root), device="cpu", cfg=_smoke_cfg())
        results_dir = os.path.join(str(root), "results", exp.results_subdir)
        return summary, results_dir

    def test_summary_hashes_match_the_files_on_disk(self, real_run):
        summary, results_dir = real_run
        hashes = summary["per_fold_val_predictions_sha256"]
        assert len(hashes) == len(summary["per_fold_val"]) == 2
        for fold, digest in zip(summary["fold_indices"], hashes):
            assert _is_sha256_hex(digest)
            csv_path = os.path.join(results_dir, f"fold_{fold}", "predictions_val.csv")
            assert os.path.exists(csv_path)
            assert digest == _sha256(csv_path)

    def test_validation_fold_entries_carry_the_hash_at_entry_level(self, real_run):
        summary, _ = real_run
        m = _load_run_experiment()
        result = m.summary_to_result_json(summary, 1.0)
        folds = result["validation_folds"]
        assert len(folds) == 2
        for entry, digest in zip(folds, summary["per_fold_val_predictions_sha256"]):
            assert entry["val_predictions_sha256"] == digest
            # ENTRY level only: the campaign controller exact-key-locks
            # `metrics`, and CR-1b folds every value in there into the
            # primary_value. A hash inside metrics would corrupt selection.
            assert "val_predictions_sha256" not in entry["metrics"]

    def test_fold_metrics_json_carries_the_hash(self, real_run):
        summary, results_dir = real_run
        for fold, digest in zip(
            summary["fold_indices"], summary["per_fold_val_predictions_sha256"],
        ):
            with open(os.path.join(results_dir, f"fold_{fold}", "metrics.json")) as f:
                on_disk = json.load(f)
            assert on_disk["val_predictions_sha256"] == digest


# ---------------------------------------------------------------------------
# CLAM classification: the full real path (real clam_train, real CLAM model).
# ---------------------------------------------------------------------------


class TestClamClassificationRealRun:
    N_SLIDES = 12
    EMBED_DIM = 64
    N_PATCHES = 24  # >= 2 * ModelConfig.B so instance clustering can sample

    @pytest.fixture(scope="class")
    def clam_fold(self, tmp_path_factory):
        from autobench.pipeline.clam.dataset import create_dataset, load_fold_splits
        from autobench.pipeline.clam.train import train_fold
        from autobench.pipeline.config import (
            ExperimentConfig,
            Framework,
            ModelConfig,
            TaskConfig,
            TrainConfig,
        )
        from autobench.pipeline.policy_dispatch import PolicyRuntime

        root = tmp_path_factory.mktemp("clam-valhash")
        benchmark_dir = str(root / "benchmark")
        encoder = "conch_v15"

        rng = np.random.default_rng(3)
        slide_ids = [f"c{i}" for i in range(self.N_SLIDES)]
        labels = {sid: i % 2 for i, sid in enumerate(slide_ids)}

        pt_dir = os.path.join(benchmark_dir, "features", encoder, "pt_files")
        os.makedirs(pt_dir, exist_ok=True)
        for sid in slide_ids:
            feats = rng.standard_normal(
                (self.N_PATCHES, self.EMBED_DIM)
            ).astype("float32") + labels[sid] * 3.0
            torch.save(torch.from_numpy(feats), os.path.join(pt_dir, f"{sid}.pt"))

        csv_dir = os.path.join(benchmark_dir, "dataset_csv")
        os.makedirs(csv_dir, exist_ok=True)
        names = {0: "neg", 1: "pos"}
        pd.DataFrame({
            "slide_id": slide_ids,
            "case_id": slide_ids,
            "label": [names[labels[s]] for s in slide_ids],
        }).to_csv(os.path.join(csv_dir, "brca.csv"), index=False)

        splits_dir = os.path.join(benchmark_dir, "splits", "standard", "brca")
        os.makedirs(splits_dir, exist_ok=True)
        test_ids, val_ids = slide_ids[0:2], slide_ids[2:4]  # one class each
        train_ids = slide_ids[4:]
        pd.DataFrame({
            "train": train_ids,
            "val": val_ids + [None] * (len(train_ids) - len(val_ids)),
            "test": test_ids + [None] * (len(train_ids) - len(test_ids)),
        }).to_csv(os.path.join(splits_dir, "splits_0.csv"), index=False)

        exp_cfg = ExperimentConfig(
            task=TaskConfig(
                name="brca", label_col="label",
                label_dict={"neg": 0, "pos": 1}, n_classes=2,
            ),
            encoder_key=encoder,
            embed_dim=self.EMBED_DIM,
            model=ModelConfig(model_type="clam_sb"),
            train=TrainConfig(max_epochs=2, early_stopping=False, seed=42),
            n_folds=1,
            framework=Framework.CLAM,
            strategy="standard",
        )

        dataset = create_dataset(exp_cfg, benchmark_dir)
        train_split, val_split, test_split = load_fold_splits(
            dataset, benchmark_dir, os.path.join("standard", "brca"), 0,
            task_csv_name="brca",
        )

        results_dir = str(root / "results")
        result = train_fold(
            exp_cfg, train_split, val_split, test_split,
            fold=0, results_dir=results_dir, device=torch.device("cpu"),
            policy_runtime=PolicyRuntime(),
        )
        return result, os.path.join(results_dir, "fold_0")

    def test_predictions_val_csv_is_written(self, clam_fold):
        _, fold_dir = clam_fold
        csv_path = os.path.join(fold_dir, "predictions_val.csv")
        assert os.path.exists(csv_path)
        df = pd.read_csv(csv_path)
        assert list(df.columns[:2]) == ["slide_id", "y_true"]
        assert len(df) == 2  # the val split

    def test_fold_result_hash_matches_the_file(self, clam_fold):
        result, fold_dir = clam_fold
        digest = result["val_predictions_sha256"]
        assert _is_sha256_hex(digest)
        assert digest == _sha256(os.path.join(fold_dir, "predictions_val.csv"))
        with open(os.path.join(fold_dir, "metrics.json")) as f:
            assert json.load(f)["val_predictions_sha256"] == digest


# ---------------------------------------------------------------------------
# Survival: the cheapest real survival trainer persists val risk scores.
# ---------------------------------------------------------------------------


class TestAbmilSurvivalRealFold:
    @pytest.fixture(scope="class")
    def survival_fold(self, tmp_path_factory):
        import h5py

        from autobench.pipeline.abmil.config import ABMILConfig
        from autobench.pipeline.abmil.dataset import ABMILSurvivalSlide
        from autobench.pipeline.abmil.survival_train import train_abmil_survival_fold

        root = tmp_path_factory.mktemp("abmil-surv-valhash")
        rng = np.random.default_rng(11)

        def _samples(prefix, n):
            out = []
            for i in range(n):
                path = str(root / f"{prefix}{i}.h5")
                with h5py.File(path, "w") as f:
                    f.create_dataset(
                        "features",
                        data=rng.standard_normal((15, 32)).astype("float32"),
                    )
                out.append(ABMILSurvivalSlide(
                    slide_id=f"{prefix}{i}", h5_path=path,
                    status=i % 2, time=float(100 + 50 * i),
                    patient_id=f"P{prefix}{i}",
                ))
            return out

        fold_dir = str(root / "fold_0")
        os.makedirs(fold_dir, exist_ok=True)
        result = train_abmil_survival_fold(
            "abmil", _samples("tr", 8), _samples("va", 4), _samples("te", 4),
            embed_dim=32, survival_loss="cox", nll_bins=4,
            cfg=ABMILConfig(M=16, L=8, max_epochs=2, early_stopping=False),
            device=torch.device("cpu"), seed=7, fold_dir=fold_dir,
        )
        return result, fold_dir

    def test_val_risk_scores_are_persisted(self, survival_fold):
        result, fold_dir = survival_fold
        csv_path = os.path.join(fold_dir, "predictions_val.csv")
        assert os.path.exists(csv_path)
        df = pd.read_csv(csv_path)
        assert list(df.columns) == ["patient_id", "status", "time", "risk_score"]
        assert len(df) == len(result["val_records"]["risks"]) == 4

    def test_csv_rows_are_the_val_records_already_in_hand(self, survival_fold):
        result, fold_dir = survival_fold
        df = pd.read_csv(os.path.join(fold_dir, "predictions_val.csv"))
        np.testing.assert_allclose(
            df["risk_score"].to_numpy(),
            np.asarray(result["val_records"]["risks"], dtype=float),
            rtol=1e-6,
        )
        assert df["patient_id"].tolist() == result["val_records"]["patient_ids"]
