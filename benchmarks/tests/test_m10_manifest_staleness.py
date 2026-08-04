"""M-10: prepare_all must fail loudly when the manifest was rebuilt.

Reproduces the original defect end-to-end through the real orchestrator
(``prepare_all``), not just the ``manifest_guard`` unit -- proving the
existing schema/coverage checks (PRELAUNCH_REVIEW B2, and the splits
fold0-vs-csv check) would have passed a rebuilt-but-same-shape manifest
silently before this fix.
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import pandas as pd
import pytest

from _helpers import make_test_ds

from autobench.config import TaskDef
from autobench.pipeline.manifest_guard import StaleManifestError
from autobench.pipeline.prepare import prepare_all


N_SLIDES = 40  # large enough for n_splits=2's inner val carve to stay feasible
FEAT_DIM = 16


def _write_mapping_csv(path: str, labels: list[int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = [
        {
            "new_name": f"slide_{i:03d}.svs",
            "status": "mapped_unique_case_id",
            "primary_case_id": f"K{i:03d}",
            "BRCA_predict_label": labels[i],
        }
        for i in range(N_SLIDES)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_h5_features(features_base_dir: str) -> None:
    feat_dir = os.path.join(features_base_dir, "features_fake_enc")
    os.makedirs(feat_dir, exist_ok=True)
    for i in range(N_SLIDES):
        n_patches = 10
        with h5py.File(os.path.join(feat_dir, f"slide_{i:03d}.h5"), "w") as f:
            f.create_dataset(
                "features",
                data=np.random.RandomState(i).randn(n_patches, FEAT_DIM).astype(np.float32),
            )


def _ds(mapping_csv: str, benchmark_dir: str, features_base_dir: str):
    return make_test_ds(
        mapping_csv=mapping_csv,
        benchmark_dir=benchmark_dir,
        features_base_dir=features_base_dir,
        tasks={
            "brca": TaskDef(
                name="brca", label_col="BRCA_predict_label",
                label_map={0: "neg", 1: "pos"}, n_classes=2,
            ),
        },
        task_strategy_feasibility={"brca": ["standard"]},
        encoder_models={"test/fake": "fake_enc"},
        encoder_dims={"fake_enc": FEAT_DIM},
    )


@pytest.fixture
def prepared_benchmark(tmp_path):
    """A benchmark_dir that has already been through prepare_all once."""
    mapping_csv = str(tmp_path / "mapping.csv")
    # Balanced labels so 5-fold-safe n_splits=2 stratification is feasible.
    labels = [i % 2 for i in range(N_SLIDES)]
    _write_mapping_csv(mapping_csv, labels)

    benchmark_dir = str(tmp_path / "benchmark")
    features_base_dir = str(tmp_path / "features")
    _write_h5_features(features_base_dir)

    ds = _ds(mapping_csv, benchmark_dir, features_base_dir)
    prepare_all(
        benchmark_dir=benchmark_dir, mapping_csv=mapping_csv,
        features_base_dir=features_base_dir, encoder_keys=["fake_enc"],
        ds=ds, seed=42, n_splits=2,
    )
    return {
        "mapping_csv": mapping_csv, "benchmark_dir": benchmark_dir,
        "features_base_dir": features_base_dir, "ds": ds, "labels": labels,
    }


class TestPrepareAllManifestStaleness:
    def test_first_run_succeeds_and_stamps_the_sidecar(self, prepared_benchmark):
        sidecar = os.path.join(
            prepared_benchmark["benchmark_dir"], "dataset_csv", "manifest_fingerprint.json",
        )
        assert os.path.exists(sidecar)
        csv_path = os.path.join(prepared_benchmark["benchmark_dir"], "dataset_csv", "brca.csv")
        assert os.path.exists(csv_path)

    def test_rerun_with_unchanged_manifest_succeeds(self, prepared_benchmark):
        """Idempotent re-run (the documented contract) must not be blocked
        by the new check when nothing changed."""
        prepare_all(
            benchmark_dir=prepared_benchmark["benchmark_dir"],
            mapping_csv=prepared_benchmark["mapping_csv"],
            features_base_dir=prepared_benchmark["features_base_dir"],
            encoder_keys=["fake_enc"], ds=prepared_benchmark["ds"],
            seed=42, n_splits=2,
        )

    def test_rebuilt_manifest_with_same_schema_raises(self, prepared_benchmark):
        """The core defect: rebuild the manifest with DIFFERENT label values
        for the SAME slides (a same-schema, same-slide-id-set change --
        exactly what the existing B2 schema check and the splits
        fold0-vs-csv coverage check cannot see). Must now raise
        StaleManifestError instead of silently reusing the stale cache."""
        # Flip every label -- same columns, same slide_ids, different content.
        flipped = [1 - lbl for lbl in prepared_benchmark["labels"]]
        _write_mapping_csv(prepared_benchmark["mapping_csv"], flipped)

        with pytest.raises(StaleManifestError, match=r"rm -rf"):
            prepare_all(
                benchmark_dir=prepared_benchmark["benchmark_dir"],
                mapping_csv=prepared_benchmark["mapping_csv"],
                features_base_dir=prepared_benchmark["features_base_dir"],
                encoder_keys=["fake_enc"], ds=prepared_benchmark["ds"],
                seed=42, n_splits=2,
            )

    def test_raises_before_touching_any_task_csv_or_splits(self, prepared_benchmark, tmp_path):
        """Fail-fast: the manifest check runs before the per-task loop, so a
        mismatch is caught before any (further) task CSV/splits work."""
        csv_path = os.path.join(prepared_benchmark["benchmark_dir"], "dataset_csv", "brca.csv")
        before_mtime = os.path.getmtime(csv_path)

        flipped = [1 - lbl for lbl in prepared_benchmark["labels"]]
        _write_mapping_csv(prepared_benchmark["mapping_csv"], flipped)

        with pytest.raises(StaleManifestError):
            prepare_all(
                benchmark_dir=prepared_benchmark["benchmark_dir"],
                mapping_csv=prepared_benchmark["mapping_csv"],
                features_base_dir=prepared_benchmark["features_base_dir"],
                encoder_keys=["fake_enc"], ds=prepared_benchmark["ds"],
                seed=42, n_splits=2,
            )
        # The cached CSV must be untouched -- not rewritten, not deleted.
        assert os.path.exists(csv_path)
        assert os.path.getmtime(csv_path) == before_mtime

    def test_fresh_benchmark_dir_is_unaffected(self, tmp_path):
        """A brand new benchmark_dir (no prior prepare_all run) must not be
        rejected -- first encounter stamps and proceeds."""
        mapping_csv = str(tmp_path / "mapping.csv")
        _write_mapping_csv(mapping_csv, [i % 2 for i in range(N_SLIDES)])
        benchmark_dir = str(tmp_path / "benchmark")
        features_base_dir = str(tmp_path / "features")
        _write_h5_features(features_base_dir)
        ds = _ds(mapping_csv, benchmark_dir, features_base_dir)

        prepare_all(
            benchmark_dir=benchmark_dir, mapping_csv=mapping_csv,
            features_base_dir=features_base_dir, encoder_keys=["fake_enc"],
            ds=ds, seed=42, n_splits=2,
        )
        assert os.path.exists(
            os.path.join(benchmark_dir, "dataset_csv", "manifest_fingerprint.json")
        )
