"""M-10: manifest-content fingerprint, unit-level.

See ``autobench.pipeline.manifest_guard`` module docstring for the defect
this closes: prepare_all validated a cached task CSV's schema but had no way
to tell the manifest it was derived FROM had since been rebuilt with
different values (same schema, same slide_id set, different content).
"""
from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from autobench.pipeline.manifest_guard import (
    FINGERPRINT_FILENAME,
    StaleManifestError,
    check_manifest_fingerprint,
    manifest_fingerprint,
)


def _write_manifest(path: str, status: str = "mapped_unique_case_id") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({
        "slide_id": ["s0", "s1", "s2"],
        "case_id": ["c0", "c1", "c2"],
        "status": [status, status, status],
        "label": [0, 1, 0],
    }).to_csv(path, index=False)


class TestManifestFingerprint:
    def test_same_content_same_digest(self, tmp_path):
        p1 = str(tmp_path / "a.csv")
        p2 = str(tmp_path / "b.csv")
        _write_manifest(p1)
        _write_manifest(p2)
        assert manifest_fingerprint(p1)["digest"] == manifest_fingerprint(p2)["digest"]

    def test_different_content_different_digest(self, tmp_path):
        p1 = str(tmp_path / "a.csv")
        p2 = str(tmp_path / "b.csv")
        _write_manifest(p1, status="mapped_unique_case_id")
        _write_manifest(p2, status="DIFFERENT_STATUS")
        assert manifest_fingerprint(p1)["digest"] != manifest_fingerprint(p2)["digest"]

    def test_mtime_size_mode_available(self, tmp_path):
        p = str(tmp_path / "a.csv")
        _write_manifest(p)
        fp = manifest_fingerprint(p, use_content_hash=False)
        assert fp["digest"].startswith("mtime_size:")


class TestCheckManifestFingerprint:
    def test_first_encounter_stamps_and_passes(self, tmp_path):
        mapping_csv = str(tmp_path / "manifest.csv")
        _write_manifest(mapping_csv)
        dataset_csv_dir = str(tmp_path / "benchmark" / "dataset_csv")

        check_manifest_fingerprint(dataset_csv_dir, mapping_csv)

        sidecar = os.path.join(dataset_csv_dir, FINGERPRINT_FILENAME)
        assert os.path.exists(sidecar)
        with open(sidecar) as f:
            stamped = json.load(f)
        assert stamped["digest"] == manifest_fingerprint(mapping_csv)["digest"]

    def test_matching_manifest_passes_silently(self, tmp_path):
        mapping_csv = str(tmp_path / "manifest.csv")
        _write_manifest(mapping_csv)
        dataset_csv_dir = str(tmp_path / "benchmark" / "dataset_csv")

        check_manifest_fingerprint(dataset_csv_dir, mapping_csv)
        # Second call, same manifest content: must not raise.
        check_manifest_fingerprint(dataset_csv_dir, mapping_csv)

    def test_rebuilt_manifest_raises_stale_manifest_error(self, tmp_path):
        """The core defect: a manifest rebuilt with a DIFFERENT value (same
        schema, same slide_id set) must be caught."""
        mapping_csv = str(tmp_path / "manifest.csv")
        _write_manifest(mapping_csv, status="mapped_unique_case_id")
        dataset_csv_dir = str(tmp_path / "benchmark" / "dataset_csv")
        check_manifest_fingerprint(dataset_csv_dir, mapping_csv)

        # Manifest rebuilt with a corrected value -- same columns, same
        # slide_ids, different content. No schema check could ever catch
        # this.
        _write_manifest(mapping_csv, status="CORRECTED_VALUE")

        with pytest.raises(StaleManifestError, match=r"rm -rf"):
            check_manifest_fingerprint(dataset_csv_dir, mapping_csv)

    def test_error_names_both_manifest_paths_and_the_purge_command(self, tmp_path):
        mapping_csv = str(tmp_path / "manifest.csv")
        _write_manifest(mapping_csv)
        dataset_csv_dir = str(tmp_path / "benchmark" / "dataset_csv")
        check_manifest_fingerprint(dataset_csv_dir, mapping_csv)
        _write_manifest(mapping_csv, status="CHANGED")

        with pytest.raises(StaleManifestError) as exc_info:
            check_manifest_fingerprint(dataset_csv_dir, mapping_csv)
        msg = str(exc_info.value)
        assert dataset_csv_dir in msg
        assert "splits" in msg

    def test_does_not_self_heal_the_stale_directory(self, tmp_path):
        """Deliberately not-self-healing (PRELAUNCH_REVIEW B2 / CR-5b
        precedent): the sidecar and any files in dataset_csv_dir must
        survive a raise untouched, since prepare_all runs concurrently
        against a SHARED benchmark_dir and self-purging previously caused
        FileNotFoundErrors across concurrent processes."""
        mapping_csv = str(tmp_path / "manifest.csv")
        _write_manifest(mapping_csv)
        dataset_csv_dir = str(tmp_path / "benchmark" / "dataset_csv")
        check_manifest_fingerprint(dataset_csv_dir, mapping_csv)

        marker = os.path.join(dataset_csv_dir, "some_task.csv")
        with open(marker, "w") as f:
            f.write("untouched")

        _write_manifest(mapping_csv, status="CHANGED")
        with pytest.raises(StaleManifestError):
            check_manifest_fingerprint(dataset_csv_dir, mapping_csv)

        assert os.path.exists(marker)
        assert os.path.exists(os.path.join(dataset_csv_dir, FINGERPRINT_FILENAME))

    def test_unreadable_sidecar_is_restamped_not_fatal(self, tmp_path):
        mapping_csv = str(tmp_path / "manifest.csv")
        _write_manifest(mapping_csv)
        dataset_csv_dir = str(tmp_path / "benchmark" / "dataset_csv")
        os.makedirs(dataset_csv_dir, exist_ok=True)
        with open(os.path.join(dataset_csv_dir, FINGERPRINT_FILENAME), "w") as f:
            f.write("{not valid json")

        # Must not raise -- re-stamps instead of blocking on a corrupt sidecar.
        check_manifest_fingerprint(dataset_csv_dir, mapping_csv)
        with open(os.path.join(dataset_csv_dir, FINGERPRINT_FILENAME)) as f:
            assert json.load(f)["digest"] == manifest_fingerprint(mapping_csv)["digest"]
