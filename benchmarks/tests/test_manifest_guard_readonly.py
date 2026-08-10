"""M-10b: the fingerprint guard must tolerate read-only shared prep dirs.

Found by preprint-130 baseline-repair job 53780472 (2026-08-09). The campaign
declares the legacy ``benchmark_5fold/`` prep tree immutable
(``legacy_policy: "read-only"``), and on the shared cluster those dirs can be
owned by another lab member with group read-only permissions. The guard's
first-encounter stamp (``_write_atomic``) then dies with ``PermissionError``
inside ``prepare_all`` -- before any training -- and the whole cohort is lost
(cptac_pdac: 0/12 cells, dataset_csv owned by another user, mode 2750).

An unstampable read-only dir cannot have its derived CSVs/splits rebuilt
either, so the staleness risk the sidecar exists to catch cannot arise from
this account. Degrade to a loud warning; keep every other path intact:
- existing sidecar + matching digest: silent success (unchanged)
- existing sidecar + different digest: StaleManifestError (unchanged)
- writable dir, no sidecar: stamp (unchanged)
"""
from __future__ import annotations

import json
import os
import stat

import pytest

from autobench.pipeline.manifest_guard import (
    FINGERPRINT_FILENAME,
    StaleManifestError,
    check_manifest_fingerprint,
    manifest_fingerprint,
)


@pytest.fixture
def mapping_csv(tmp_path):
    path = tmp_path / "mapping.csv"
    path.write_text("new_name,label\nslide_000.svs,1\n")
    return str(path)


@pytest.fixture
def readonly_csv_dir(tmp_path):
    csv_dir = tmp_path / "dataset_csv"
    csv_dir.mkdir()
    yield csv_dir
    csv_dir.chmod(stat.S_IRWXU)  # restore so pytest can clean tmp_path


class TestReadOnlyPrepDir:
    def test_first_encounter_warns_instead_of_crashing(
        self, readonly_csv_dir, mapping_csv,
    ):
        readonly_csv_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
        with pytest.warns(RuntimeWarning, match="read-only"):
            check_manifest_fingerprint(str(readonly_csv_dir), mapping_csv)
        assert not (readonly_csv_dir / FINGERPRINT_FILENAME).exists()

    def test_matching_sidecar_still_passes_silently(
        self, readonly_csv_dir, mapping_csv,
    ):
        current = manifest_fingerprint(mapping_csv)
        (readonly_csv_dir / FINGERPRINT_FILENAME).write_text(json.dumps(current))
        readonly_csv_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
        check_manifest_fingerprint(str(readonly_csv_dir), mapping_csv)

    def test_stale_sidecar_still_raises(self, readonly_csv_dir, mapping_csv):
        current = manifest_fingerprint(mapping_csv)
        stale = {**current, "digest": "0" * 64}
        (readonly_csv_dir / FINGERPRINT_FILENAME).write_text(json.dumps(stale))
        readonly_csv_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)
        with pytest.raises(StaleManifestError):
            check_manifest_fingerprint(str(readonly_csv_dir), mapping_csv)


class TestWritableDirUnchanged:
    def test_first_encounter_still_stamps(self, tmp_path, mapping_csv):
        csv_dir = tmp_path / "dataset_csv"
        csv_dir.mkdir()
        check_manifest_fingerprint(str(csv_dir), mapping_csv)
        sidecar = csv_dir / FINGERPRINT_FILENAME
        assert sidecar.exists()
        stored = json.loads(sidecar.read_text())
        assert stored["digest"] == manifest_fingerprint(mapping_csv)["digest"]
