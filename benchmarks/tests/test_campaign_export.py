"""Contract tests for the standalone baseline mirror (campaign_export.py)."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "campaign_export", REPO_ROOT / "benchmarks/scripts/campaign_export.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ce = _load_module()

CELL_ID = "tcga_luad__kras__uni_v2__clam__s42__preprint-v3"


def _sha256_bytes(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


@pytest.fixture()
def fabricated_cell(tmp_path):
    """A registered cell root with hash-consistent ledger and archive."""
    runtime = tmp_path / "runtime"
    archive = runtime / CELL_ID / "baseline-execution" / "archive"
    certify = archive / "certify"
    (certify / "results" / "fold_0").mkdir(parents=True)

    result_bytes = json.dumps({"status": "completed", "metrics": {}}).encode()
    (archive / "result.json").write_bytes(result_bytes)
    (archive / "run.log").write_text("log\n")

    sealed_hashes = {}
    for fold in range(5):
        payload = json.dumps({"fold": fold, "held_out": {"test_auc": 0.5}}).encode()
        (certify / f"fold_{fold}_result.json").write_bytes(payload)
        sealed_hashes[f"fold_{fold}_result.json"] = _sha256_bytes(payload)
    (certify / "result.json").write_text("{\"held_out\": {}}")
    (certify / "results" / "fold_0" / "predictions.csv").write_text("sid,y\n")

    attestation = {"attestation_sha256": "a" * 64, "cell_id": CELL_ID}
    (archive / "baseline_attestation.json").write_text(json.dumps(attestation))

    state = {
        "cell_id": CELL_ID,
        "baseline": {
            "result_sha256": _sha256_bytes(result_bytes),
            "attestation_sha256": "a" * 64,
            "sealed_fold_sha256": sealed_hashes,
        },
    }
    (runtime / CELL_ID / "campaign_state.json").write_text(json.dumps(state))

    export_root = tmp_path / "version3"
    export_root.mkdir()
    return runtime, export_root


def test_mapping_is_collision_free_over_the_frozen_manifest(tmp_path):
    cells = ce._manifest_cells(REPO_ROOT)
    leaves = {ce.leaf_dir(tmp_path, cell["cell_id"]) for cell in cells}
    assert len(leaves) == len(cells) == 130


def test_mapping_shape_is_cohort_framework_task_encoder(tmp_path):
    leaf = ce.leaf_dir(tmp_path, CELL_ID)
    assert leaf == tmp_path / "tcga_luad" / "clam" / "kras" / "uni_v2"
    sealed = ce.sealed_dir(tmp_path, CELL_ID)
    assert sealed == tmp_path / "sealed" / "tcga_luad" / "clam" / "kras" / "uni_v2"


@pytest.mark.parametrize(
    "bad", ["", "a__b", "a__b__c__d__e", "a__b__c__d__e__f__g", "a____c__d__e__f"]
)
def test_malformed_cell_ids_are_refused(tmp_path, bad):
    with pytest.raises(ce.ExportError):
        ce.leaf_dir(tmp_path, bad)


def test_roster_census_matches_the_committed_declaration():
    cells = ce.roster_cell_ids(REPO_ROOT)
    assert len(cells) == 78
    assert {cell.split("__")[0] for cell in cells} == {
        "tcga_luad", "tcga_hnsc", "cptac_pdac",
    }


def test_export_cycle_splits_sealed_from_public(fabricated_cell):
    runtime, export_root = fabricated_cell
    assert ce.export_cell(runtime, export_root, CELL_ID) == "exported"

    leaf = ce.leaf_dir(export_root, CELL_ID)
    assert (leaf / "baseline" / "result.json").is_file()
    assert (leaf / "baseline" / "run.log").is_file()
    assert (leaf / "baseline" / "baseline_attestation.json").is_file()
    assert not (leaf / "baseline" / "certify").exists()
    assert (leaf / "campaign_state.json").is_file()
    assert (leaf / "EXPORT_OK").is_file()

    sealed = ce.sealed_dir(export_root, CELL_ID)
    assert (sealed / "certify" / "fold_0_result.json").is_file()
    assert (sealed / "certify" / "results" / "fold_0" / "predictions.csv").is_file()
    mode = os.stat(export_root / "sealed").st_mode & 0o777
    assert mode == 0o700

    marker = json.loads((leaf / "EXPORT_OK").read_text())
    state = json.loads((runtime / CELL_ID / "campaign_state.json").read_text())
    assert marker["result_sha256"] == state["baseline"]["result_sha256"]
    assert marker["attestation_sha256"] == state["baseline"]["attestation_sha256"]


def test_public_mirror_is_group_readable_despite_owner_only_sources(fabricated_cell):
    """Compute-node umask yields 0600 sources; the team must still read."""
    runtime, export_root = fabricated_cell
    archive = runtime / CELL_ID / "baseline-execution" / "archive"
    for path in archive.rglob("*"):
        os.chmod(path, 0o700 if path.is_dir() else 0o600)
    ce.export_cell(runtime, export_root, CELL_ID)

    leaf = ce.leaf_dir(export_root, CELL_ID)
    assert os.stat(leaf).st_mode & 0o070 == 0o050
    for public in (
        leaf / "EXPORT_OK",
        leaf / "campaign_state.json",
        leaf / "baseline" / "result.json",
    ):
        assert os.stat(public).st_mode & 0o070 == 0o040, public
    assert os.stat(export_root / "sealed").st_mode & 0o777 == 0o700


def test_reexport_is_a_cheap_no_op_when_current(fabricated_cell):
    runtime, export_root = fabricated_cell
    assert ce.export_cell(runtime, export_root, CELL_ID) == "exported"
    assert ce.export_cell(runtime, export_root, CELL_ID) == "current"


def test_partial_mirror_without_marker_is_repaired(fabricated_cell):
    runtime, export_root = fabricated_cell
    ce.export_cell(runtime, export_root, CELL_ID)
    (ce.leaf_dir(export_root, CELL_ID) / "EXPORT_OK").unlink()
    assert ce.export_cell(runtime, export_root, CELL_ID) == "exported"


def test_ledger_hash_mismatch_fails_and_leaves_no_marker(fabricated_cell):
    """Forged violation: destination bytes not matching the ledger must fail."""
    runtime, export_root = fabricated_cell
    state_path = runtime / CELL_ID / "campaign_state.json"
    state = json.loads(state_path.read_text())
    state["baseline"]["result_sha256"] = "0" * 64
    state_path.write_text(json.dumps(state))

    with pytest.raises(ce.ExportError, match="hash mismatch"):
        ce.export_cell(runtime, export_root, CELL_ID)
    assert not (ce.leaf_dir(export_root, CELL_ID) / "EXPORT_OK").exists()


def test_missing_sealed_fold_fails_closed(fabricated_cell):
    runtime, export_root = fabricated_cell
    archive = runtime / CELL_ID / "baseline-execution" / "archive"
    (archive / "certify" / "fold_3_result.json").unlink()

    with pytest.raises(ce.ExportError, match="fold_3"):
        ce.export_cell(runtime, export_root, CELL_ID)
    assert not (ce.leaf_dir(export_root, CELL_ID) / "EXPORT_OK").exists()


def test_unregistered_cell_is_refused(fabricated_cell):
    runtime, export_root = fabricated_cell
    state_path = runtime / CELL_ID / "campaign_state.json"
    state = json.loads(state_path.read_text())
    state["baseline"] = None
    state_path.write_text(json.dumps(state))
    with pytest.raises(ce.ExportError, match="no registered baseline"):
        ce.export_cell(runtime, export_root, CELL_ID)


def test_state_cell_identity_mismatch_is_refused(fabricated_cell):
    runtime, export_root = fabricated_cell
    state_path = runtime / CELL_ID / "campaign_state.json"
    state = json.loads(state_path.read_text())
    state["cell_id"] = "tcga_luad__os__uni_v2__clam__s42__preprint-v3"
    state_path.write_text(json.dumps(state))
    with pytest.raises(ce.ExportError, match="carries cell_id"):
        ce.export_cell(runtime, export_root, CELL_ID)


def test_marker_from_older_registration_forces_reexport(fabricated_cell):
    """A stale EXPORT_OK (different result hash) must not mask a re-export."""
    runtime, export_root = fabricated_cell
    ce.export_cell(runtime, export_root, CELL_ID)

    archive = runtime / CELL_ID / "baseline-execution" / "archive"
    new_result = json.dumps({"status": "completed", "metrics": {"v": 2}}).encode()
    (archive / "result.json").write_bytes(new_result)
    state_path = runtime / CELL_ID / "campaign_state.json"
    state = json.loads(state_path.read_text())
    state["baseline"]["result_sha256"] = _sha256_bytes(new_result)
    state_path.write_text(json.dumps(state))

    assert ce.export_cell(runtime, export_root, CELL_ID) == "exported"
    marker = json.loads(
        (ce.leaf_dir(export_root, CELL_ID) / "EXPORT_OK").read_text()
    )
    assert marker["result_sha256"] == _sha256_bytes(new_result)


def test_export_root_is_required(tmp_path, monkeypatch):
    monkeypatch.delenv("AUTOBENCH_EXPORT_ROOT", raising=False)
    with pytest.raises(SystemExit, match="export root is not set"):
        ce._resolve_export_root(None)


def test_unknown_roster_cohort_is_refused(tmp_path):
    """Forged violation: a roster naming a cohort outside the manifest."""
    repo = tmp_path / "repo"
    campaign = repo / "benchmarks/campaigns/preprint_130"
    campaign.mkdir(parents=True)
    (campaign / "manifest.json").write_text(json.dumps({
        "cells": [{"cell_id": CELL_ID, "dataset": "tcga_luad"}],
    }))
    (campaign / "active_roster.json").write_text(json.dumps({
        "cohorts": ["tcga_luad", "tcga_fake"], "cells": 1,
    }))
    with pytest.raises(ce.ExportError, match="absent from the manifest"):
        ce.roster_cell_ids(repo)


def test_roster_census_mismatch_is_refused(tmp_path):
    repo = tmp_path / "repo"
    campaign = repo / "benchmarks/campaigns/preprint_130"
    campaign.mkdir(parents=True)
    (campaign / "manifest.json").write_text(json.dumps({
        "cells": [{"cell_id": CELL_ID, "dataset": "tcga_luad"}],
    }))
    (campaign / "active_roster.json").write_text(json.dumps({
        "cohorts": ["tcga_luad"], "cells": 2,
    }))
    with pytest.raises(ce.ExportError, match="census mismatch"):
        ce.roster_cell_ids(repo)
