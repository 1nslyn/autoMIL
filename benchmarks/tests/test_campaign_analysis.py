"""Publication report follows the frozen, dependency-aware analysis plan."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from autobench.campaign import CAMPAIGN_ID, content_sha256, file_sha256, load_manifest
from autobench.campaign_analysis import (
    CampaignAnalysisError,
    build_publication_report,
    write_publication_report,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "benchmarks/campaigns/preprint_130/manifest.json"


def _mean_metrics(folds: list[dict]) -> dict[str, float]:
    keys = folds[0]["held_out"]
    return {
        key: math.fsum(row["held_out"][key] for row in folds) / len(folds)
        for key in keys
    }


def _folds(task_type: str, primary: float) -> list[dict]:
    rows = []
    for fold in range(5):
        value = primary + fold / 1000
        held_out = (
            {"test_auc": value + 0.01, "test_bacc": value - 0.01}
            if task_type == "classification"
            else {"test_c_index": value}
        )
        rows.append({"fold_index": fold, "held_out": held_out})
    return rows


def _certified_campaign(runtime_root: Path) -> dict[str, Path]:
    manifest = load_manifest(MANIFEST)
    usage = {
        "status": "exact",
        "input_tokens": 100,
        "output_tokens": 50,
        "cached_input_tokens": 25,
        "cost_usd": 0.1,
        "basis": "test fixture",
    }
    freeze_entries = [
        {
            "cell_id": cell["cell_id"],
            "agent_session_sha256": "f" * 64,
            "agent_usage": usage,
        }
        for cell in manifest["cells"]
    ]
    freeze = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "manifest_sha256": file_sha256(MANIFEST),
        "protocol_sha256": "c" * 64,
        "agent_protocol_sha256": "d" * 64,
        "base_commit": "b" * 40,
        "cell_count": len(freeze_entries),
        "cells": freeze_entries,
        "frozen_at": "2026-08-04T00:00:00+00:00",
    }
    freeze["freeze_sha256"] = content_sha256(freeze)
    (runtime_root / "selection_freeze.json").parent.mkdir(parents=True)
    (runtime_root / "selection_freeze.json").write_text(json.dumps(freeze))
    freeze_hash = freeze["freeze_sha256"]
    framework_baseline = {
        "clam": 0.50,
        "nnmil": 0.51,
        "abmil": 0.52,
        "dtfd": 0.53,
        "titan": 0.515,
    }
    framework_lift = {
        "clam": 0.08,
        "nnmil": 0.04,
        "abmil": 0.01,
        "dtfd": -0.01,
        "titan": 0.02,
    }
    entries = []
    paths: dict[str, Path] = {}
    for cell in manifest["cells"]:
        baseline = _folds(
            cell["task_type"], framework_baseline[cell["framework"]],
        )
        winner = _folds(
            cell["task_type"],
            framework_baseline[cell["framework"]]
            + framework_lift[cell["framework"]],
        )
        bundle = {
            "schema_version": 2,
            "campaign_id": CAMPAIGN_ID,
            "cell_id": cell["cell_id"],
            "selection_freeze_sha256": freeze_hash,
            "winner": {"kind": "searched"},
            "baseline_held_out_folds": baseline,
            "held_out_folds": winner,
            "baseline_held_out": _mean_metrics(baseline),
            "held_out": _mean_metrics(winner),
        }
        bundle["bundle_sha256"] = content_sha256(bundle)
        path = runtime_root / cell["cell_id"] / "certification/certify.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(bundle))
        paths[cell["cell_id"]] = path
        entries.append({
            "cell_id": cell["cell_id"],
            "bundle": path.relative_to(runtime_root).as_posix(),
            "bundle_sha256": bundle["bundle_sha256"],
            "file_sha256": file_sha256(path),
        })
    index = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "manifest_sha256": file_sha256(MANIFEST),
        "selection_freeze_sha256": freeze["freeze_sha256"],
        "cell_count": len(entries),
        "cells": entries,
        "certified_at": "2026-08-04T00:00:00+00:00",
    }
    index["certification_sha256"] = content_sha256(index)
    (runtime_root / "campaign_certification.json").write_text(json.dumps(index))
    return paths


def test_report_contains_complete_lift_and_cross_arm_ranking_estimands(tmp_path):
    runtime_root = tmp_path / "runtime"
    _certified_campaign(runtime_root)

    report = write_publication_report(
        runtime_root=runtime_root,
        manifest_path=MANIFEST,
        repo_root=REPO_ROOT,
    )

    assert report["cell_count"] == 130
    assert len(report["cells"]) == 130
    assert len(report["tile_ranking_blocks"]) == 30
    assert report["summaries"]["tile_ranking_response"]["classification"][
        "blocks"
    ] == 15
    assert report["summaries"]["tile_ranking_response"]["survival"][
        "blocks"
    ] == 15
    assert report["summaries"]["titan"]["n"] == 10
    assert report["summaries"]["agent_resources"]["input_tokens"] == {
        "reported_cells": 130,
        "total": 13000.0,
    }
    assert all(
        block["top_arm_set_changed"]
        for block in report["tile_ranking_blocks"]
    )
    assert (runtime_root / "publication_report.json").is_file()


def test_report_fails_closed_if_one_of_130_certifications_is_missing(tmp_path):
    runtime_root = tmp_path / "runtime"
    paths = _certified_campaign(runtime_root)
    next(iter(paths.values())).unlink()

    with pytest.raises(CampaignAnalysisError, match="cannot hash"):
        build_publication_report(
            runtime_root=runtime_root,
            manifest_path=MANIFEST,
            repo_root=REPO_ROOT,
        )


def test_report_rejects_pooled_survival_substitutes(tmp_path):
    runtime_root = tmp_path / "runtime"
    paths = _certified_campaign(runtime_root)
    manifest = load_manifest(MANIFEST)
    survival_id = next(
        cell["cell_id"] for cell in manifest["cells"]
        if cell["task_type"] == "survival"
    )
    path = paths[survival_id]
    bundle = json.loads(path.read_text())
    for row in bundle["held_out_folds"]:
        row["held_out"] = {"pooled_test_c_index": 0.99}
    bundle["bundle_sha256"] = content_sha256({
        key: value for key, value in bundle.items() if key != "bundle_sha256"
    })
    path.write_text(json.dumps(bundle))
    index_path = runtime_root / "campaign_certification.json"
    index = json.loads(index_path.read_text())
    entry = next(row for row in index["cells"] if row["cell_id"] == survival_id)
    entry["bundle_sha256"] = bundle["bundle_sha256"]
    entry["file_sha256"] = file_sha256(path)
    index["certification_sha256"] = content_sha256({
        key: value for key, value in index.items()
        if key != "certification_sha256"
    })
    index_path.write_text(json.dumps(index))

    with pytest.raises(CampaignAnalysisError, match="lacks.*test_c_index"):
        build_publication_report(
            runtime_root=runtime_root,
            manifest_path=MANIFEST,
            repo_root=REPO_ROOT,
        )
