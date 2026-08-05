"""Publication report follows the frozen, dependency-aware analysis plan."""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

import pytest

from autobench.campaign import (
    AGENT_PROTOCOL_FILE,
    CAMPAIGN_ID,
    PROTOCOL_VERSION,
    content_sha256,
    file_sha256,
    load_manifest,
)
from autobench.campaign_analysis import (
    CampaignAnalysisError,
    _direction_only,
    _primary_values,
    _task_stratified_lift,
    _validated_process_evidence,
    build_publication_report,
    write_publication_report,
)
from autobench.campaign_stages import (
    CampaignStageError,
    SELECTION_FREEZE_SCHEMA_VERSION,
    certify_campaign,
    initialize_stage_state,
    validate_certification_bundle_artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "benchmarks/campaigns/preprint_130/manifest.json"
AGENT_PROTOCOL = {
    "schema_version": 2,
    "campaign_id": CAMPAIGN_ID,
    "purpose": "publication",
    "provider": "test-provider",
    "runtime": "test-runtime",
    "runtime_version": "test-runtime-1",
    "model": "test-model",
    "model_version": "test-model-1",
    "effort": "high",
    "network_access": "enabled",
    "fallback_model": None,
    "proposal_policy_content": "test proposal policy",
    "proposal_policy_sha256": hashlib.sha256(b"test proposal policy").hexdigest(),
    "toolset_content": "test toolset",
    "toolset_sha256": hashlib.sha256(b"test toolset").hexdigest(),
    "max_sessions_per_cell": 1,
}


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


def _missing_resources(count: int) -> dict:
    return {
        "elapsed_seconds": {
            "reported": 0,
            "missing": count,
            "maximum": None,
            "total": None,
            "gpu_attached_job_hours": None,
        },
        "peak_vram_mb": {
            "reported": 0,
            "missing": count,
            "maximum": None,
        },
    }


def _search_process(cell_id: str) -> dict:
    attempts = [
        {
            "node_id": f"node_{index + 1:04d}",
            "source_spec_sha256": f"{index + 1:064x}",
            "submitted_at": f"2026-08-04T00:{index:02d}:00+00:00",
            "agent_session_id": f"session-{cell_id}",
            "agent_session_binding_sha256": "b" * 64,
            "candidate_class": "config-only",
            "policy_hash": "c" * 64,
            "result_status": "crash",
            "termination_reason": "unspecified",
            "budget_killed": False,
            "outcome_class": "crash",
            "elapsed_seconds": None,
            "peak_vram_mb": None,
            "eligible": False,
            "reason": "fixture crash",
            "candidate_sha256": None,
            "validation_mean": None,
        }
        for index in range(60)
    ]
    return {
        "schema_version": 1,
        "baseline": {
            "folds": list(range(5)),
            "result_status": "completed",
            "resources": _missing_resources(1),
        },
        "discovery": {
            "attempt_budget": 60,
            "attempts_charged": 60,
            "baseline_validation_mean": 0.5,
            "complete_candidates": 0,
            "unique_complete_candidates": 0,
            "promoted_candidates": 0,
            "candidate_class_counts": {
                "config-only": 60,
                "train-only-source": 0,
                "inadmissible": 0,
            },
            "result_status_counts": {"crash": 60},
            "outcome_class_counts": {
                "completed": 0,
                "budget-killed": 0,
                "timeout": 0,
                "oom": 0,
                "cancelled": 0,
                "partial": 0,
                "crash": 60,
                "missing-result": 0,
                "unknown": 0
            },
            "attempts": attempts,
            "validation_anytime": [
                {
                    "attempt_index": index + 1,
                    "node_id": row["node_id"],
                    "result_status": "crash",
                    "outcome_class": "crash",
                    "eligible": False,
                    "validation_mean": None,
                    "running_best_candidate_id": "baseline",
                    "running_best_validation_mean": 0.5,
                }
                for index, row in enumerate(attempts)
            ],
            "resources": _missing_resources(60),
        },
        "promotion": {
            "candidate_budget": 10,
            "attempts_charged": 0,
            "status_counts": {"eligible": 0, "ineligible": 0},
            "outcome_class_counts": {
                "completed": 0,
                "budget-killed": 0,
                "timeout": 0,
                "oom": 0,
                "cancelled": 0,
                "partial": 0,
                "crash": 0,
                "missing-result": 0,
                "unknown": 0
            },
            "yield": None,
            "jobs": [],
            "resources": _missing_resources(0),
        },
    }


def _eligible_process(cell_id: str = "fixture-cell") -> dict:
    process = _search_process(cell_id)
    attempts = process["discovery"]["attempts"]
    candidates = (("d" * 64, 0.7), ("e" * 64, 0.6))
    for row, (candidate_sha256, validation_mean) in zip(
        attempts, candidates, strict=False,
    ):
        row.update({
            "result_status": "completed",
            "outcome_class": "completed",
            "eligible": True,
            "reason": "complete",
            "candidate_sha256": candidate_sha256,
            "validation_mean": validation_mean,
        })
    discovery = process["discovery"]
    discovery.update({
        "complete_candidates": 2,
        "unique_complete_candidates": 2,
        "promoted_candidates": 2,
        "result_status_counts": {"completed": 2, "crash": 58},
        "outcome_class_counts": {
            **discovery["outcome_class_counts"],
            "completed": 2,
            "crash": 58,
        },
    })
    best = 0.5
    best_id = "baseline"
    anytime = []
    for index, row in enumerate(attempts, 1):
        if row["eligible"] and row["validation_mean"] > best:
            best = row["validation_mean"]
            best_id = row["node_id"]
        anytime.append({
            "attempt_index": index,
            "node_id": row["node_id"],
            "result_status": row["result_status"],
            "outcome_class": row["outcome_class"],
            "eligible": row["eligible"],
            "validation_mean": row["validation_mean"],
            "running_best_candidate_id": best_id,
            "running_best_validation_mean": best,
        })
    discovery["validation_anytime"] = anytime
    jobs = []
    for rank, row in enumerate(attempts[:2], 1):
        promotion_identity = {
            "overlay_manifest": {},
            "deletions": [],
            "candidate_class": row["candidate_class"],
            "policy_hash": row["policy_hash"],
            "variant_selection_hash": None,
            "override_hash": None,
        }
        jobs.append({
            "rank": rank,
            "source_node_id": row["node_id"],
            "source_candidate_sha256": row["candidate_sha256"],
            "promotion_node_id": f"promotion_{rank:04d}",
            "promotion_candidate_sha256": content_sha256(promotion_identity),
            "promotion_identity": promotion_identity,
            "status": "eligible",
            "candidate_class": row["candidate_class"],
            "policy_hash": row["policy_hash"],
            "result_status": "completed",
            "source_spec_sha256": row["source_spec_sha256"],
            "promotion_spec_sha256": f"{rank:064x}",
            "submitted_at": f"2026-08-04T01:0{rank}:00+00:00",
            "elapsed_seconds": None,
            "peak_vram_mb": None,
            "validation_mean": 0.65,
            "termination_reason": "unspecified",
            "budget_killed": False,
            "outcome_class": "completed",
            "reason": "complete five-fold validation",
        })
    process["promotion"] = {
        "candidate_budget": 10,
        "attempts_charged": 2,
        "status_counts": {"eligible": 2, "ineligible": 0},
        "outcome_class_counts": {
            "completed": 2,
            "budget-killed": 0,
            "timeout": 0,
            "oom": 0,
            "cancelled": 0,
            "partial": 0,
            "crash": 0,
            "missing-result": 0,
            "unknown": 0,
        },
        "yield": 1.0,
        "jobs": jobs,
        "resources": _missing_resources(2),
    }
    return process


def _write_certified_state(
    runtime_root: Path, cell: dict, freeze_entry: dict, bundle: dict,
) -> None:
    state = initialize_stage_state(
        runtime_root / cell["cell_id"],
        cell=cell,
        manifest_sha256=file_sha256(MANIFEST),
    )
    process = freeze_entry["process_evidence"]
    metric_key = (
        "val_c_index" if cell["task_type"] == "survival" else "val_auc"
    )

    def validation_folds(value: float) -> list[dict]:
        return [
            {
                "fold_index": fold,
                "metrics": {metric_key: value},
                "composite": value,
            }
            for fold in range(5)
        ]

    promoted = [
        {
            "candidate_id": job["source_node_id"],
            "candidate_sha256": job["source_candidate_sha256"],
            "source_spec_sha256": job["source_spec_sha256"],
            "identity": {
                "candidate_class": job["candidate_class"],
                "policy_hash": job["policy_hash"],
            },
        }
        for job in process["promotion"]["jobs"]
    ]
    state["baseline"] = {
        "candidate_sha256": freeze_entry["baseline_candidate_sha256"],
        "validation_mean": freeze_entry["baseline_validation_mean"],
        "validation_folds": validation_folds(
            freeze_entry["baseline_validation_mean"]
        ),
        "sealed_fold_sha256": {
            filename: record["sha256"]
            for filename, record in freeze_entry["baseline_source_folds"].items()
        },
        "result_status": process["baseline"]["result_status"],
        "resources": process["baseline"]["resources"],
    }
    state["discovery"].update({
        "attempt_budget": process["discovery"]["attempt_budget"],
        "attempts_charged": process["discovery"]["attempts_charged"],
        "attempt_audit": process["discovery"]["attempts"],
        "complete_candidates": process["discovery"]["complete_candidates"],
        "unique_complete_candidates": process["discovery"][
            "unique_complete_candidates"
        ],
        "promoted_candidates": promoted,
    })
    state["promotion"].update({
        "jobs": process["promotion"]["jobs"],
        "attempts_charged": process["promotion"]["attempts_charged"],
        "frozen": True,
    })
    state["winner"] = {
        "kind": freeze_entry["winner_kind"],
        "candidate_id": freeze_entry["winner_candidate_id"],
        "candidate_sha256": freeze_entry["winner_candidate_sha256"],
        "promotion_node_id": freeze_entry["winner_promotion_node_id"],
        "validation_mean": freeze_entry["winner_validation_mean"],
        "validation_folds": validation_folds(
            freeze_entry["winner_validation_mean"]
        ),
        "sealed_fold_sha256": {
            filename: record["sha256"]
            for filename, record in freeze_entry["winner_source_folds"].items()
        },
        "selection_sha256": freeze_entry["selection_sha256"],
    }
    state["certification"] = {
        "bundle": "certification/certify.json",
        "bundle_sha256": bundle["bundle_sha256"],
        "certified_at": bundle["certified_at"],
        "selection_state_sha256": freeze_entry["state_sha256"],
    }
    state["phase"] = "certified"
    state["state_sha256"] = content_sha256({
        key: value for key, value in state.items() if key != "state_sha256"
    })
    (runtime_root / cell["cell_id"] / "campaign_state.json").write_text(
        json.dumps(state)
    )


def _certified_campaign(runtime_root: Path) -> dict[str, Path]:
    manifest = load_manifest(MANIFEST)
    runtime_root.mkdir(parents=True, exist_ok=True)
    (runtime_root / AGENT_PROTOCOL_FILE).write_text(json.dumps(AGENT_PROTOCOL))
    usage = {
        "status": "exact",
        "input_tokens": 100,
        "output_tokens": 50,
        "cached_input_tokens": 25,
        "cost_usd": 0.1,
        "basis": "test fixture",
    }
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
    freeze_entries = []
    for cell in manifest["cells"]:
        session_id = f"session-{cell['cell_id']}"
        session_binding = content_sha256({
            "campaign_id": CAMPAIGN_ID,
            "cell_id": cell["cell_id"],
            "agent_protocol_sha256": content_sha256(AGENT_PROTOCOL),
            "session_id": session_id,
            "started_at": "2026-08-04T00:00:00+00:00",
            "bound_at": "2026-08-04T00:00:01+00:00",
        })
        process = _eligible_process(cell["cell_id"])
        for attempt in process["discovery"]["attempts"]:
            attempt["agent_session_binding_sha256"] = session_binding
        first_job = process["promotion"]["jobs"][0]
        baseline_folds = _folds(
            cell["task_type"], framework_baseline[cell["framework"]],
        )
        winner_folds = _folds(
            cell["task_type"],
            framework_baseline[cell["framework"]]
            + framework_lift[cell["framework"]],
        )
        source_anchors = {}
        for label, folds in (
            ("baseline", baseline_folds), ("winner", winner_folds),
        ):
            anchors = {}
            source_dir = runtime_root / cell["cell_id"] / "source" / label
            source_dir.mkdir(parents=True)
            for fold in folds:
                filename = f"fold_{fold['fold_index']}_result.json"
                path = source_dir / filename
                path.write_text(json.dumps(fold))
                anchors[filename] = {
                    "path": path.relative_to(runtime_root).as_posix(),
                    "sha256": file_sha256(path),
                }
            source_anchors[label] = anchors
        baseline_candidate_sha256 = content_sha256({
            "baseline": cell["cell_id"]
        })
        freeze_entries.append({
            "cell_id": cell["cell_id"],
            "cell_sha256": cell["cell_sha256"],
            "state_sha256": content_sha256({"state": cell["cell_id"]}),
            "selection_sha256": content_sha256({"selection": cell["cell_id"]}),
            "winner_kind": "searched",
            "winner_candidate_id": first_job["source_node_id"],
            "winner_candidate_sha256": first_job["source_candidate_sha256"],
            "winner_promotion_node_id": first_job["promotion_node_id"],
            "winner_validation_mean": first_job["validation_mean"],
            "baseline_validation_mean": 0.5,
            "baseline_candidate_sha256": baseline_candidate_sha256,
            "winner_source_folds": source_anchors["winner"],
            "baseline_source_folds": source_anchors["baseline"],
            "agent_session_sha256": None,
            "agent_session_id": session_id,
            "agent_session_binding_sha256": session_binding,
            "agent_usage": usage,
            "process_evidence": process,
            "process_sha256": content_sha256(process),
        })
        session = {
            "schema_version": 2,
            "campaign_id": CAMPAIGN_ID,
            "cell_id": cell["cell_id"],
            "agent_protocol_sha256": content_sha256(AGENT_PROTOCOL),
            "status": "finalized",
            "session": {
                "session_id": session_id,
                "started_at": "2026-08-04T00:00:00+00:00",
                "bound_at": "2026-08-04T00:00:01+00:00",
                "ended_at": "2026-08-04T02:00:00+00:00",
                "termination_reason": "budget-complete",
                "usage": usage,
            },
            "binding_sha256": session_binding,
            "attestation_sha256": None,
        }
        session["attestation_sha256"] = content_sha256({
            key: value for key, value in session.items()
            if key != "attestation_sha256"
        })
        freeze_entries[-1]["agent_session_sha256"] = session[
            "attestation_sha256"
        ]
        cell_root = runtime_root / cell["cell_id"]
        (cell_root / "agent_session.json").write_text(json.dumps(session))
        config = cell_root / "automil/config.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(json.dumps({
            "campaign": {
                "agent_protocol_sha256": content_sha256(AGENT_PROTOCOL),
            },
        }))
    freeze = {
        "schema_version": SELECTION_FREEZE_SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "manifest_sha256": file_sha256(MANIFEST),
        "protocol_version": PROTOCOL_VERSION,
        "agent_protocol_sha256": content_sha256(AGENT_PROTOCOL),
        "roster_sha256": content_sha256(dict(sorted(
            (row["cell_id"], row["cell_sha256"]) for row in freeze_entries
        ))),
        "cell_count": len(freeze_entries),
        "cells": freeze_entries,
        "frozen_at": "2026-08-04T00:00:00+00:00",
    }
    freeze["freeze_sha256"] = content_sha256(freeze)
    (runtime_root / "selection_freeze.json").write_text(json.dumps(freeze))
    freeze_hash = freeze["freeze_sha256"]
    frozen_by_cell = {row["cell_id"]: row for row in freeze_entries}
    entries = []
    paths: dict[str, Path] = {}
    for cell in manifest["cells"]:
        freeze_entry = frozen_by_cell[cell["cell_id"]]
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
            "selection_state_sha256": frozen_by_cell[cell["cell_id"]][
                "state_sha256"
            ],
            "winner": {
                "kind": freeze_entry["winner_kind"],
                "candidate_id": freeze_entry["winner_candidate_id"],
                "candidate_sha256": freeze_entry["winner_candidate_sha256"],
                "promotion_node_id": freeze_entry["winner_promotion_node_id"],
            },
            "selection_sha256": freeze_entry["selection_sha256"],
            "validation_mean": freeze_entry["winner_validation_mean"],
            "baseline": {
                "candidate_id": "baseline",
                "candidate_sha256": freeze_entry["baseline_candidate_sha256"],
                "validation_mean": freeze_entry["baseline_validation_mean"],
            },
            "baseline_held_out_folds": baseline,
            "held_out_folds": winner,
            "baseline_held_out": _mean_metrics(baseline),
            "held_out": _mean_metrics(winner),
            "source_fold_sha256": {
                filename: record["sha256"]
                for filename, record in frozen_by_cell[cell["cell_id"]][
                    "winner_source_folds"
                ].items()
            },
            "baseline_source_fold_sha256": {
                filename: record["sha256"]
                for filename, record in frozen_by_cell[cell["cell_id"]][
                    "baseline_source_folds"
                ].items()
            },
            "paired_fold_deltas": [
                {
                    "fold_index": fold,
                    "held_out_delta": {
                        key: winner[fold]["held_out"][key]
                        - baseline[fold]["held_out"][key]
                        for key in baseline[fold]["held_out"]
                    },
                }
                for fold in range(5)
            ],
            "held_out_lift": {
                key: _mean_metrics(winner)[key] - _mean_metrics(baseline)[key]
                for key in _mean_metrics(baseline)
            },
            "retrained": False,
            "certified_at": "2026-08-04T00:00:00+00:00",
        }
        bundle["bundle_sha256"] = content_sha256(bundle)
        path = runtime_root / cell["cell_id"] / "certification/certify.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(bundle))
        _write_certified_state(runtime_root, cell, freeze_entry, bundle)
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


def _rehash_freeze_and_index(runtime_root: Path, freeze: dict) -> None:
    freeze["roster_sha256"] = content_sha256(dict(sorted(
        (row["cell_id"], row["cell_sha256"])
        for row in freeze.get("cells", [])
        if isinstance(row, dict)
        and isinstance(row.get("cell_id"), str)
        and isinstance(row.get("cell_sha256"), str)
    )))
    freeze["freeze_sha256"] = content_sha256({
        key: value for key, value in freeze.items() if key != "freeze_sha256"
    })
    (runtime_root / "selection_freeze.json").write_text(json.dumps(freeze))
    index_path = runtime_root / "campaign_certification.json"
    index = json.loads(index_path.read_text())
    index["selection_freeze_sha256"] = freeze["freeze_sha256"]
    index["certification_sha256"] = content_sha256({
        key: value for key, value in index.items()
        if key != "certification_sha256"
    })
    index_path.write_text(json.dumps(index))


def _resign_bundle_and_index(
    runtime_root: Path, bundle_path: Path, bundle: dict,
) -> None:
    bundle["bundle_sha256"] = content_sha256({
        key: value for key, value in bundle.items() if key != "bundle_sha256"
    })
    bundle_path.write_text(json.dumps(bundle))
    index_path = runtime_root / "campaign_certification.json"
    index = json.loads(index_path.read_text())
    cell_id = bundle["cell_id"]
    entry = next(row for row in index["cells"] if row["cell_id"] == cell_id)
    entry["bundle_sha256"] = bundle["bundle_sha256"]
    entry["file_sha256"] = file_sha256(bundle_path)
    index["certification_sha256"] = content_sha256({
        key: value for key, value in index.items()
        if key != "certification_sha256"
    })
    index_path.write_text(json.dumps(index))


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
    assert report["summaries"]["titan_by_task_type"]["classification"]["n"] == 5
    assert report["summaries"]["titan_by_task_type"]["survival"]["n"] == 5
    assert "all_cells" not in report["summaries"]
    assert set(report["summaries"]["agentic_lift"]["by_task_type"]) == {
        "classification", "survival",
    }
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


@pytest.mark.parametrize("mutation", ["alias", "sibling-cell"])
def test_report_rejects_rehashed_noncanonical_bundle_paths(tmp_path, mutation):
    runtime_root = tmp_path / "runtime"
    paths = _certified_campaign(runtime_root)
    index_path = runtime_root / "campaign_certification.json"
    index = json.loads(index_path.read_text())
    entry = index["cells"][0]
    original = paths[entry["cell_id"]]
    if mutation == "alias":
        target = runtime_root / "alternate" / entry["cell_id"] / "certify.json"
        target.parent.mkdir(parents=True)
        target.write_bytes(original.read_bytes())
    else:
        sibling = index["cells"][1]
        target = runtime_root / sibling["bundle"]
    entry["bundle"] = target.relative_to(runtime_root).as_posix()
    entry["file_sha256"] = file_sha256(target)
    index["certification_sha256"] = content_sha256({
        key: value for key, value in index.items()
        if key != "certification_sha256"
    })
    index_path.write_text(json.dumps(index))

    with pytest.raises(CampaignAnalysisError, match="path is not canonical"):
        build_publication_report(
            runtime_root=runtime_root, manifest_path=MANIFEST, repo_root=REPO_ROOT,
        )


def test_rehashed_index_cannot_predate_its_bundles(tmp_path):
    runtime_root = tmp_path / "runtime"
    _certified_campaign(runtime_root)
    index_path = runtime_root / "campaign_certification.json"
    index = json.loads(index_path.read_text())
    index["certified_at"] = "2000-01-01T00:00:00+00:00"
    index["certification_sha256"] = content_sha256({
        key: value for key, value in index.items()
        if key != "certification_sha256"
    })
    index_path.write_text(json.dumps(index))

    with pytest.raises(CampaignStageError, match="freeze/bundle/index order"):
        certify_campaign(runtime_root, MANIFEST)
    with pytest.raises(CampaignAnalysisError, match="freeze/bundle/index order"):
        build_publication_report(
            runtime_root=runtime_root, manifest_path=MANIFEST, repo_root=REPO_ROOT,
        )


def test_rehashed_bundle_cannot_predate_selection_freeze(tmp_path):
    runtime_root = tmp_path / "runtime"
    paths = _certified_campaign(runtime_root)
    bundle_path = next(iter(paths.values()))
    bundle = json.loads(bundle_path.read_text())
    bundle["certified_at"] = "2000-01-01T00:00:00+00:00"
    bundle["bundle_sha256"] = content_sha256({
        key: value for key, value in bundle.items() if key != "bundle_sha256"
    })
    bundle_path.write_text(json.dumps(bundle))
    state_path = runtime_root / bundle["cell_id"] / "campaign_state.json"
    state = json.loads(state_path.read_text())
    state["certification"]["certified_at"] = bundle["certified_at"]
    state["certification"]["bundle_sha256"] = bundle["bundle_sha256"]
    state["state_sha256"] = content_sha256({
        key: value for key, value in state.items() if key != "state_sha256"
    })
    state_path.write_text(json.dumps(state))
    index_path = runtime_root / "campaign_certification.json"
    index = json.loads(index_path.read_text())
    entry = next(
        row for row in index["cells"] if row["cell_id"] == bundle["cell_id"]
    )
    entry["bundle_sha256"] = bundle["bundle_sha256"]
    entry["file_sha256"] = file_sha256(bundle_path)
    index["certification_sha256"] = content_sha256({
        key: value for key, value in index.items()
        if key != "certification_sha256"
    })
    index_path.write_text(json.dumps(index))

    with pytest.raises(CampaignStageError, match="freeze/bundle/index order"):
        certify_campaign(runtime_root, MANIFEST)
    with pytest.raises(CampaignAnalysisError, match="freeze/bundle/index order"):
        build_publication_report(
            runtime_root=runtime_root, manifest_path=MANIFEST, repo_root=REPO_ROOT,
        )


def test_report_rejects_a_rehashed_duplicate_freeze_row(tmp_path):
    runtime_root = tmp_path / "runtime"
    _certified_campaign(runtime_root)
    freeze = json.loads((runtime_root / "selection_freeze.json").read_text())
    freeze["cells"].append(dict(freeze["cells"][0]))
    _rehash_freeze_and_index(runtime_root, freeze)

    with pytest.raises(CampaignAnalysisError, match="selection freeze is invalid"):
        build_publication_report(
            runtime_root=runtime_root, manifest_path=MANIFEST, repo_root=REPO_ROOT,
        )


def test_report_rejects_rehashed_invalid_agent_usage(tmp_path):
    runtime_root = tmp_path / "runtime"
    _certified_campaign(runtime_root)
    freeze = json.loads((runtime_root / "selection_freeze.json").read_text())
    first = min(freeze["cells"], key=lambda row: row["cell_id"])
    first["agent_usage"]["input_tokens"] = -1
    _rehash_freeze_and_index(runtime_root, freeze)

    with pytest.raises(CampaignAnalysisError, match="usage input_tokens"):
        build_publication_report(
            runtime_root=runtime_root, manifest_path=MANIFEST, repo_root=REPO_ROOT,
        )


def test_resigned_held_out_metric_drift_is_rejected_everywhere(tmp_path):
    runtime_root = tmp_path / "runtime"
    paths = _certified_campaign(runtime_root)
    bundle_path = next(
        path for path in paths.values()
        if "test_auc" in json.loads(path.read_text())["held_out"]
    )
    bundle = json.loads(bundle_path.read_text())
    bundle["held_out_folds"][0]["held_out"]["test_auc"] += 0.1
    bundle["held_out"] = _mean_metrics(bundle["held_out_folds"])
    bundle["paired_fold_deltas"] = [
        {
            "fold_index": fold,
            "held_out_delta": {
                key: bundle["held_out_folds"][fold]["held_out"][key]
                - bundle["baseline_held_out_folds"][fold]["held_out"][key]
                for key in bundle["held_out"]
            },
        }
        for fold in range(5)
    ]
    bundle["held_out_lift"] = {
        key: bundle["held_out"][key] - bundle["baseline_held_out"][key]
        for key in bundle["held_out"]
    }
    _resign_bundle_and_index(runtime_root, bundle_path, bundle)

    with pytest.raises(CampaignStageError, match="anchored source folds"):
        certify_campaign(runtime_root, MANIFEST)
    with pytest.raises(CampaignAnalysisError, match="source evidence"):
        build_publication_report(
            runtime_root=runtime_root, manifest_path=MANIFEST, repo_root=REPO_ROOT,
        )


def test_coherent_downstream_rehash_cannot_override_certified_state(tmp_path):
    runtime_root = tmp_path / "runtime"
    paths = _certified_campaign(runtime_root)
    target_path = next(
        path for path in paths.values()
        if "test_auc" in json.loads(path.read_text())["held_out"]
    )
    target_bundle = json.loads(target_path.read_text())
    target_id = target_bundle["cell_id"]
    freeze_path = runtime_root / "selection_freeze.json"
    freeze = json.loads(freeze_path.read_text())
    freeze_entry = next(
        row for row in freeze["cells"] if row["cell_id"] == target_id
    )
    source_record = freeze_entry["winner_source_folds"]["fold_0_result.json"]
    source_path = runtime_root / source_record["path"]
    source = json.loads(source_path.read_text())
    source["held_out"]["test_auc"] += 0.1
    source_path.write_text(json.dumps(source))
    source_record["sha256"] = file_sha256(source_path)
    freeze["freeze_sha256"] = content_sha256({
        key: value for key, value in freeze.items() if key != "freeze_sha256"
    })
    freeze_path.write_text(json.dumps(freeze))

    index_path = runtime_root / "campaign_certification.json"
    index = json.loads(index_path.read_text())
    indexed = {row["cell_id"]: row for row in index["cells"]}
    for cell_id, bundle_path in paths.items():
        bundle = json.loads(bundle_path.read_text())
        bundle["selection_freeze_sha256"] = freeze["freeze_sha256"]
        if cell_id == target_id:
            winner_folds = [
                json.loads((runtime_root / record["path"]).read_text())
                for _, record in sorted(
                    freeze_entry["winner_source_folds"].items()
                )
            ]
            bundle["held_out_folds"] = winner_folds
            bundle["held_out"] = _mean_metrics(winner_folds)
            bundle["source_fold_sha256"] = {
                filename: record["sha256"]
                for filename, record in freeze_entry[
                    "winner_source_folds"
                ].items()
            }
            bundle["paired_fold_deltas"] = [
                {
                    "fold_index": fold,
                    "held_out_delta": {
                        key: winner_folds[fold]["held_out"][key]
                        - bundle["baseline_held_out_folds"][fold]["held_out"][key]
                        for key in bundle["held_out"]
                    },
                }
                for fold in range(5)
            ]
            bundle["held_out_lift"] = {
                key: bundle["held_out"][key] - bundle["baseline_held_out"][key]
                for key in bundle["held_out"]
            }
        bundle["bundle_sha256"] = content_sha256({
            key: value for key, value in bundle.items()
            if key != "bundle_sha256"
        })
        bundle_path.write_text(json.dumps(bundle))
        indexed[cell_id]["bundle_sha256"] = bundle["bundle_sha256"]
        indexed[cell_id]["file_sha256"] = file_sha256(bundle_path)
    index["selection_freeze_sha256"] = freeze["freeze_sha256"]
    index["certification_sha256"] = content_sha256({
        key: value for key, value in index.items()
        if key != "certification_sha256"
    })
    index_path.write_text(json.dumps(index))

    with pytest.raises(CampaignStageError, match="certified cell state"):
        certify_campaign(runtime_root, MANIFEST)
    with pytest.raises(CampaignAnalysisError, match="certified cell state"):
        build_publication_report(
            runtime_root=runtime_root, manifest_path=MANIFEST, repo_root=REPO_ROOT,
        )


def test_resigned_baseline_identity_drift_is_rejected_everywhere(tmp_path):
    runtime_root = tmp_path / "runtime"
    paths = _certified_campaign(runtime_root)
    bundle_path = next(iter(paths.values()))
    bundle = json.loads(bundle_path.read_text())
    bundle["baseline"]["candidate_sha256"] = "1" * 64
    _resign_bundle_and_index(runtime_root, bundle_path, bundle)

    with pytest.raises(CampaignStageError, match="frozen validation winner"):
        certify_campaign(runtime_root, MANIFEST)
    with pytest.raises(CampaignAnalysisError, match="frozen validation winner"):
        build_publication_report(
            runtime_root=runtime_root, manifest_path=MANIFEST, repo_root=REPO_ROOT,
        )


def test_resigned_promotion_identity_drift_is_rejected_everywhere(tmp_path):
    runtime_root = tmp_path / "runtime"
    paths = _certified_campaign(runtime_root)
    bundle_path = next(iter(paths.values()))
    bundle = json.loads(bundle_path.read_text())
    bundle["winner"]["promotion_node_id"] = "forged-promotion"
    _resign_bundle_and_index(runtime_root, bundle_path, bundle)

    with pytest.raises(CampaignStageError, match="frozen validation winner"):
        certify_campaign(runtime_root, MANIFEST)
    with pytest.raises(CampaignAnalysisError, match="frozen validation winner"):
        build_publication_report(
            runtime_root=runtime_root, manifest_path=MANIFEST, repo_root=REPO_ROOT,
        )


def test_process_evidence_rejects_an_incomplete_attempt_census():
    process = _search_process("fixture-cell")
    process["discovery"]["attempts"].pop()

    with pytest.raises(CampaignAnalysisError, match="discovery census"):
        _validated_process_evidence(
            process, content_sha256(process), "fixture-cell",
        )


def test_process_evidence_recomputes_completion_counts():
    process = _search_process("fixture-cell")
    process["discovery"]["complete_candidates"] = 1

    with pytest.raises(CampaignAnalysisError, match="summaries do not reconcile"):
        _validated_process_evidence(
            process, content_sha256(process), "fixture-cell",
        )


def test_process_evidence_rejects_eligible_discovery_crash():
    process = _eligible_process()
    process["discovery"]["attempts"][0].update({
        "result_status": "crash",
        "outcome_class": "crash",
    })

    with pytest.raises(CampaignAnalysisError, match="eligible discovery attempt"):
        _validated_process_evidence(
            process, content_sha256(process), "fixture-cell",
        )


def test_process_evidence_accepts_a_consistent_eligible_promotion_chain():
    process = _eligible_process()

    assert _validated_process_evidence(
        process, content_sha256(process), "fixture-cell",
    ) == process


def test_process_evidence_rejects_reclassified_terminal_outcome():
    process = _search_process("fixture-cell")
    process["discovery"]["attempts"][0]["outcome_class"] = "oom"

    with pytest.raises(CampaignAnalysisError, match="discovery attempt value drift"):
        _validated_process_evidence(
            process, content_sha256(process), "fixture-cell",
        )


def test_process_evidence_rejects_mixed_negative_resource_values():
    process = _search_process("fixture-cell")
    attempts = process["discovery"]["attempts"]
    attempts[0]["elapsed_seconds"] = 100.0
    attempts[1]["elapsed_seconds"] = 50.0
    attempts[2]["elapsed_seconds"] = -1.0
    process["discovery"]["resources"]["elapsed_seconds"] = {
        "reported": 3,
        "missing": 57,
        "maximum": 100.0,
        "total": 149.0,
        "gpu_attached_job_hours": 149.0 / 3600,
    }

    with pytest.raises(CampaignAnalysisError, match="elapsed_seconds is invalid"):
        _validated_process_evidence(
            process, content_sha256(process), "fixture-cell",
        )


def test_process_evidence_recomputes_exact_top10_promotion_sources():
    process = _eligible_process()
    process["promotion"]["jobs"][0]["source_node_id"] = "node_0059"

    with pytest.raises(CampaignAnalysisError, match="promotion job value drift"):
        _validated_process_evidence(
            process, content_sha256(process), "fixture-cell",
        )


def test_process_evidence_rejects_incomplete_eligible_promotion():
    process = _eligible_process()
    process["promotion"]["jobs"][0].update({
        "result_status": "crash",
        "outcome_class": "crash",
        "validation_mean": None,
    })

    with pytest.raises(CampaignAnalysisError, match="eligible promotion job"):
        _validated_process_evidence(
            process, content_sha256(process), "fixture-cell",
        )


@pytest.mark.parametrize(
    "field", ["promotion_node_id", "promotion_spec_sha256"],
)
def test_process_evidence_rejects_duplicate_promotion_identity(field):
    process = _eligible_process()
    jobs = process["promotion"]["jobs"]
    jobs[1][field] = jobs[0][field]

    with pytest.raises(CampaignAnalysisError, match="identities are not unique"):
        _validated_process_evidence(
            process, content_sha256(process), "fixture-cell",
        )


def test_lift_magnitudes_never_pool_classification_and_survival():
    cells = [
        {"task_type": "classification", "framework": "clam", "primary_lift": 0.9},
        {"task_type": "survival", "framework": "clam", "primary_lift": 0.1},
    ]

    stratified = _task_stratified_lift(cells, "framework")

    assert stratified["classification"]["clam"]["mean"] == pytest.approx(0.9)
    assert stratified["survival"]["clam"]["mean"] == pytest.approx(0.1)
    assert set(_direction_only(cell["primary_lift"] for cell in cells)) == {
        "n", "positive", "zero", "negative",
    }


def test_held_out_metrics_must_remain_in_the_unit_interval():
    baseline = _folds("classification", 0.5)
    winner = _folds("classification", 0.6)
    winner[0]["held_out"]["test_auc"] = 1.01
    bundle = {
        "baseline_held_out_folds": baseline,
        "held_out_folds": winner,
        "baseline_held_out": _mean_metrics(baseline),
        "held_out": _mean_metrics(winner),
    }

    with pytest.raises(CampaignAnalysisError, match=r"must be in \[0, 1\]"):
        _primary_values(bundle, "classification", "fixture-cell")


def test_certification_bundle_rejects_false_baseline_winner_lift(tmp_path):
    runtime_root = tmp_path / "runtime"
    paths = _certified_campaign(runtime_root)
    bundle = json.loads(next(iter(paths.values())).read_text())
    bundle["winner"] = {
        "kind": "baseline",
        "candidate_id": "baseline",
        "candidate_sha256": bundle["baseline"]["candidate_sha256"],
        "promotion_node_id": None,
    }
    bundle["bundle_sha256"] = content_sha256({
        key: value for key, value in bundle.items() if key != "bundle_sha256"
    })

    with pytest.raises(CampaignStageError, match="baseline winner evidence"):
        validate_certification_bundle_artifact(bundle)


def test_certification_bundle_recomputes_paired_fold_deltas(tmp_path):
    runtime_root = tmp_path / "runtime"
    paths = _certified_campaign(runtime_root)
    bundle = json.loads(next(iter(paths.values())).read_text())
    metric = next(iter(bundle["paired_fold_deltas"][0]["held_out_delta"]))
    bundle["paired_fold_deltas"][0]["held_out_delta"][metric] += 0.1
    bundle["bundle_sha256"] = content_sha256({
        key: value for key, value in bundle.items() if key != "bundle_sha256"
    })

    with pytest.raises(CampaignStageError, match="paired deltas"):
        validate_certification_bundle_artifact(bundle)


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

    with pytest.raises(CampaignAnalysisError, match="metric schema is not locked"):
        build_publication_report(
            runtime_root=runtime_root,
            manifest_path=MANIFEST,
            repo_root=REPO_ROOT,
        )
