"""Frozen 130-cell campaign census, identity, and state-isolation contract."""
from __future__ import annotations

import json
import hashlib
import shlex
import shutil
from pathlib import Path

import pytest
import yaml

from automil.cells.state import make_cell_id, normalize_mil_model
from autobench.campaign import (
    ANALYSIS_PLAN_PATH,
    BASELINE_FOLDS,
    CANARY_AGENT_PROTOCOL,
    CERTIFICATION_FOLDS,
    CAMPAIGN_ID,
    DATASETS,
    PROTOCOL,
    PROTOCOL_VERSION,
    CampaignManifestError,
    build_preprint_manifest,
    content_sha256,
    file_sha256,
    load_manifest,
    materialize_discovery_cells,
    run_materialization_canary,
    validate_manifest,
    write_manifest,
)
from autobench.campaign_stages import CampaignStageError, freeze_campaign_selections

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "benchmarks/campaigns/preprint_130/manifest.json"
AGENT_PROTOCOL = {
    "schema_version": 1,
    "campaign_id": CAMPAIGN_ID,
    "purpose": "publication",
    "provider": "test-provider",
    "runtime": "test-runtime",
    "runtime_version": "test-runtime-1",
    "model": "test-model",
    "model_version": "test-model-1",
    "proposal_policy_content": "test proposal policy",
    "proposal_policy_sha256": hashlib.sha256(
        b"test proposal policy"
    ).hexdigest(),
    "toolset_content": "test toolset",
    "toolset_sha256": hashlib.sha256(b"test toolset").hexdigest(),
    "max_sessions_per_cell": 1,
}


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(MANIFEST)


def test_checked_in_manifest_is_exactly_reproducible(manifest):
    assert manifest == build_preprint_manifest(REPO_ROOT)


def test_manifest_has_130_unique_budget_and_run_identities(manifest):
    cells = manifest["cells"]
    assert len(cells) == 130
    assert len({cell["cell_id"] for cell in cells}) == 130
    assert len({cell["budget_identity"]["cell_id"] for cell in cells}) == 130
    identities = [cell["identity"] for cell in cells]
    assert all(set(identity) == {
        "dataset", "task", "encoder", "arm", "seed", "protocol_version",
    } for identity in identities)
    assert all(identity["protocol_version"] == PROTOCOL_VERSION
               for identity in identities)
    assert len({tuple(identity.values()) for identity in identities}) == 130


def test_manifest_locks_the_predeclared_analysis_plan(manifest):
    plan = REPO_ROOT / ANALYSIS_PLAN_PATH
    assert manifest["analysis_plan"] == {
        "path": ANALYSIS_PLAN_PATH,
        "sha256": file_sha256(plan),
    }
    payload = json.loads(plan.read_text())
    assert payload["status"] == "frozen-before-held-out-certification"
    assert payload["inference"]["confirmatory_p_values"] is False
    assert payload["missingness"]["expected_cells"] == 130


def test_manifest_census_matches_the_frozen_paper_roster(manifest):
    cells = manifest["cells"]
    assert {dataset: sum(c["dataset"] == dataset for c in cells)
            for dataset in DATASETS} == {dataset: 26 for dataset in DATASETS}
    assert sum(c["task_type"] == "classification" for c in cells) == 65
    assert sum(c["task_type"] == "survival" for c in cells) == 65
    assert sum(c["regime"] == "tile" for c in cells) == 120
    assert sum(c["regime"] == "slide" for c in cells) == 10


def test_every_command_and_budget_identity_reconstructs_from_the_cell(manifest):
    for cell in manifest["cells"]:
        assert manifest["dataset_sources"][cell["dataset_config"]] == (
            cell["dataset_config_sha256"]
        )
        assert manifest["policy_sources"][cell["policy_template"]] == (
            cell["policy_template_sha256"]
        )
        budget = cell["budget_identity"]
        assert budget["cell_id"] == make_cell_id(
            cell["dataset"], cell["encoder"],
            normalize_mil_model(cell["model"]), cell["task"],
        )
        command_folds = {
            "baseline": PROTOCOL["baseline"]["folds"],
            **PROTOCOL["stage_folds"],
        }
        for stage, folds in command_folds.items():
            tokens = shlex.split(cell["commands"][stage])
            values = {tokens[i]: tokens[i + 1] for i in range(len(tokens) - 1)
                      if tokens[i].startswith("--") and tokens[i] != "--no_wandb"}
            assert values["--dataset"] == cell["dataset"]
            assert values["--task"] == cell["task"]
            assert values["--encoder"] == cell["encoder"]
            assert values["--model"] == cell["model"]
            assert values["--framework"] == cell["framework"]
            assert values["--seed"] == str(cell["seed"])
            assert values["--n_folds"] == "5"
            assert values["--folds"] == ",".join(map(str, folds))


def test_protocol_uses_all_five_validation_folds_without_final_retraining(manifest):
    protocol = manifest["protocol"]
    assert protocol["stage_folds"] == {
        "discovery": [0, 1, 2],
        "promotion": [3, 4],
    }
    assert protocol["baseline"] == {
        "folds": list(BASELINE_FOLDS),
        "incumbent": True,
        "counts_toward_agentic_budget": False,
    }
    assert set(protocol["stage_folds"]["discovery"]).isdisjoint(
        protocol["stage_folds"]["promotion"]
    )
    assert sorted(
        protocol["stage_folds"]["discovery"]
        + protocol["stage_folds"]["promotion"]
    ) == list(CERTIFICATION_FOLDS)
    assert protocol["winner_selection"] == {
        "metric_source": "validation",
        "aggregation": "mean",
        "folds": list(CERTIFICATION_FOLDS),
    }
    assert protocol["certification"] == {
        "mode": "unseal-existing-held-out",
        "folds": list(CERTIFICATION_FOLDS),
        "retrain": False,
    }
    assert protocol["agentic_fold_trainings_per_cell"] == {
        "discovery": 60 * 3,
        "promotion_per_candidate": 2,
        "promotion_candidates_min": 0,
        "promotion_candidates_max": 10,
        "minimum": 180,
        "maximum": 200,
    }
    assert protocol["attempt_timeout"] == {
        "minutes": 360,
        "role": "failure-containment-not-search-budget",
        "scope": "one-multi-fold-attempt",
    }
    assert all(set(cell["commands"]) == {"baseline", "discovery", "promotion"}
               for cell in manifest["cells"])


def test_manifest_lock_detects_byte_tampering(tmp_path):
    path = tmp_path / "manifest.json"
    write_manifest(build_preprint_manifest(REPO_ROOT), path)
    path.write_text(path.read_text().replace('"seed": 42', '"seed": 7', 1))
    with pytest.raises(CampaignManifestError, match="protocol|hash"):
        load_manifest(path)


def test_materializer_rejects_policy_boundary_drift(tmp_path):
    fake_repo = tmp_path / "repo"
    _copy_campaign_sources(fake_repo)
    manifest_path = fake_repo / "benchmarks/campaigns/preprint_130/manifest.json"
    write_manifest(build_preprint_manifest(fake_repo), manifest_path)
    template = (
        fake_repo / "benchmarks/experiments" / DATASETS[0]
        / "automil/config.yaml"
    )
    template.write_text(template.read_text() + "\n# unauthorized drift\n")
    with pytest.raises(CampaignManifestError, match="policy template drift"):
        materialize_discovery_cells(
            manifest_path,
            fake_repo / "benchmarks/campaigns/preprint_130/runtime",
            fake_repo,
            agent_protocol=AGENT_PROTOCOL,
        )


def _copy_campaign_sources(fake_repo: Path) -> None:
    plan_dst = fake_repo / "benchmarks/campaigns/preprint_130/analysis_plan.json"
    plan_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        REPO_ROOT / "benchmarks/campaigns/preprint_130/analysis_plan.json",
        plan_dst,
    )
    for dataset in DATASETS:
        source_group = "cptac" if dataset.startswith("cptac_") else "tcga"
        dataset_src = REPO_ROOT / "benchmarks/datasets" / source_group / f"{dataset}.yaml"
        dataset_dst = fake_repo / "benchmarks/datasets" / source_group / f"{dataset}.yaml"
        dataset_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dataset_src, dataset_dst)
        template_src = REPO_ROOT / "benchmarks/experiments" / dataset / "automil/config.yaml"
        template_dst = fake_repo / "benchmarks/experiments" / dataset / "automil/config.yaml"
        template_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(template_src, template_dst)


def test_materializer_creates_130_independent_discovery_states(tmp_path):
    fake_repo = tmp_path / "repo"
    _copy_campaign_sources(fake_repo)
    manifest_path = fake_repo / "benchmarks/campaigns/preprint_130/manifest.json"
    write_manifest(build_preprint_manifest(fake_repo), manifest_path)
    output_root = fake_repo / "benchmarks/campaigns/preprint_130/runtime"
    roots = materialize_discovery_cells(
        manifest_path, output_root, fake_repo,
        agent_protocol=AGENT_PROTOCOL,
    )

    assert len(roots) == 130
    assert len(set(roots)) == 130
    assert all((root / "config.yaml").exists() for root in roots)
    assert all((root / "campaign_cell.json").exists() for root in roots)
    assert not any((root / "graph.json").exists() for root in roots)
    assert json.loads((output_root / "agent_protocol.json").read_text()) == AGENT_PROTOCOL

    for root in (roots[0], roots[64], roots[-1]):
        config = yaml.safe_load((root / "config.yaml").read_text())
        cell = json.loads((root / "campaign_cell.json").read_text())
        assert config["run"]["command"] == cell["commands"]["discovery"]
        assert config["run"]["mil_model"] == cell["model"]
        assert config["project"]["name"] == cell["budget_identity"]["dataset"]
        assert config["task"]["name"] == cell["budget_identity"]["task"]
        assert config["encoders"]["primary"] == cell["budget_identity"]["encoder"]
        assert config["campaign"]["budget_cell_id"] == cell["budget_identity"]["cell_id"]
        assert config["campaign"]["manifest_sha256"] == file_sha256(manifest_path)
        assert config["campaign"]["agent_protocol_sha256"] == content_sha256(
            AGENT_PROTOCOL
        )
        assert config["campaign"]["protocol_version"] == PROTOCOL_VERSION
        assert "base_commit" not in config["campaign"]
        assert config["cap"]["eval_budget"] == 60
        assert config["training"]["fold_count"] == 3
        assert config["orchestrator"]["default_timeout_min"] == 360
        assert config["files"]["editable"] == [
            f"{root.relative_to(fake_repo).as_posix()}/variants/_policies/*.py"
        ]


def test_materializer_rejects_unresolvable_agent_policy_hashes(tmp_path):
    fake_repo = tmp_path / "repo"
    _copy_campaign_sources(fake_repo)
    manifest_path = fake_repo / "benchmarks/campaigns/preprint_130/manifest.json"
    write_manifest(build_preprint_manifest(fake_repo), manifest_path)
    bad_protocol = {
        **AGENT_PROTOCOL,
        "proposal_policy_content": "different instructions",
    }

    with pytest.raises(CampaignManifestError, match="content/hash binding mismatch"):
        materialize_discovery_cells(
            manifest_path,
            fake_repo / "benchmarks/campaigns/preprint_130/runtime",
            fake_repo,
            agent_protocol=bad_protocol,
        )


def test_canary_agent_protocol_cannot_enter_publication_freeze(tmp_path):
    fake_repo = tmp_path / "repo"
    _copy_campaign_sources(fake_repo)
    manifest_path = fake_repo / "benchmarks/campaigns/preprint_130/manifest.json"
    write_manifest(build_preprint_manifest(fake_repo), manifest_path)
    output_root = fake_repo / "benchmarks/campaigns/preprint_130/runtime"
    materialize_discovery_cells(
        manifest_path,
        output_root,
        fake_repo,
        agent_protocol=CANARY_AGENT_PROTOCOL,
        allow_canary_protocol=True,
    )

    with pytest.raises(CampaignStageError, match="purpose is not allowed"):
        freeze_campaign_selections(output_root, manifest_path)


def test_materializer_restart_preserves_progress_files(tmp_path):
    fake_repo = tmp_path / "repo"
    _copy_campaign_sources(fake_repo)
    manifest_path = fake_repo / "benchmarks/campaigns/preprint_130/manifest.json"
    write_manifest(build_preprint_manifest(fake_repo), manifest_path)
    output_root = fake_repo / "benchmarks/campaigns/preprint_130/runtime"
    roots = materialize_discovery_cells(
        manifest_path, output_root, fake_repo,
        agent_protocol=AGENT_PROTOCOL,
    )
    root = roots[0]
    plan = root / "plan.md"
    learnings = root / "learnings.md"
    state = root.parent / "campaign_state.json"
    plan.write_text("agent plan must survive\n")
    learnings.write_text("earned knowledge must survive\n")
    state_before = state.read_bytes()

    restarted = materialize_discovery_cells(
        manifest_path, output_root, fake_repo,
        agent_protocol=AGENT_PROTOCOL,
    )

    assert restarted == roots
    assert plan.read_text() == "agent plan must survive\n"
    assert learnings.read_text() == "earned knowledge must survive\n"
    assert state.read_bytes() == state_before

    with pytest.raises(CampaignManifestError, match="different agent protocol"):
        materialize_discovery_cells(
            manifest_path, output_root, fake_repo,
            agent_protocol={**AGENT_PROTOCOL, "model_version": "test-model-2"},
        )


def test_materializer_refuses_half_created_existing_root(tmp_path):
    fake_repo = tmp_path / "repo"
    _copy_campaign_sources(fake_repo)
    manifest_path = fake_repo / "benchmarks/campaigns/preprint_130/manifest.json"
    write_manifest(build_preprint_manifest(fake_repo), manifest_path)
    output_root = fake_repo / "benchmarks/campaigns/preprint_130/runtime"
    first_cell = load_manifest(manifest_path)["cells"][0]
    partial = output_root / first_cell["cell_id"] / "automil"
    partial.mkdir(parents=True)
    (partial / "plan.md").write_text("do not overwrite me\n")

    with pytest.raises(CampaignManifestError, match="incomplete or corrupt"):
        materialize_discovery_cells(
            manifest_path, output_root, fake_repo,
            agent_protocol=AGENT_PROTOCOL,
        )

    assert (partial / "plan.md").read_text() == "do not overwrite me\n"
    assert not (partial / "config.yaml").exists()


def test_full_campaign_canary_covers_every_arm_task_regime_without_gpu(tmp_path):
    fake_repo = tmp_path / "repo"
    _copy_campaign_sources(fake_repo)
    manifest_path = fake_repo / "benchmarks/campaigns/preprint_130/manifest.json"
    write_manifest(build_preprint_manifest(fake_repo), manifest_path)

    report = run_materialization_canary(manifest_path, repo_root=fake_repo)
    assert report["cells"] == 130
    assert len(report["regimes"]) == 10
    assert report["gpu_processes_started"] == 0
    assert not list(manifest_path.parent.glob(".canary-*"))
