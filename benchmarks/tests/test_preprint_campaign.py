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
    ACTIVITY_METRICS_PORT,
    ACTIVE_CELL_COUNT,
    ACTIVE_ROSTER,
    ANALYSIS_PLAN_PATH,
    BASELINE_FOLDS,
    CANARY_AGENT_PROTOCOL,
    CERTIFICATION_FOLDS,
    CAMPAIGN_ID,
    DATASETS,
    DISCOVERY_AGENT_ACTIVE_BUDGET,
    DISCOVERY_ATTEMPTS,
    PROTOCOL,
    PROTOCOL_VERSION,
    CampaignManifestError,
    audit_materialized_campaign,
    build_preprint_manifest,
    content_sha256,
    file_sha256,
    load_active_roster,
    load_manifest,
    materialize_discovery_cells,
    run_materialization_canary,
    validate_manifest,
    write_manifest,
)
from autobench.campaign_stages import CampaignStageError, freeze_campaign_selections
from autobench.guard_margin import derived_margin_for_counts

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
    "effort": "max",
    "network_access": "enabled",
    "fallback_model": None,
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
        "discovery": DISCOVERY_ATTEMPTS * 3,
        "promotion_per_candidate": 2,
        "promotion_candidates_min": 0,
        "promotion_candidates_max": 10,
        "minimum": 90,
        "maximum": 110,
    }
    assert protocol["discovery_agent_active_budget"] == "12h"
    assert protocol["attempt_timeout"] == {
        "minutes": 600,
        "role": "failure-containment-not-search-budget",
        "scope": "one-multi-fold-attempt",
    }
    assert all(set(cell["commands"]) == {"baseline", "discovery", "promotion"}
               for cell in manifest["cells"])


def test_manifest_lock_detects_byte_tampering(tmp_path):
    fake_repo = tmp_path / "repo"
    _copy_campaign_sources(fake_repo)
    path = tmp_path / "manifest.json"
    write_manifest(build_preprint_manifest(fake_repo), path)
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
    _write_guard_margins(fake_repo)
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


def _write_guard_margins(fake_repo: Path) -> dict[str, dict]:
    """Companion-guard margins for a fake repo, keyed like the real artifact.

    Synthesized rather than copied: the real file is derived from mounted
    cohort splits, and a host holding only some of them could not exercise
    the manifest/materialize/audit chain at all. The VALUES are arbitrary;
    what these tests assert is that the chain carries whatever was derived,
    unchanged, from artifact to manifest to cell config.
    """
    margins: dict[str, dict] = {}
    for index, dataset in enumerate(DATASETS):
        group = "cptac" if dataset.startswith("cptac_") else "tcga"
        raw = yaml.safe_load(
            (REPO_ROOT / "benchmarks/datasets" / group / f"{dataset}.yaml").read_text()
        ) or {}
        for task, spec in (raw.get("tasks") or {}).items():
            if (spec or {}).get("task_type", "classification") == "survival":
                continue
            counts = {
                str(fold): {"a": 11 + index, "b": 20}
                for fold in CERTIFICATION_FOLDS
            }
            margins[f"{dataset}__{task}"] = {
                "metric": "val_bacc",
                # Self-consistent like a real artifact: the counts cover every
                # certification fold (stages average different subsets) while
                # the declared margin is the one the framework gate consumes.
                "margin": derived_margin_for_counts(
                    counts, PROTOCOL["stage_folds"]["discovery"]
                ),
                "basis": f"synthetic fixture margin for {dataset}__{task}",
                "validation_class_counts": counts,
            }
    path = fake_repo / "benchmarks/campaigns/preprint_130/guard_margins.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(margins, indent=2, sort_keys=True) + "\n")
    return margins


def test_materializer_creates_roster_count_independent_discovery_states(tmp_path):
    fake_repo = tmp_path / "repo"
    _copy_campaign_sources(fake_repo)
    manifest_path = fake_repo / "benchmarks/campaigns/preprint_130/manifest.json"
    manifest = build_preprint_manifest(fake_repo)
    write_manifest(manifest, manifest_path)
    output_root = fake_repo / "benchmarks/campaigns/preprint_130/runtime"
    roots = materialize_discovery_cells(
        manifest_path, output_root, fake_repo,
        agent_protocol=AGENT_PROTOCOL,
    )
    # The manifest's own row index per cell — ports are pinned to this, not to
    # a cell's position within the roster-filtered `roots` list, since
    # off-roster rows (tcga_lgg, cptac_gbm) are skipped without renumbering
    # the rows around them.
    manifest_row = {
        cell["cell_id"]: index for index, cell in enumerate(manifest["cells"])
    }

    assert len(roots) == ACTIVE_CELL_COUNT
    assert len(set(roots)) == ACTIVE_CELL_COUNT
    assert all((root / "config.yaml").exists() for root in roots)
    assert all((root / "campaign_cell.json").exists() for root in roots)
    assert not any((root / "graph.json").exists() for root in roots)
    assert json.loads((output_root / "agent_protocol.json").read_text()) == AGENT_PROTOCOL

    for root in (roots[0], roots[30], roots[-1]):
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
        assert config["cap"]["eval_budget"] == DISCOVERY_ATTEMPTS
        assert config["cap"]["budget"] == DISCOVERY_AGENT_ACTIVE_BUDGET
        assert config["training"]["fold_count"] == 3
        assert config["orchestrator"]["default_timeout_min"] == 600
        assert config["files"]["editable"] == [
            f"{root.relative_to(fake_repo).as_posix()}/variants/_policies/*.py"
        ]
        # The exporter port is pinned to the cell's MANIFEST row, preserved
        # even though off-roster rows are skipped in between.
        expected_port = 9464 + manifest_row[cell["cell_id"]]
        assert config["activity"] == {"exporter_port": expected_port}
        settings = json.loads((root.parent / ".claude/settings.json").read_text())
        assert settings["env"]["OTEL_EXPORTER_PROMETHEUS_PORT"] == str(expected_port)

    # Every cell exports on its own deterministic port, so any number of
    # cells can meter concurrently on one host. Off-roster manifest rows are
    # skipped without renumbering the rows around them, so the roster's
    # ports land at their ORIGINAL manifest-row offset (e.g. cptac_pdac
    # starts at 9464+78, tcga_hnsc at 9464+104) rather than forming one
    # contiguous 0..77 block.
    ports = [
        yaml.safe_load((root / "config.yaml").read_text())["activity"][
            "exporter_port"
        ]
        for root in roots
    ]
    cells_per_dataset = 26
    expected_ports = [
        9464 + row
        for dataset_index, dataset in enumerate(DATASETS)
        if dataset in ACTIVE_ROSTER["cohorts"]
        for row in range(
            dataset_index * cells_per_dataset,
            (dataset_index + 1) * cells_per_dataset,
        )
    ]
    assert ports == expected_ports


def test_materializer_only_cells_builds_exactly_those_roots_with_pinned_rows(tmp_path):
    """A rehearsal set materializes a named subset of the active roster; row
    indices (and so exporter ports) stay those of the full manifest."""
    fake_repo = tmp_path / "repo"
    _copy_campaign_sources(fake_repo)
    manifest_path = fake_repo / "benchmarks/campaigns/preprint_130/manifest.json"
    manifest = build_preprint_manifest(fake_repo)
    write_manifest(manifest, manifest_path)
    luad = [c for c in manifest["cells"] if c["dataset"] == "tcga_luad"]
    chosen = {luad[3]["cell_id"], luad[7]["cell_id"]}
    output_root = fake_repo / "benchmarks/campaigns/preprint_130/runtime-rehearsal"
    roots = materialize_discovery_cells(
        manifest_path, output_root, fake_repo,
        agent_protocol=AGENT_PROTOCOL, only_cells=chosen,
    )
    assert {r.parent.name for r in roots} == chosen          # roots are the automil dirs
    assert {p.name for p in output_root.iterdir() if p.is_dir()} == chosen
    rows = {c["cell_id"]: i for i, c in enumerate(manifest["cells"])}
    for root in roots:
        config = yaml.safe_load((root / "config.yaml").read_text())
        assert config["activity"]["exporter_port"] == ACTIVITY_METRICS_PORT + rows[root.parent.name]
    off_roster = next(c["cell_id"] for c in manifest["cells"] if c["dataset"] not in ACTIVE_ROSTER["cohorts"])
    with pytest.raises(CampaignManifestError, match="not on the active roster"):
        materialize_discovery_cells(
            manifest_path, output_root, fake_repo,
            agent_protocol=AGENT_PROTOCOL, only_cells={off_roster},
        )
    with pytest.raises(CampaignManifestError, match="unknown cell"):
        materialize_discovery_cells(
            manifest_path, output_root, fake_repo,
            agent_protocol=AGENT_PROTOCOL, only_cells={"tcga_luad__nope"},
        )


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


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"effort": "high"}, "requires max agent effort"),
        ({"network_access": "disabled"}, "requires external network access"),
        ({"fallback_model": "test-model-2"}, "forbids model fallback"),
    ],
)
def test_materializer_rejects_drift_in_frozen_agent_axes(
    tmp_path, override, message,
):
    fake_repo = tmp_path / "repo"
    _copy_campaign_sources(fake_repo)
    manifest_path = fake_repo / "benchmarks/campaigns/preprint_130/manifest.json"
    write_manifest(build_preprint_manifest(fake_repo), manifest_path)

    with pytest.raises(CampaignManifestError, match=message):
        materialize_discovery_cells(
            manifest_path,
            fake_repo / "benchmarks/campaigns/preprint_130/runtime",
            fake_repo,
            agent_protocol={**AGENT_PROTOCOL, **override},
        )


def test_companion_guard_reaches_classification_cells_only(tmp_path):
    """The derived margin travels artifact → manifest → cell config intact.

    Selection stays single-metric; the guard only adds a veto, so it must land
    on every cell whose family reports balanced accuracy and on no other.
    """
    fake_repo = tmp_path / "repo"
    _copy_campaign_sources(fake_repo)
    margins = json.loads(
        (fake_repo / "benchmarks/campaigns/preprint_130/guard_margins.json").read_text()
    )
    manifest_path = fake_repo / "benchmarks/campaigns/preprint_130/manifest.json"
    manifest = build_preprint_manifest(fake_repo)
    write_manifest(manifest, manifest_path)
    output_root = fake_repo / "benchmarks/campaigns/preprint_130/runtime"
    roots = materialize_discovery_cells(
        manifest_path, output_root, fake_repo, agent_protocol=AGENT_PROTOCOL,
    )

    by_id = {cell["cell_id"]: cell for cell in manifest["cells"]}
    seen = {"classification": 0, "survival": 0}
    for root in roots:
        cell = by_id[root.parent.name]
        scoring = yaml.safe_load((root / "config.yaml").read_text())["scoring"]
        if cell["task_family"] == "survival":
            assert cell["guard"] is None
            assert "guard" not in scoring
        else:
            expected = margins[f"{cell['dataset']}__{cell['task']}"]
            assert cell["guard"] == expected
            assert scoring["guard"] == expected
            # The guard never displaces the selection signal.
            assert scoring["formula"] == "val_auc"
        seen[cell["task_type"]] += 1
    # Every roster dataset contributes 13 classification + 13 survival cells.
    assert seen == {
        "classification": ACTIVE_CELL_COUNT // 2,
        "survival": ACTIVE_CELL_COUNT // 2,
    }


def test_audit_rejects_a_widened_companion_guard(tmp_path):
    """A cell that loosened its own margin gates on a number the frozen
    validation counts do not justify."""
    fake_repo = tmp_path / "repo"
    _copy_campaign_sources(fake_repo)
    manifest_path = fake_repo / "benchmarks/campaigns/preprint_130/manifest.json"
    write_manifest(build_preprint_manifest(fake_repo), manifest_path)
    output_root = fake_repo / "benchmarks/campaigns/preprint_130/runtime"
    roots = materialize_discovery_cells(
        manifest_path, output_root, fake_repo, agent_protocol=AGENT_PROTOCOL,
    )
    classification = next(
        root for root in roots
        if yaml.safe_load((root / "config.yaml").read_text())["scoring"].get("guard")
    )
    config_path = classification / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["scoring"]["guard"]["margin"] = 0.5
    config_path.write_text(yaml.safe_dump(config))

    with pytest.raises(CampaignManifestError, match="companion-guard drift"):
        audit_materialized_campaign(
            roots=roots, manifest_path=manifest_path, repo_root=fake_repo,
        )


def test_the_canary_still_exercises_every_cell_when_a_margin_is_underived(tmp_path):
    """Readiness is not constructability.

    The canary materializes all 130 cells into a temp dir under a protocol
    that can never enter a publication freeze, purely to prove the machinery
    builds them. Refusing there would stop the dry-run from exercising the
    very cells it exists to exercise, while every REAL materialization is
    still refused and baseline registration refuses again.
    """
    fake_repo = tmp_path / "repo"
    _copy_campaign_sources(fake_repo)
    path = fake_repo / "benchmarks/campaigns/preprint_130/guard_margins.json"
    margins = json.loads(path.read_text())
    margins.pop(next(iter(margins)))
    path.write_text(json.dumps(margins, indent=2, sort_keys=True))
    manifest_path = fake_repo / "benchmarks/campaigns/preprint_130/manifest.json"
    write_manifest(build_preprint_manifest(fake_repo), manifest_path)

    summary = run_materialization_canary(manifest_path, repo_root=fake_repo)
    assert summary["cells"] == 130
    # ...and the underived cells carry no guard rather than a fabricated one.
    by_id = {c["cell_id"]: c for c in load_manifest(manifest_path)["cells"]}
    assert any(c["guard"] is None and c["task_family"] != "survival"
               for c in by_id.values())


def test_underived_classification_cell_cannot_be_materialized(tmp_path):
    """Fail closed at the last honest moment.

    Deriving a margin needs the cohort mounted, so a manifest built on a
    partial host records the gap as null rather than inventing a number that
    would be published as if it came from the split. Nothing may RUN on that
    gap: materialization is the only way to a runnable cell, so that is where
    the requirement bites.
    """
    fake_repo = tmp_path / "repo"
    _copy_campaign_sources(fake_repo)
    path = fake_repo / "benchmarks/campaigns/preprint_130/guard_margins.json"
    margins = json.loads(path.read_text())
    # Must be an ON-ROSTER cohort's key: an off-roster cell (e.g. cptac_gbm,
    # dropped from the active roster) is skipped by materialization before
    # its guard is ever checked, so removing its margin would not exercise
    # this fail-closed path at all.
    roster_key = next(key for key in margins if key.split("__")[0] in ACTIVE_ROSTER["cohorts"])
    margins.pop(roster_key)
    path.write_text(json.dumps(margins, indent=2, sort_keys=True))

    manifest_path = fake_repo / "benchmarks/campaigns/preprint_130/manifest.json"
    manifest = build_preprint_manifest(fake_repo)   # builds; records the gap
    write_manifest(manifest, manifest_path)
    assert any(
        cell["guard"] is None and cell["task_family"] != "survival"
        for cell in manifest["cells"]
    )
    with pytest.raises(CampaignManifestError, match="no companion-guard margin"):
        materialize_discovery_cells(
            manifest_path,
            fake_repo / "benchmarks/campaigns/preprint_130/runtime",
            fake_repo,
            agent_protocol=AGENT_PROTOCOL,
        )


def test_manifest_refuses_a_margin_its_own_counts_do_not_imply(tmp_path):
    """A hand-edited margin beside honest counts must not pass as "derived".

    Everything downstream — the hash, materialization, graph seeding — treats
    a manifest guard as derived from the counts travelling with it. Without
    this check a cell could search under a 0.5 margin (i.e. no guard at all)
    while publishing counts that imply 0.0099.
    """
    fake_repo = tmp_path / "repo"
    _copy_campaign_sources(fake_repo)
    path = fake_repo / "benchmarks/campaigns/preprint_130/guard_margins.json"
    margins = json.loads(path.read_text())
    key = next(iter(margins))
    margins[key]["margin"] = 0.5
    path.write_text(json.dumps(margins, indent=2, sort_keys=True))

    with pytest.raises(CampaignManifestError, match="its own published counts"):
        build_preprint_manifest(fake_repo)


def test_manifest_refuses_counts_for_the_wrong_fold_set(tmp_path):
    """The counts have to cover every fold set the guard is applied over.

    K is part of the lattice, so each stage's margin is derived over the folds
    IT averages — the search gate and the discovery freeze over folds 0-2, the
    promotion freeze over all five. Counts covering only the discovery folds
    are internally consistent and still leave the promotion margin underivable.
    """
    fake_repo = tmp_path / "repo"
    _copy_campaign_sources(fake_repo)
    path = fake_repo / "benchmarks/campaigns/preprint_130/guard_margins.json"
    margins = json.loads(path.read_text())
    key = next(iter(margins))
    counts = margins[key]["validation_class_counts"]
    margins[key]["validation_class_counts"] = {
        fold: counts[fold] for fold in map(str, PROTOCOL["stage_folds"]["discovery"])
    }
    path.write_text(json.dumps(margins, indent=2, sort_keys=True))

    with pytest.raises(CampaignManifestError, match="the campaign averages"):
        build_preprint_manifest(fake_repo)


def test_manifest_refuses_a_margin_without_its_counts(tmp_path):
    """The number has to stay checkable by hand from published counts."""
    fake_repo = tmp_path / "repo"
    _copy_campaign_sources(fake_repo)
    path = fake_repo / "benchmarks/campaigns/preprint_130/guard_margins.json"
    margins = json.loads(path.read_text())
    margins[next(iter(margins))].pop("validation_class_counts")
    path.write_text(json.dumps(margins, indent=2, sort_keys=True))

    with pytest.raises(CampaignManifestError, match="no validation class counts"):
        build_preprint_manifest(fake_repo)


def test_audit_rejects_a_hand_edited_frozen_guard(tmp_path):
    """The FROZEN guard is the one that governs every keep/discard.

    `graph.json` `meta.scoring` uses setdefault freeze semantics, so a
    hand-edited margin wins over `config.yaml` for the rest of the campaign
    while the config-vs-manifest check above still reports the cell clean —
    the same mechanism the neighbouring `scoring.formula` lock exists for.
    """
    fake_repo = tmp_path / "repo"
    _copy_campaign_sources(fake_repo)
    manifest_path = fake_repo / "benchmarks/campaigns/preprint_130/manifest.json"
    manifest = build_preprint_manifest(fake_repo)
    write_manifest(manifest, manifest_path)
    output_root = fake_repo / "benchmarks/campaigns/preprint_130/runtime"
    roots = materialize_discovery_cells(
        manifest_path, output_root, fake_repo, agent_protocol=AGENT_PROTOCOL,
    )
    by_id = {cell["cell_id"]: cell for cell in manifest["cells"]}
    target = next(r for r in roots if by_id[r.parent.name]["guard"] is not None)
    frozen = by_id[target.parent.name]["guard"]
    (target / "graph.json").write_text(json.dumps({
        "schema_version": 3,
        "meta": {"scoring": {"formula": "val_auc",
                             "guard": {**frozen, "margin": 0.5}}},
        "nodes": {},
    }))

    with pytest.raises(CampaignManifestError, match="froze scoring.guard"):
        audit_materialized_campaign(
            roots=roots, manifest_path=manifest_path, repo_root=fake_repo,
        )


def test_audit_rejects_a_tampered_exporter_port(tmp_path):
    fake_repo = tmp_path / "repo"
    _copy_campaign_sources(fake_repo)
    manifest_path = fake_repo / "benchmarks/campaigns/preprint_130/manifest.json"
    write_manifest(build_preprint_manifest(fake_repo), manifest_path)
    output_root = fake_repo / "benchmarks/campaigns/preprint_130/runtime"
    roots = materialize_discovery_cells(
        manifest_path, output_root, fake_repo, agent_protocol=AGENT_PROTOCOL,
    )

    config_path = roots[3] / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["activity"]["exporter_port"] = 9464
    config_path.write_text(yaml.safe_dump(config))

    with pytest.raises(CampaignManifestError, match="activity exporter port drift"):
        audit_materialized_campaign(
            roots=roots, manifest_path=manifest_path, repo_root=fake_repo,
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


def test_committed_manifest_and_roster_match_the_real_repo():
    """The exact artifacts the roster-census contract rests on: the checked-in
    manifest is byte-identical to its own committed lock, and the committed
    roster loads to the declared 3-cohort/78-cell census — against the REAL
    repo, not a synthetic fixture."""
    lock_path = MANIFEST.with_suffix(MANIFEST.suffix + ".sha256")
    expected_hash = lock_path.read_text().split()[0]
    assert file_sha256(MANIFEST) == expected_hash

    roster = load_active_roster(REPO_ROOT)
    assert roster == {
        "cohorts": ("tcga_luad", "tcga_hnsc", "cptac_pdac"),
        "cells": 78,
    }


def _write_roster(fake_repo: Path, roster: dict) -> Path:
    path = fake_repo / "benchmarks/campaigns/preprint_130/active_roster.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(roster))
    return path


@pytest.mark.parametrize(
    ("roster", "match"),
    [
        # A cohort outside DATASETS.
        ({"cohorts": ["not_a_real_dataset"], "cells": 26}, "outside DATASETS"),
        # cells is not 26 * len(cohorts).
        ({"cohorts": ["tcga_luad"], "cells": 25}, "cells must equal"),
        # Empty cohorts.
        ({"cohorts": [], "cells": 0}, "non-empty list of unique"),
        # Duplicate cohorts.
        (
            {"cohorts": ["tcga_luad", "tcga_luad"], "cells": 52},
            "non-empty list of unique",
        ),
    ],
)
def test_load_active_roster_fails_closed_on_every_violation(tmp_path, roster, match):
    fake_repo = tmp_path / "repo"
    _write_roster(fake_repo, roster)

    with pytest.raises(CampaignManifestError, match=match):
        load_active_roster(fake_repo)
