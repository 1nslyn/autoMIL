"""One-off migration and repair coverage for the preprint-130 baselines."""
from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import pytest

from autobench.campaign import load_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "benchmarks/campaigns/preprint_130/manifest.json"
SCRIPT = REPO_ROOT / "benchmarks/campaigns/preprint_130/repair_baselines.py"


def _load_repair_module():
    name = "preprint_130_repair_baselines"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


baselines = _load_repair_module()


def _cell(*, framework: str = "nnmil", task_type: str = "classification") -> dict:
    task = "os" if task_type == "survival" else "idh1"
    return {
        "cell_id": f"cell-{framework}-{task_type}",
        "dataset": "tcga_lgg",
        "task": task,
        "task_type": task_type,
        "encoder": "uni_v2" if framework != "titan" else "titan",
        "model": {
            "nnmil": "simple_mil",
            "dtfd": "dtfd_mil",
            "titan": "titan",
            "clam": "clam_mb",
            "abmil": "abmil",
        }[framework],
        "framework": framework,
        "seed": 42,
    }


def _legacy_result(root: Path, cell: dict) -> Path:
    source = baselines.historical_result_dir(root, cell)
    source.mkdir(parents=True)
    is_survival = cell["task_type"] == "survival"
    loss = "nllsurv" if is_survival else None
    (source / "config.json").write_text(json.dumps({
        "task": {
            "name": cell["task"],
            "task_type": cell["task_type"],
        },
        "encoder_key": cell["encoder"],
        "model": {"model_type": cell["model"]},
        "train": {"seed": 42},
        "dataset": cell["dataset"],
        "n_folds": 5,
        "framework": cell["framework"],
        "strategy": "standard",
        "survival_loss": loss,
    }))
    per_fold_val = []
    per_fold_test = []
    for fold in range(5):
        if is_survival:
            val = {"c_index": 0.60 + fold / 100}
            test = {"c_index": 0.55 + fold / 100}
        else:
            val = {
                "auc_roc": 0.70 + fold / 100,
                "balanced_accuracy": 0.60 + fold / 100,
            }
            test = {
                "auc_roc": 0.65 + fold / 100,
                "balanced_accuracy": 0.55 + fold / 100,
            }
        per_fold_val.append(val)
        per_fold_test.append(test)
        fold_dir = source / f"fold_{fold}"
        fold_dir.mkdir()
        (fold_dir / "metrics.json").write_text(json.dumps({
            "fold": fold,
            "val_metrics": val,
            "test_metrics": test,
            "elapsed_seconds": 10 + fold,
        }))
    primary = ("c_index",) if is_survival else ("auc_roc", "balanced_accuracy")
    summary = {
        "dataset": cell["dataset"],
        "task": cell["task"],
        "encoder": cell["encoder"],
        "model_type": cell["model"],
        "framework": cell["framework"],
        "strategy": "standard",
        "n_folds": 5,
        "seed": 42,
        "survival_loss": loss,
        "per_fold_val": per_fold_val,
        "per_fold_test": per_fold_test,
        "val": {
            key: {"mean": sum(row[key] for row in per_fold_val) / 5}
            for key in primary
        },
        "test": {
            key: {"mean": sum(row[key] for row in per_fold_test) / 5}
            for key in primary
        },
    }
    (source / "summary.json").write_text(json.dumps(summary))
    return source


def test_manifest_reuse_partition_is_70_reuse_and_60_rerun():
    cells = load_manifest(MANIFEST)["cells"]
    assert sum(
        cell["framework"] in baselines.HISTORICAL_REUSABLE_FRAMEWORKS
        for cell in cells
    ) == 70
    assert sum(
        cell["framework"] in baselines.HISTORICAL_STALE_FRAMEWORKS
        for cell in cells
    ) == 60


@pytest.mark.parametrize("task_type", ["classification", "survival"])
def test_historical_validator_proves_exact_five_fold_evidence(tmp_path, task_type):
    cell = _cell(task_type=task_type)
    source = _legacy_result(tmp_path, cell)

    validated = baselines.validate_historical_baseline(cell, source)

    assert len(validated["folds"]) == 5
    assert set(validated["source_sha256"]) == {
        "config.json", "summary.json",
        *(f"fold_{fold}/metrics.json" for fold in range(5)),
    }


@pytest.mark.parametrize("framework", ["clam", "abmil"])
def test_historical_validator_rejects_every_stale_recipe_arm(tmp_path, framework):
    cell = _cell(framework=framework)
    source = baselines.historical_result_dir(tmp_path, cell)
    with pytest.raises(baselines.HistoricalBaselineError, match="recipe changed"):
        baselines.validate_historical_baseline(cell, source)


def test_historical_validator_rejects_summary_fold_disagreement(tmp_path):
    cell = _cell()
    source = _legacy_result(tmp_path, cell)
    summary_path = source / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["per_fold_val"][3]["auc_roc"] = 0.01
    summary_path.write_text(json.dumps(summary))

    with pytest.raises(
        baselines.HistoricalBaselineError, match="validation differs from summary",
    ):
        baselines.validate_historical_baseline(cell, source)


def test_conversion_exposes_validation_only_and_seals_test(tmp_path):
    cell = _cell()
    source = _legacy_result(tmp_path / "legacy", cell)
    cell_root = tmp_path / "runtime" / cell["cell_id"]
    cell_root.mkdir(parents=True)
    observed = {}

    def fake_register(root, archive):
        public = json.loads((archive / "result.json").read_text())
        sealed = [
            json.loads(
                (archive / "certify" / f"fold_{fold}_result.json").read_text()
            )
            for fold in range(5)
        ]
        observed.update({"root": root, "public": public, "sealed": sealed})
        return {"baseline": {
            "candidate_sha256": "a" * 64,
            "attestation_sha256": "b" * 64,
        }}

    result = baselines.convert_and_register_historical_baseline(
        cell_root, cell, source, register=fake_register,
    )

    assert observed["root"] == cell_root
    assert "held_out" not in observed["public"]
    assert "summary" not in observed["public"]
    assert [row["fold_index"] for row in observed["sealed"]] == list(range(5))
    assert all(set(row["held_out"]) == {"test_auc", "test_bacc"}
               for row in observed["sealed"])
    assert result["disposition"] == "registered-reuse"


def test_canonical_path_is_dataset_rooted_and_seed_aware(tmp_path):
    cell = _cell(task_type="survival")
    assert baselines.canonical_result_dir(tmp_path, cell) == (
        tmp_path / "tcga_lgg/results/nnmil/standard/os/uni_v2/simple_mil/"
        "nllsurv/s42"
    )


def test_atomic_copy_never_overwrites_a_conflicting_destination(tmp_path):
    cell = _cell()
    source = _legacy_result(tmp_path / "legacy", cell)
    destination = baselines.canonical_result_dir(tmp_path / "canonical", cell)
    destination.mkdir(parents=True)
    (destination / "unexpected.txt").write_text("do not overwrite")

    with pytest.raises(
        baselines.HistoricalBaselineError, match="refusing to overwrite",
    ):
        baselines._copy_result_tree_atomic(source, destination)
    assert (destination / "unexpected.txt").read_text() == "do not overwrite"


def test_atomic_copy_hash_verifies_and_is_idempotent(tmp_path):
    cell = _cell()
    source = _legacy_result(tmp_path / "legacy", cell)
    destination = baselines.canonical_result_dir(tmp_path / "canonical", cell)

    first = baselines._copy_result_tree_atomic(source, destination)
    second = baselines._copy_result_tree_atomic(source, destination)

    assert first["disposition"] == "copied"
    assert second["disposition"] == "already-present"
    assert first["tree_sha256"] == second["tree_sha256"]
    assert baselines._tree_inventory(source) == baselines._tree_inventory(destination)


def test_prepared_links_are_relative_verified_and_do_not_link_results(tmp_path):
    dataset_root = tmp_path / "tcga_lgg"
    legacy = dataset_root / "benchmark_5fold"
    for component in baselines._PREP_COMPONENTS:
        (legacy / component).mkdir(parents=True)

    rows = baselines.ensure_prepared_links(tmp_path, ["tcga_lgg"], create=True)

    assert len(rows) == len(baselines._PREP_COMPONENTS)
    for component in baselines._PREP_COMPONENTS:
        link = dataset_root / component
        assert link.is_symlink()
        assert link.readlink() == Path("benchmark_5fold") / component
    assert not (dataset_root / "results").exists()
    assert all(
        row["disposition"] == "verified"
        for row in baselines.ensure_prepared_links(
            tmp_path, ["tcga_lgg"], create=False,
        )
    )


def test_rerun_plan_is_exact_60_cells_and_four_canary_regimes(tmp_path):
    plan = baselines.rerun_plan(MANIFEST, tmp_path)

    assert plan["count"] == 60
    assert len(plan["canary_indices"]) == 4
    assert {row["framework"] for row in plan["cells"]} == {"clam", "abmil"}
    assert all("--benchmark-dir" in row["command"] for row in plan["cells"])
    assert all("--skip-prep" in row["command"] for row in plan["cells"])
    assert all("benchmark_5fold" not in row["destination"] for row in plan["cells"])


def test_recoverable_dataset_subset_has_stable_plan_and_slurm_binding(tmp_path):
    datasets = ("cptac_gbm", "tcga_hnsc", "tcga_luad")
    plan = baselines.rerun_plan(MANIFEST, tmp_path, datasets=datasets)

    assert plan["count"] == 36
    assert plan["datasets"] == sorted(datasets)
    assert len(plan["canary_indices"]) == 4
    assert {row["dataset"] for row in plan["cells"]} == set(datasets)

    output = baselines.write_slurm_runner(
        MANIFEST, tmp_path, REPO_ROOT, tmp_path / "ops", datasets=datasets,
    )
    runner = Path(output["runner"]).read_text()
    assert "--datasets cptac_gbm,tcga_hnsc,tcga_luad" in runner
    assert '"${SLURM_ARRAY_TASK_ID}"' in runner


def test_dataset_selection_rejects_names_outside_manifest():
    manifest = load_manifest(MANIFEST)
    with pytest.raises(
        baselines.HistoricalBaselineError, match="outside the manifest",
    ):
        baselines._selected_datasets(manifest, ("not_a_dataset",))
