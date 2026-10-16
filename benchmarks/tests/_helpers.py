"""Shared test helpers for autobench tests."""

from autobench.config import DatasetConfig, StrategyDef, TaskDef


def make_test_ds(**kwargs):
    """Create a minimal DatasetConfig for testing.

    Mirrors the ovarian dataset structure with sensible defaults.
    Override any field via kwargs.
    """
    defaults = dict(
        name="test",
        description="Test dataset",
        data_root="/tmp/test",
        wsi_dir="/tmp/test/wsi",
        mapping_csv="/tmp/test/mapping.csv",
        output_dir="/tmp/test/output",
        benchmark_dir="/tmp/test/benchmark",
        features_base_dir="/tmp/test/features",
        tasks={
            "brca": TaskDef(
                name="brca",
                label_col="BRCA_predict_label",
                label_map={0: "neg", 1: "pos"},
                n_classes=2,
            ),
            "hrd": TaskDef(
                name="hrd",
                label_col="HRD_label",
                label_map={0: "neg", 1: "pos"},
                n_classes=2,
            ),
        },
        split_strategies={
            "standard": StrategyDef(
                name="standard",
                train_cohorts=[],
                test_cohorts=[],
            ),
            "devcv": StrategyDef(
                name="devcv",
                train_cohorts=[],
                test_cohorts=[],
            ),
            "devcv_r1": StrategyDef(
                name="devcv_r1",
                train_cohorts=[],
                test_cohorts=[],
            ),
            "devcv_r2": StrategyDef(
                name="devcv_r2",
                train_cohorts=[],
                test_cohorts=[],
            ),
        },
        task_strategy_feasibility={
            "brca": ["standard"],
            "hrd": ["standard"],
        },
        slide_id_column="new_name",
        slide_id_transform="strip_svs",
        wsi_extension=None,
        case_id_column="primary_case_id",
        status_column="status",
        status_value="mapped_unique_case_id",
        encoder_models={
            "histai/hibou-L": "hibou_l",
            "MahmoodLab/conchv1_5": "conch_v15",
            "paige-ai/Virchow2": "virchow2",
            "kaiko-ai/midnight": "midnight12k",
            "MahmoodLab/UNI2-h": "uni_v2",
            "bioptimus/H-optimus-1": "hoptimus1",
            "bioptimus/H0-mini": "h0_mini",
        },
        encoder_dims={
            "hibou_l": 1024,
            "conch_v15": 768,
            "virchow2": 2560,
            "midnight12k": 1536,
            "uni_v2": 1536,
            "hoptimus1": 1536,
            "h0_mini": 768,
        },
        nnmil_models=[
            "trans_mil", "simple_mil", "ds_mil", "dtfd_mil",
            "wikg_mil", "ilra_mil", "rrt", "vision_transformer",
        ],
        dtfd_models=["dtfd_mil"],
        abmil_models=["abmil", "abmil_gated"],
        magnification=20,
        patch_size=224,
        batch_size=64,
    )
    defaults.update(kwargs)
    return DatasetConfig(**defaults)


def make_ledger_exp(
    *,
    task_name: str = "grade",
    label_dict: dict | None = None,
    encoder: str = "virchow2",
    embed_dim: int = 768,
    model_type: str = "clam_mb",
    framework=None,
    dataset: str = "tcga_hnsc",
    seed: int = 42,
):
    """One experiment at the REAL path shape (``results_subdir``).

    The single authority for completion-ledger/resume tests: two suites used
    to carry drifting private copies, so a path-shape change could defang one
    of them silently — the surviving copy kept writing summaries at the old
    location and every resume test passed for the wrong reason.
    """
    from autobench.pipeline.config import (
        ExperimentConfig,
        Framework,
        ModelConfig,
        TaskConfig,
        TrainConfig,
    )

    return ExperimentConfig(
        task=TaskConfig(
            name=task_name,
            label_col="label",
            label_dict=label_dict or {"g1": 0, "g2": 1, "g3": 2},
            task_type="classification",
        ),
        encoder_key=encoder,
        embed_dim=embed_dim,
        model=ModelConfig(model_type=model_type),
        train=TrainConfig(seed=seed),
        n_folds=5,
        framework=framework if framework is not None else Framework.CLAM,
        strategy="standard",
        dataset=dataset,
    )


def write_ledger_summary(benchmark_dir: str, exp, **extra) -> str:
    """Materialize the cell directory the orchestrator will look for."""
    import json
    import os

    cell = os.path.join(benchmark_dir, "results", exp.results_subdir)
    os.makedirs(cell, exist_ok=True)
    path = os.path.join(cell, "summary.json")
    with open(path, "w") as f:
        json.dump({"experiment_id": exp.experiment_id, **extra}, f)
    return path


#: The GPU type a fixture runtime set declares (reproduction_policy.json "gpu").
DECLARED_GPU = {"name": "NVIDIA H100 80GB HBM3", "mig": False}


def write_reproduction_policy(
    repo_root, *, epsilon=0.025, runtimes=("runtime",), gpu=DECLARED_GPU,
):
    """Declare the reproduction policy in a fixture repo: the tolerance, and
    ``gpu`` as the GPU type of each named runtime set."""
    import json
    from pathlib import Path

    from autobench.campaign import REPRODUCTION_POLICY_PATH

    path = Path(repo_root) / REPRODUCTION_POLICY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "epsilon": epsilon,
        "gpu": {runtime: gpu for runtime in runtimes},
    }))
    return path


def full_h100_nvidia_smi(command):
    """nvidia-smi's answer on a node of four full H100s; None for any other
    command, so a fake ``subprocess.run`` can delegate to it first."""
    from types import SimpleNamespace

    if not command or command[0] != "nvidia-smi":
        return None
    listing = "".join(
        f"{index}, {DECLARED_GPU['name']}, Disabled\n" for index in range(4)
    )
    return SimpleNamespace(returncode=0, stdout=listing, stderr="")


def write_fake_rescore(
    cell_root,
    state,
    *,
    held_out,
    task_family="binary",
    validation=None,
    commit="e" * 40,
    attempt="rescore/attempt-fixture",
):
    """Write the archives of a finished winner re-score and return the
    ``rescore`` block ``run_winner_rescore`` would have recorded for them.

    ``held_out(strategy, candidate_id, fold)`` is the held-out metrics of one
    sealed fold file; ``validation(strategy, candidate_id, fold)`` is that
    fold's primary validation value. Fixtures that need a winner-rescored
    state commit the block themselves.
    """
    import hashlib
    import json
    import math
    from datetime import datetime, timezone
    from pathlib import Path

    from autobench.campaign import (
        CERTIFICATION_FOLDS,
        RESCORE_STRATEGIES,
        VALIDATION_SCHEMA_BY_FAMILY,
    )

    winner = state["winner"]
    baseline_wins = winner["kind"] == "baseline"
    candidate_ids = ["baseline"] + ([] if baseline_wins else [winner["candidate_id"]])
    validation = validation or (lambda strategy, candidate_id, fold: 0.60 + fold / 100)
    metric_keys = VALIDATION_SCHEMA_BY_FAMILY[task_family]
    strategies: dict = {}
    for strategy in RESCORE_STRATEGIES:
        strategies[strategy] = {}
        for candidate_id in candidate_ids:
            archive = Path(cell_root) / attempt / strategy / candidate_id / "archive"
            sealed = archive / "certify"
            sealed.mkdir(parents=True)
            folds, hashes = [], {}
            for fold in CERTIFICATION_FOLDS:
                value = validation(strategy, candidate_id, fold)
                folds.append({
                    "fold_index": fold,
                    "metrics": {key: value for key in metric_keys},
                    "primary_value": value,
                })
                path = sealed / f"fold_{fold}_result.json"
                path.write_text(json.dumps({
                    "fold_index": fold,
                    "held_out": held_out(strategy, candidate_id, fold),
                }))
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            mean = math.fsum(fold["primary_value"] for fold in folds) / len(folds)
            (archive / "result.json").write_text(json.dumps({
                "status": "completed",
                "primary_value": mean,
                "metrics": {key: mean for key in metric_keys},
                "validation_folds": folds,
            }))
            strategies[strategy][candidate_id] = {
                "archive": f"{attempt}/{strategy}/{candidate_id}/archive",
                "validation_folds": folds,
                "validation_mean": mean,
                "sealed_fold_sha256": hashes,
            }
    deltas = [] if baseline_wins else [
        fold["primary_value"] - base["primary_value"]
        for per_candidate in strategies.values()
        for fold, base in zip(
            per_candidate[winner["candidate_id"]]["validation_folds"],
            per_candidate["baseline"]["validation_folds"],
            strict=True,
        )
    ]
    return {
        "attempt": attempt,
        "commit": commit,
        "strategies": strategies,
        "validation_lift": math.fsum(deltas) / len(deltas) if deltas else 0.0,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
