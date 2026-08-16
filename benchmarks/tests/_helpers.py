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
