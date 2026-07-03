"""Generate nnMIL dataset artifacts (dataset.json, dataset.csv, dataset_plan.json).

We generate ``dataset_plan.json`` ourselves (not using nnMIL's ExperimentPlanner)
to control splits precisely -- ensuring they match the shared split CSVs used by
CLAM.  We DO use nnMIL's ``ClassificationTrainer`` for actual training.
"""

from __future__ import annotations

import json
import os

import h5py
import numpy as np
import pandas as pd


def prepare_nnmil_experiment(
    benchmark_dir: str,
    task_name: str,
    encoder_key: str,
    strategy: str,
    label_col: str,
    label_dict: dict[str, int],
    embed_dim: int,
    features_base_dir: str,
    dataset_name: str = "dataset",
    seed: int = 42,
    n_splits: int = 5,
    max_seq_multiplier: float = 0.5,
    use_original_length: bool = False,
    goldmark_recipe: bool = False,
) -> str:
    """Prepare nnMIL dataset artifacts for one (task, encoder, strategy) combo.

    Returns the path to the generated ``dataset_plan.json``.

    The generated plan embeds the SAME splits from the shared split CSVs
    into nnMIL's ``data_splits`` format so that CLAM and nnMIL use
    identical patient/slide assignments.

    Patch-cap knobs (default reproduces upstream planner behavior exactly):

    ``max_seq_multiplier`` (default 0.5)
        Train-time sub-bag length = ``int(median_patches * max_seq_multiplier)``.
        nnMIL's planner uses 0.5 (keep half the median); GOLDMARK and CLAM keep
        ALL patches. On rare-event tasks the 0.5x cap can drop the few
        discriminative tiles, hurting train signal. Raise toward 1.0+ to retain
        more (note: lengths above a slide's patch count are zero-padded).
    ``use_original_length`` (default False)
        When True, train on the full variable-length bag (dropping only ~10%
        random patches) instead of the fixed sub-bag — the closest match to
        GOLDMARK's keep-all protocol. Val/test always use all patches either way.
    """
    dataset_dir = os.path.join(
        benchmark_dir, "nnmil", strategy, f"{task_name}_{encoder_key}"
    )
    plan_path = os.path.join(dataset_dir, "dataset_plan.json")

    if os.path.exists(plan_path):
        return plan_path

    os.makedirs(dataset_dir, exist_ok=True)

    # Task CSV is always {task_name}.csv
    task_csv_path = os.path.join(benchmark_dir, "dataset_csv", f"{task_name}.csv")

    task_df = pd.read_csv(task_csv_path)
    h5_dir = os.path.join(features_base_dir, f"features_{encoder_key}")

    # Filter to slides that have H5 feature files
    has_h5 = task_df["slide_id"].apply(
        lambda sid: os.path.exists(os.path.join(h5_dir, f"{sid}.h5"))
    )
    n_missing = (~has_h5).sum()
    if n_missing > 0:
        print(f"  Skipping {n_missing} slides without H5 features for {encoder_key}")
        task_df = task_df[has_h5].reset_index(drop=True)

    # --- dataset.json ---
    # Invert label_dict: {"neg": 0, "pos": 1} -> {"0": "neg", "1": "pos"}
    labels_map = {str(v): k for k, v in label_dict.items()}
    dataset_json = {
        "name": f"{dataset_name}_{task_name}_{strategy}",
        "description": f"{dataset_name} {task_name.upper()} classification, strategy {strategy}",
        "task_type": "classification",
        "task_name": f"{task_name}_{strategy}_{encoder_key}",
        "evaluation_setting": f"{n_splits}fold",
        "feature_dir": h5_dir,
        "labels": labels_map,
        # GOLDMARK selects the best checkpoint by validation ROC-AUC; match it
        # (nnMIL's EarlyStopping uses this metric for best-model selection too).
        "metric": "auc" if goldmark_recipe else "bacc",
        "modality": {"0": "Histopathology"},
    }
    with open(os.path.join(dataset_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)

    # --- dataset.csv ---
    # Map string labels to ints for nnMIL
    csv_df = pd.DataFrame({
        "slide_id": task_df["slide_id"],
        "patient_id": task_df["case_id"],
        "label": task_df["label"].map(label_dict),
    })
    csv_df.to_csv(os.path.join(dataset_dir, "dataset.csv"), index=False)

    # --- feature statistics (from a sample of H5 files) ---
    feature_stats = _analyze_features(
        h5_dir, task_df["slide_id"].tolist(), embed_dim,
        max_seq_multiplier=max_seq_multiplier,
    )

    # --- data splits (from shared split CSVs) ---
    splits_dir = os.path.join(benchmark_dir, "splits", strategy, task_name)
    data_splits = _load_splits_as_nnmil_format(
        splits_dir, task_df, label_dict, n_splits
    )

    # --- dataset_plan.json ---
    plan = {
        **dataset_json,
        "feature_statistics": feature_stats,
        "data_splits": data_splits,
        "training_configuration": _generate_training_config(
            feature_stats,
            len(task_df),
            n_classes=len(label_dict),
            metric=dataset_json["metric"],
            min_class_count=int(task_df["label"].value_counts().min()),
            max_seq_multiplier=max_seq_multiplier,
            use_original_length=use_original_length,
            goldmark_recipe=goldmark_recipe,
        ),
        "random_seed": seed,
    }
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2)

    print(f"  nnMIL plan: {plan_path}")
    return plan_path


def _analyze_features(
    h5_dir: str,
    slide_ids: list[str],
    expected_dim: int,
    sample_size: int = 100,
    max_seq_multiplier: float = 0.5,
) -> dict:
    """Analyze H5 feature files to get statistics for nnMIL config.

    Raises FileNotFoundError if no H5 file exists in the sample. The
    upstream planner (``experiment_planner.py:113-114``) raises in the
    same case; silently substituting fallback values would produce a
    dataset_plan.json with median=max=min=256 and the trainer would
    proceed against a phantom dataset.
    """
    patch_counts: list[int] = []
    feat_dim = expected_dim

    sample = slide_ids[:sample_size] if len(slide_ids) > sample_size else slide_ids
    for sid in sample:
        h5_path = os.path.join(h5_dir, f"{sid}.h5")
        if not os.path.exists(h5_path):
            continue
        with h5py.File(h5_path, "r") as f:
            shape = f["features"].shape
            feat_dim = shape[1]
            patch_counts.append(shape[0])

    if not patch_counts:
        raise FileNotFoundError(
            f"No H5 feature files found in {h5_dir} for the {len(sample)} "
            f"sampled slide_ids (first sample: {sample[:3]!r}). Verify the "
            "feature-extraction step ran for this encoder and the path "
            "matches the dataset YAML's features_base_dir."
        )

    arr = np.array(patch_counts)
    median = float(np.median(arr))
    return {
        "feature_dimension": feat_dim,
        "num_patches_per_slide": {
            "min": int(arr.min()),
            "max": int(arr.max()),
            "mean": float(arr.mean()),
            "median": median,
            "percentile_25": float(np.percentile(arr, 25)),
            "percentile_75": float(np.percentile(arr, 75)),
            "percentile_95": float(np.percentile(arr, 95)),
        },
        "recommended_max_seq_length": int(median * max_seq_multiplier),
    }


def _load_splits_as_nnmil_format(
    splits_dir: str,
    task_df: pd.DataFrame,
    label_dict: dict[str, int],
    n_splits: int,
) -> dict:
    """Convert shared split CSVs to nnMIL's data_splits format.

    Each fold in the output has::

        {
            "train": {"slide_ids": [...], "slide_info": [...]},
            "val":   {"slide_ids": [...], "slide_info": [...]},
            "test":  {"slide_ids": [...], "slide_info": [...]},
        }

    where each ``slide_info`` entry is::

        {"slide_id": "...", "patient_id": "...", "label": 0}
    """
    # Build lookup: slide_id -> (case_id, label_int)
    lookup: dict[str, tuple[str, int]] = {}
    for _, row in task_df.iterrows():
        label_int = label_dict.get(row["label"], row["label"])
        if isinstance(label_int, str):
            label_int = int(label_int)
        lookup[row["slide_id"]] = (row["case_id"], int(label_int))

    data_splits: dict[str, dict] = {}

    for fold_idx in range(n_splits):
        split_path = os.path.join(splits_dir, f"splits_{fold_idx}.csv")
        if not os.path.exists(split_path):
            break

        split_df = pd.read_csv(split_path)
        fold_data: dict[str, dict] = {}

        for split_name in ("train", "val", "test"):
            if split_name not in split_df.columns:
                continue
            sids = split_df[split_name].dropna().tolist()
            slide_info = []
            for sid in sids:
                if sid in lookup:
                    case_id, label_int = lookup[sid]
                    slide_info.append({
                        "slide_id": sid,
                        "patient_id": case_id,
                        "label": label_int,
                    })
            fold_data[split_name] = {
                "slide_ids": [si["slide_id"] for si in slide_info],
                "slide_info": slide_info,
            }

        data_splits[f"fold_{fold_idx}"] = fold_data

    return data_splits


def _generate_training_config(
    feature_stats: dict,
    n_samples: int,
    n_classes: int = 2,
    metric: str = "bacc",
    min_class_count: int | None = None,
    max_seq_multiplier: float = 0.5,
    use_original_length: bool = False,
    goldmark_recipe: bool = False,
) -> dict:
    """Generate nnMIL training configuration, matching upstream planner.

    Replicates ``lib/nnMIL/preprocessing/experiment_planner.py:573-725``
    so the wrapper's plan is bit-identical to what
    ``nnMIL_plan_experiment.py`` would emit.

    When ``goldmark_recipe`` is set, the protocol-level training knobs are
    overridden to match GOLDMARK's published recipe (AdamW lr=1e-4 wd=1e-4,
    120 epochs, no early stop, best-val-AUC selection). nnMIL's optimizer is
    already AdamW; its batch size, cosine+warmup schedule, and padded-bag
    sampler stay native (architecture-coupled, not GOLDMARK protocol levers),
    and keep-all-patches is handled via ``use_original_length`` upstream.
    """
    feat_dim = feature_stats["feature_dimension"]
    hidden_dim = max(256, feat_dim // 4)

    # Prefer the value computed once by _analyze_features to keep the
    # planner.py:129 formula in one place. Fall back to the live
    # computation for callers that hand-build a stats dict without going
    # through _analyze_features (e.g. unit tests). Both branches use the
    # same formula; this just keeps them from drifting silently.
    if "recommended_max_seq_length" in feature_stats:
        max_seq_length = int(feature_stats["recommended_max_seq_length"])
    else:
        max_seq_length = int(
            feature_stats["num_patches_per_slide"]["median"] * max_seq_multiplier
        )

    # planner.py:596 fallback: train set ≈ 80% of total when split info absent
    num_train_samples = int(n_samples * 0.8)

    # planner.py:602-657 batch_size formula (replicated verbatim, including the
    # buggy minority-visibility upgrade where min(candidates)=16 always no-ops)
    batch_size_candidates: list[int] = []
    if min_class_count is not None and min_class_count > 0:
        p_rare = min_class_count / n_samples
        batch_size_candidates.append(int(3 / p_rare))
    batch_size_candidates.extend([16, 48])
    if num_train_samples < 200:
        batch_size_candidates.append(16)
    elif num_train_samples <= 800:
        batch_size_candidates.extend([24, 32])
    else:
        batch_size_candidates.extend([32, 48])

    if num_train_samples < 200:
        batch_size = 16
    elif num_train_samples <= 800:
        batch_size = 24 if num_train_samples < 400 else 32
    else:
        batch_size = 32

    minority_constraint = [bs for bs in batch_size_candidates if 16 <= bs <= 48]
    if minority_constraint:
        min_minority = min(minority_constraint)
        if min_minority > batch_size:
            batch_size = min(min_minority, 48)
    batch_size = max(16, min(48, batch_size))

    # planner.py:660-674 — adaptive batch_sampler on metric
    metric_lower = metric.lower()
    if "auc" in metric_lower:
        batch_sampler = "auc"
    elif metric_lower in ("bacc", "balanced_accuracy", "f1", "f1_score"):
        batch_sampler = "balanced"
    else:
        batch_sampler = "random"

    cfg = {
        "feature_dimension": feat_dim,
        "hidden_dim": hidden_dim,
        "max_seq_length": max_seq_length,
        "use_original_length": use_original_length,
        "batch_size": batch_size,
        "batch_sampler": batch_sampler,
        "learning_rate": 3e-4,
        "weight_decay": 0.01 if hidden_dim >= 512 else 1e-4,
        "num_epochs": 100,
        "warmup_epochs": 10 if num_train_samples < 500 else 5,
        "dropout": 0.25,
        "patience": 10,
        "num_classes": n_classes,
    }
    if goldmark_recipe:
        # GOLDMARK protocol levers (configs/train_task_v2_pub.sh): lr=1e-4,
        # wd=1e-4, 120 epochs, no early stop (large patience -> trains full 120
        # and keeps the best-val-AUC checkpoint via EarlyStopping's metric).
        cfg["learning_rate"] = 1e-4
        cfg["weight_decay"] = 1e-4
        cfg["num_epochs"] = 120
        cfg["patience"] = 999
        # batch_size MUST be 1 for the GOLDMARK recipe. The recipe forces
        # use_original_length=True (keep-all-patches, matching GOLDMARK), which
        # yields variable-length bags; nnMIL's train collate calls torch.stack
        # which requires equal sizes and crashes on a multi-bag batch
        # ("stack expects each tensor to be equal size, got [1646,D] vs
        # [29843,D]"). With bs==1 the loader skips the batch samplers (guarded by
        # `batch_size > 1`) and never stacks two bags. This is also the most
        # faithful choice: GOLDMARK trains with batch_size=1, CrossEntropyLoss,
        # and no class sampler. SimpleMIL has no BatchNorm, so bs==1 is safe.
        cfg["batch_size"] = 1
    return cfg
