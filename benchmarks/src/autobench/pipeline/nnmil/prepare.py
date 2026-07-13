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


def nnmil_plan_dir(
    benchmark_dir: str,
    strategy: str,
    task_name: str,
    encoder_key: str,
    survival_loss: str | None = None,
) -> str:
    """Return the nnMIL plan directory for one experiment.

    Survival experiments get a per-loss suffix so cox/mse/mae/nllsurv don't
    overwrite each other's plans. Classification (``survival_loss=None``)
    keeps the original ``{task}_{encoder}`` layout unchanged.
    """
    leaf = f"{task_name}_{encoder_key}"
    if survival_loss is not None:
        leaf = f"{leaf}_{survival_loss}"
    return os.path.join(benchmark_dir, "nnmil", strategy, leaf)


def prepare_nnmil_experiment(
    benchmark_dir: str,
    task_name: str,
    encoder_key: str,
    strategy: str,
    label_col: str | None = None,
    label_dict: dict[str, int] | None = None,
    embed_dim: int = 0,
    features_base_dir: str = "",
    dataset_name: str = "dataset",
    seed: int = 42,
    n_splits: int = 5,
    *,
    task_type: str = "classification",
    event_col: str | None = None,
    time_col: str | None = None,
    survival_loss: str | None = None,
    nll_bins: int = 4,
) -> str:
    """Prepare nnMIL dataset artifacts for one (task, encoder, strategy) combo.

    Returns the path to the generated ``dataset_plan.json``.

    The generated plan embeds the SAME splits from the shared split CSVs
    into nnMIL's ``data_splits`` format so that CLAM and nnMIL use
    identical patient/slide assignments.

    For ``task_type="survival"`` the plan declares ``task_type: survival``,
    ``metric: c_index``, carries the chosen ``survival_loss`` (+ ``nll_bins``
    for nllsurv), and emits per-slide ``status``/``time`` instead of ``label``.
    """
    is_survival = task_type == "survival"
    dataset_dir = nnmil_plan_dir(
        benchmark_dir, strategy, task_name, encoder_key,
        survival_loss=survival_loss if is_survival else None,
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
    if is_survival:
        dataset_json = {
            "name": f"{dataset_name}_{task_name}_{strategy}",
            "description": (
                f"{dataset_name} {task_name.upper()} survival "
                f"({survival_loss}), strategy {strategy}"
            ),
            "task_type": "survival",
            "task_name": f"{task_name}_{strategy}_{encoder_key}",
            "evaluation_setting": f"{n_splits}fold",
            "feature_dir": h5_dir,
            "metric": "c_index",
            "modality": {"0": "Histopathology"},
            "survival_loss": survival_loss,
            "nll_bins": nll_bins,
        }
    else:
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
            "metric": "bacc",
            "modality": {"0": "Histopathology"},
        }
    with open(os.path.join(dataset_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)

    # --- dataset.csv ---
    if is_survival:
        # nnMIL's survival dataset expects columns named exactly status/time.
        csv_df = pd.DataFrame({
            "slide_id": task_df["slide_id"],
            "patient_id": task_df["case_id"],
            "status": task_df["status"].astype(int),
            "time": task_df["time"].astype(float),
        })
    else:
        # Map string labels to ints for nnMIL
        csv_df = pd.DataFrame({
            "slide_id": task_df["slide_id"],
            "patient_id": task_df["case_id"],
            "label": task_df["label"].map(label_dict),
        })
    csv_df.to_csv(os.path.join(dataset_dir, "dataset.csv"), index=False)

    # --- feature statistics (from a sample of H5 files) ---
    feature_stats = _analyze_features(h5_dir, task_df["slide_id"].tolist(), embed_dim)

    # --- data splits (from shared split CSVs) ---
    splits_dir = os.path.join(benchmark_dir, "splits", strategy, task_name)
    data_splits = _load_splits_as_nnmil_format(
        splits_dir, task_df, label_dict, n_splits, task_type=task_type,
    )

    # --- dataset_plan.json ---
    plan = {
        **dataset_json,
        "feature_statistics": feature_stats,
        "data_splits": data_splits,
        "training_configuration": _generate_training_config(
            feature_stats,
            len(task_df),
            n_classes=(None if is_survival else len(label_dict)),
            metric=dataset_json["metric"],
            min_class_count=(
                None if is_survival
                else int(task_df["label"].value_counts().min())
            ),
            task_type=task_type,
            survival_loss=survival_loss,
            nll_bins=nll_bins,
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
        "recommended_max_seq_length": int(median * 0.5),
    }


def _load_splits_as_nnmil_format(
    splits_dir: str,
    task_df: pd.DataFrame,
    label_dict: dict[str, int] | None,
    n_splits: int,
    task_type: str = "classification",
) -> dict:
    """Convert shared split CSVs to nnMIL's data_splits format.

    Each fold in the output has::

        {
            "train": {"slide_ids": [...], "slide_info": [...]},
            "val":   {"slide_ids": [...], "slide_info": [...]},
            "test":  {"slide_ids": [...], "slide_info": [...]},
        }

    where each ``slide_info`` entry is, for classification::

        {"slide_id": "...", "patient_id": "...", "label": 0}

    and for survival (keys nnMIL's ``UnifiedMILDataset`` consumes directly)::

        {"slide_id": "...", "patient_id": "...", "status": 1, "time": 12.5}
    """
    is_survival = task_type == "survival"

    # Build lookup: slide_id -> slide_info payload (without the slide_id key)
    lookup: dict[str, dict] = {}
    for _, row in task_df.iterrows():
        if is_survival:
            lookup[row["slide_id"]] = {
                "patient_id": row["case_id"],
                "status": int(row["status"]),
                "time": float(row["time"]),
            }
        else:
            label_int = label_dict.get(row["label"], row["label"])
            if isinstance(label_int, str):
                label_int = int(label_int)
            lookup[row["slide_id"]] = {
                "patient_id": row["case_id"],
                "label": int(label_int),
            }

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
                    slide_info.append({"slide_id": sid, **lookup[sid]})
            fold_data[split_name] = {
                "slide_ids": [si["slide_id"] for si in slide_info],
                "slide_info": slide_info,
            }

        data_splits[f"fold_{fold_idx}"] = fold_data

    return data_splits


def _generate_training_config(
    feature_stats: dict,
    n_samples: int,
    n_classes: int | None = 2,
    metric: str = "bacc",
    min_class_count: int | None = None,
    task_type: str = "classification",
    survival_loss: str | None = None,
    nll_bins: int = 4,
) -> dict:
    """Generate nnMIL training configuration, matching upstream planner.

    Replicates ``lib/nnMIL/preprocessing/experiment_planner.py:573-725``
    so the wrapper's plan is bit-identical to what
    ``nnMIL_plan_experiment.py`` would emit.

    Survival tasks (``task_type="survival"``) bypass the class-count batch
    logic, set ``num_classes`` to 1 (cox/mse/mae) or ``nll_bins`` (nllsurv),
    and inject ``survival_loss``/``nll_bins`` so the survival trainers pick
    them up via the config fallback.
    """
    if task_type == "survival":
        return _generate_survival_training_config(
            feature_stats, n_samples, survival_loss, nll_bins,
        )

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
        max_seq_length = int(feature_stats["num_patches_per_slide"]["median"] * 0.5)

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

    return {
        "feature_dimension": feat_dim,
        "hidden_dim": hidden_dim,
        "max_seq_length": max_seq_length,
        "use_original_length": False,
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


def _generate_survival_training_config(
    feature_stats: dict,
    n_samples: int,
    survival_loss: str | None,
    nll_bins: int,
) -> dict:
    """Training config for a survival experiment.

    Mirrors the classification config's feature/seq sizing but bypasses the
    class-minority batch logic (survival has no classes). ``num_classes`` is
    the survival head width: 1 for cox/mse/mae, ``nll_bins`` for nllsurv.
    ``survival_loss`` (+ ``nll_bins`` for nllsurv) are injected so the
    survival trainers read them from the plan's config fallback. LR follows
    nnMIL's survival planner default (1e-4); ``c_index`` metric resolves the
    batch sampler to ``random`` (accepted by the survival trainers).
    """
    feat_dim = feature_stats["feature_dimension"]
    hidden_dim = max(256, feat_dim // 4)
    if "recommended_max_seq_length" in feature_stats:
        max_seq_length = int(feature_stats["recommended_max_seq_length"])
    else:
        max_seq_length = int(feature_stats["num_patches_per_slide"]["median"] * 0.5)

    num_train_samples = int(n_samples * 0.8)
    if num_train_samples < 200:
        batch_size = 16
    elif num_train_samples <= 800:
        batch_size = 24 if num_train_samples < 400 else 32
    else:
        batch_size = 32
    batch_size = max(16, min(48, batch_size))

    num_classes = nll_bins if survival_loss == "nllsurv" else 1

    cfg = {
        "feature_dimension": feat_dim,
        "hidden_dim": hidden_dim,
        "max_seq_length": max_seq_length,
        "use_original_length": False,
        "batch_size": batch_size,
        "batch_sampler": "random",
        "learning_rate": 1e-4,
        "weight_decay": 0.01 if hidden_dim >= 512 else 1e-4,
        "num_epochs": 100,
        "warmup_epochs": 10 if num_train_samples < 500 else 5,
        "dropout": 0.25,
        "patience": 10,
        "num_classes": num_classes,
        "survival_loss": survival_loss,
    }
    if survival_loss == "nllsurv":
        cfg["nll_bins"] = nll_bins
    return cfg
