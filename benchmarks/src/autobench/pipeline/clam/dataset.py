"""Dataset helpers that wrap CLAM's Generic_MIL_Dataset."""

from __future__ import annotations

import os
from collections import Counter

import pandas as pd

from autobench.pipeline.clam._imports import Generic_MIL_Dataset, Generic_Split
from autobench.pipeline.config import ExperimentConfig
from autobench.pipeline.dataset_guards import check_split_retention


def create_dataset(
    exp_cfg: ExperimentConfig,
    benchmark_dir: str,
    task_csv_name: str | None = None,
) -> Generic_MIL_Dataset:
    """Instantiate a CLAM Generic_MIL_Dataset for the given experiment.

    *task_csv_name* overrides the default task CSV filename (without .csv
    extension).  For multi-cohort strategies, pass e.g. ``"brca_all"``.
    """
    if task_csv_name is None:
        task_csv_name = exp_cfg.task.name
    csv_path = os.path.join(benchmark_dir, "dataset_csv", f"{task_csv_name}.csv")
    data_dir = os.path.join(benchmark_dir, "features", exp_cfg.encoder_key)

    dataset = Generic_MIL_Dataset(
        data_dir=data_dir,
        csv_path=csv_path,
        shuffle=False,
        seed=exp_cfg.train.seed,
        print_info=False,
        label_dict=exp_cfg.task.label_dict,
        label_col="label",
        patient_strat=False,
        ignore=[],
    )

    # Filter out slides missing .pt feature files (e.g. corrupted WSIs
    # that were silently skipped during feature extraction).
    pt_dir = os.path.join(data_dir, "pt_files")
    mask = dataset.slide_data["slide_id"].apply(
        lambda sid: os.path.exists(os.path.join(pt_dir, f"{sid}.pt"))
    )
    n_missing = (~mask).sum()
    if n_missing > 0:
        missing = dataset.slide_data.loc[~mask, "slide_id"].tolist()
        print(f"[WARNING] Dropping {n_missing} slides missing .pt features: {missing}")
        dataset.slide_data = dataset.slide_data[mask].reset_index(drop=True)
        dataset.cls_ids_prep()

    return dataset


def load_fold_splits(
    dataset: Generic_MIL_Dataset,
    benchmark_dir: str,
    splits_subdir: str,
    fold: int,
    *,
    task_csv_name: str | None = None,
) -> tuple[Generic_Split, Generic_Split, Generic_Split]:
    """Load pre-generated split CSV and return (train, val, test) splits.

    *splits_subdir* is the relative path under ``benchmark_dir/splits/``,
    e.g. ``"brca"`` (legacy) or ``"a/brca"`` (strategy-based).

    M-9: asserts each split retained enough of the slides assigned to it.
    CLAM's own missing-``.pt`` filter runs once, GLOBALLY, inside
    ``create_dataset`` -- before any fold is chosen -- so a slide dropped
    there is no longer visible in ``dataset.slide_data`` by the time this
    function runs; ``return_splits``' internal masking against the (already
    filtered) ``dataset.slide_data`` would otherwise make the drop invisible
    here too. This reads the raw split CSV independently to recover the
    TRUE per-split assignment. *task_csv_name*, if given (the caller's own
    task CSV, e.g. ``exp_cfg.task.name``), additionally enables the
    per-class floor by resolving each expected slide's label from the task
    CSV rather than only from what survived; omit it to skip that half (the
    retained-fraction check still runs).
    """
    split_csv = os.path.join(benchmark_dir, "splits", splits_subdir, f"splits_{fold}.csv")
    train_split, val_split, test_split = dataset.return_splits(
        from_id=False, csv_path=split_csv
    )

    raw_splits = pd.read_csv(split_csv, dtype=str)

    expected_label_by_sid: dict[str, int] | None = None
    if task_csv_name is not None:
        task_csv_path = os.path.join(benchmark_dir, "dataset_csv", f"{task_csv_name}.csv")
        task_df = pd.read_csv(task_csv_path, dtype=str)
        expected_label_by_sid = {
            str(sid): int(dataset.label_dict.get(raw, raw))
            for sid, raw in zip(task_df["slide_id"], task_df["label"])
        }

    for split_name, split_obj in (
        ("train", train_split), ("val", val_split), ("test", test_split),
    ):
        if split_name not in raw_splits.columns:
            continue
        expected_ids = raw_splits[split_name].dropna().tolist()
        expected_by_class = None
        retained_by_class = None
        if expected_label_by_sid is not None:
            expected_by_class = Counter(
                expected_label_by_sid[sid]
                for sid in expected_ids if sid in expected_label_by_sid
            )
            retained_by_class = Counter(
                split_obj.slide_data["label"].tolist() if split_obj is not None else []
            )
        check_split_retention(
            context=f"CLAM split_csv={split_csv}",
            split=split_name,
            expected_total=len(expected_ids),
            retained_total=len(split_obj) if split_obj is not None else 0,
            expected_by_class=expected_by_class,
            retained_by_class=retained_by_class,
            warn_only=(split_name == "train"),
        )

    return train_split, val_split, test_split


def load_survival_fold_splits(
    exp_cfg: ExperimentConfig,
    benchmark_dir: str,
    fold: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Load (train, val, test) survival samples for one fold, each a dict
    ``{slide_id, pt_path, status, time, patient_id}``. Joins the survival task
    CSV (status/time/case_id) with the fold split CSV; skips slides missing a
    ``.pt``. Bypasses CLAM's classification dataset (which needs a label_dict).

    M-9: asserts each split retained enough of its assigned slides (no
    per-class floor -- survival tasks have no classes).
    """
    task_csv = os.path.join(benchmark_dir, "dataset_csv", f"{exp_cfg.task.name}.csv")
    df = pd.read_csv(task_csv)
    status_map = dict(zip(df["slide_id"], df["status"]))
    time_map = dict(zip(df["slide_id"], df["time"]))
    case_map = dict(zip(df["slide_id"], df["case_id"]))

    pt_dir = os.path.join(benchmark_dir, "features", exp_cfg.encoder_key, "pt_files")
    split_csv = os.path.join(
        benchmark_dir, "splits", exp_cfg.strategy, exp_cfg.task.name, f"splits_{fold}.csv"
    )
    sdf = pd.read_csv(split_csv)

    def _build(col: str) -> list[dict]:
        samples: list[dict] = []
        if col not in sdf.columns:
            return samples
        expected_ids = sdf[col].dropna().tolist()
        for sid in expected_ids:
            pt_path = os.path.join(pt_dir, f"{sid}.pt")
            if sid not in status_map or not os.path.exists(pt_path):
                continue
            samples.append({
                "slide_id": sid,
                "pt_path": pt_path,
                "status": int(status_map[sid]),
                "time": float(time_map[sid]),
                "patient_id": str(case_map[sid]),
            })
        check_split_retention(
            context=f"CLAM-surv split_csv={split_csv}",
            split=col,
            expected_total=len(expected_ids),
            retained_total=len(samples),
            warn_only=(col == "train"),
        )
        return samples

    return _build("train"), _build("val"), _build("test")
