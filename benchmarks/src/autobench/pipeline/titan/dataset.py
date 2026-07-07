"""TITAN slide-embedding dataset: one frozen vector per slide, no bag.

Given a fold's split CSV column (train/val/test slide_ids) and the
``features_titan/`` directory, returns ``(embedding[D], label)`` pairs --
the simplest possible dataset shape in this benchmark, since TITAN itself
already pooled each slide into a single vector.
"""

from __future__ import annotations

import os

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset

from autobench.pipeline.titan.prepare import _FEATURE_KEYS


def _load_embedding(h5_path: str) -> torch.Tensor:
    """Load a slide embedding, squeezing an optional leading singleton dim."""
    with h5py.File(h5_path, "r") as f:
        key = next((k for k in _FEATURE_KEYS if k in f), None)
        if key is None:
            raise ValueError(
                f"Slide feature file {h5_path} has no recognized dataset "
                f"key (looked for {_FEATURE_KEYS!r}, found {list(f.keys())!r})."
            )
        arr = f[key][()]

    tensor = torch.as_tensor(arr, dtype=torch.float32)
    if tensor.dim() == 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor


class TitanSlideDataset(Dataset):
    """Maps slide_ids from a split column to ``(embedding, label)`` pairs."""

    def __init__(
        self,
        slide_ids: list[str],
        labels: list[int],
        features_dir: str,
    ) -> None:
        if len(slide_ids) != len(labels):
            raise ValueError(
                f"slide_ids ({len(slide_ids)}) and labels ({len(labels)}) "
                "must have the same length."
            )
        self.slide_ids = slide_ids
        self.labels = labels
        self.features_dir = features_dir

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        sid = self.slide_ids[idx]
        h5_path = os.path.join(self.features_dir, f"{sid}.h5")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(
                f"Missing TITAN slide feature for {sid!r}: {h5_path}"
            )
        embedding = _load_embedding(h5_path)
        return embedding, self.labels[idx]


class TitanSurvivalDataset(Dataset):
    """Maps slide_ids from a split column to ``(embedding, status, time, patient_id)``."""

    def __init__(
        self,
        slide_ids: list[str],
        statuses: list[int],
        times: list[float],
        patient_ids: list[str],
        features_dir: str,
    ) -> None:
        if not (len(slide_ids) == len(statuses) == len(times) == len(patient_ids)):
            raise ValueError(
                f"slide_ids ({len(slide_ids)}), statuses ({len(statuses)}), "
                f"times ({len(times)}), and patient_ids ({len(patient_ids)}) "
                "must have the same length."
            )
        self.slide_ids = slide_ids
        self.statuses = statuses
        self.times = times
        self.patient_ids = patient_ids
        self.features_dir = features_dir

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, float, str]:
        sid = self.slide_ids[idx]
        h5_path = os.path.join(self.features_dir, f"{sid}.h5")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(
                f"Missing TITAN slide feature for {sid!r}: {h5_path}"
            )
        embedding = _load_embedding(h5_path)
        return embedding, self.statuses[idx], self.times[idx], self.patient_ids[idx]


def build_survival_split_dataset(
    split_csv: str,
    split_name: str,
    task_df: pd.DataFrame,
    features_dir: str,
) -> TitanSurvivalDataset:
    """Build a ``TitanSurvivalDataset`` for one split column of a fold CSV.

    Mirrors ``build_split_dataset`` but reads the survival task CSV's
    ``status``/``time``/``case_id`` columns instead of a categorical
    ``label`` (same contract as
    ``clam/dataset.py::load_survival_fold_splits``).
    """
    split_df = pd.read_csv(split_csv)
    if split_name not in split_df.columns:
        raise ValueError(
            f"Split CSV {split_csv} has no column {split_name!r} "
            f"(columns: {list(split_df.columns)!r})."
        )
    slide_ids = split_df[split_name].dropna().tolist()

    status_lookup = dict(zip(task_df["slide_id"], task_df["status"]))
    time_lookup = dict(zip(task_df["slide_id"], task_df["time"]))
    case_lookup = dict(zip(task_df["slide_id"], task_df["case_id"]))

    missing = [sid for sid in slide_ids if sid not in status_lookup]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = f" ... +{len(missing) - 5}" if len(missing) > 5 else ""
        raise ValueError(
            f"Split CSV {split_csv} references {len(missing)} slide_ids "
            f"absent from the survival task CSV: {preview}{suffix}"
        )

    statuses = [int(status_lookup[sid]) for sid in slide_ids]
    times = [float(time_lookup[sid]) for sid in slide_ids]
    patient_ids = [str(case_lookup[sid]) for sid in slide_ids]
    return TitanSurvivalDataset(slide_ids, statuses, times, patient_ids, features_dir)


def build_split_dataset(
    split_csv: str,
    split_name: str,
    task_df: pd.DataFrame,
    label_dict: dict[str, int],
    features_dir: str,
) -> TitanSlideDataset:
    """Build a ``TitanSlideDataset`` for one split column of a fold CSV.

    *split_csv* is a ``splits_<fold>.csv`` with columns
    ``[train, val, test]`` (the shared split format from
    ``pipeline/splits.py``); *split_name* selects one of those columns.
    *task_df* provides the slide_id -> label lookup (the task CSV), and
    *label_dict* maps string labels to ints, matching every other arm.
    """
    split_df = pd.read_csv(split_csv)
    if split_name not in split_df.columns:
        raise ValueError(
            f"Split CSV {split_csv} has no column {split_name!r} "
            f"(columns: {list(split_df.columns)!r})."
        )
    slide_ids = split_df[split_name].dropna().tolist()

    lookup: dict[str, int] = {}
    for _, row in task_df.iterrows():
        raw_label = row["label"]
        label_int = label_dict.get(raw_label, raw_label)
        if isinstance(label_int, str):
            label_int = int(label_int)
        lookup[row["slide_id"]] = int(label_int)

    missing = [sid for sid in slide_ids if sid not in lookup]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = f" ... +{len(missing) - 5}" if len(missing) > 5 else ""
        raise ValueError(
            f"Split CSV {split_csv} references {len(missing)} slide_ids "
            f"absent from the task CSV: {preview}{suffix}"
        )

    labels = [lookup[sid] for sid in slide_ids]
    return TitanSlideDataset(slide_ids, labels, features_dir)
