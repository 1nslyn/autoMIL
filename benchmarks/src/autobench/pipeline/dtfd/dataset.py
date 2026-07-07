"""Trivial DTFD dataset: full ``[N, embed_dim]`` H5 bag + int label per slide.

DTFD's pseudo-bag split is random *per forward* (a trainer concern, not a data
artifact), so the dataset just returns the whole bag. Uses the SAME split CSVs
as every other arm (``splits/<strategy>/<task>/splits_<fold>.csv``) so folds are
identical across the roster -- the fairness precondition. H5 layout matches
``nnmil/prepare.py``: ``features_<encoder_key>/<slide_id>.h5`` with a
``"features"`` dataset of shape ``[N, embed_dim]``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class DTFDSlide:
    """One bag: slide id, ``[N, embed_dim]`` float32 features, int label."""

    slide_id: str
    features: torch.Tensor  # [N, embed_dim], float32, CPU
    label: int


@dataclass(frozen=True)
class DTFDSurvivalSlide:
    """One survival bag: slide id, features, event status, time, patient id."""

    slide_id: str
    features: torch.Tensor  # [N, embed_dim], float32, CPU
    status: int  # 1=event, 0=censored
    time: float
    patient_id: str


def _read_bag(h5_path: str) -> torch.Tensor:
    with h5py.File(h5_path, "r") as f:
        feats = np.asarray(f["features"][:], dtype=np.float32)
    return torch.from_numpy(feats)


def _load_split_ids(split_csv: str, column: str) -> list[str]:
    """Read one split column, coercing numeric-looking ids back to bare strings."""
    df = pd.read_csv(split_csv, dtype=str)
    if column not in df.columns:
        return []
    ids = df[column].dropna().tolist()
    # A slide_id read as "1234.0" (pandas float coercion) -> "1234".
    return [str(x).split(".")[0] if "." in str(x) else str(x) for x in ids]


def load_dtfd_split(
    task_csv: str,
    split_csv: str,
    h5_dir: str,
    label_dict: dict[str, int],
    split: str,
) -> list[DTFDSlide]:
    """Load one split (train/val/test) as a list of ``DTFDSlide`` bags.

    Slides missing an H5 file are dropped with a warning (mirrors the other
    arms' behaviour), so a partially-extracted cohort still runs.
    """
    task_df = pd.read_csv(task_csv, dtype=str)
    label_lookup: dict[str, int] = {}
    for _, row in task_df.iterrows():
        raw = row["label"]
        label_lookup[str(row["slide_id"])] = int(label_dict.get(raw, raw))

    slides: list[DTFDSlide] = []
    missing: list[str] = []
    for sid in _load_split_ids(split_csv, split):
        if sid not in label_lookup:
            continue
        h5_path = os.path.join(h5_dir, f"{sid}.h5")
        if not os.path.exists(h5_path):
            missing.append(sid)
            continue
        slides.append(
            DTFDSlide(slide_id=sid, features=_read_bag(h5_path), label=label_lookup[sid])
        )

    if missing:
        print(
            f"  [DTFD] {split}: dropping {len(missing)} slide(s) missing H5 "
            f"features (first: {missing[:3]!r})"
        )
    return slides


def min_bag_size(slides: list[DTFDSlide]) -> int:
    """Smallest patch count across a split (used to guard ``numGroup``)."""
    if not slides:
        return 0
    return min(int(s.features.shape[0]) for s in slides)


def load_dtfd_survival_split(
    task_csv: str,
    split_csv: str,
    h5_dir: str,
    split: str,
) -> list[DTFDSurvivalSlide]:
    """Load one split (train/val/test) as a list of ``DTFDSurvivalSlide`` bags.

    Mirrors ``load_dtfd_split`` but reads the survival task CSV's
    ``status``/``time``/``case_id`` columns instead of a classification
    ``label`` column (same contract as
    ``clam/dataset.py::load_survival_fold_splits``).
    """
    task_df = pd.read_csv(task_csv, dtype=str)
    status_map = dict(zip(task_df["slide_id"], task_df["status"]))
    time_map = dict(zip(task_df["slide_id"], task_df["time"]))
    case_map = dict(zip(task_df["slide_id"], task_df["case_id"]))

    slides: list[DTFDSurvivalSlide] = []
    missing: list[str] = []
    for sid in _load_split_ids(split_csv, split):
        if sid not in status_map:
            continue
        h5_path = os.path.join(h5_dir, f"{sid}.h5")
        if not os.path.exists(h5_path):
            missing.append(sid)
            continue
        slides.append(
            DTFDSurvivalSlide(
                slide_id=sid,
                features=_read_bag(h5_path),
                status=int(status_map[sid]),
                time=float(time_map[sid]),
                patient_id=str(case_map[sid]),
            )
        )

    if missing:
        print(
            f"  [DTFD-surv] {split}: dropping {len(missing)} slide(s) missing H5 "
            f"features (first: {missing[:3]!r})"
        )
    return slides
