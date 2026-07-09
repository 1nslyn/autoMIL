"""Trivial ABMIL dataset: full ``[N, embed_dim]`` H5 bag + int label per slide.

Uses the SAME split CSVs as every other arm (``splits/<strategy>/<task>/
splits_<fold>.csv``) so folds are identical across the roster -- the fairness
precondition. H5 layout matches ``nnmil/prepare.py``:
``features_<encoder_key>/<slide_id>.h5`` with a ``"features"`` dataset of
shape ``[N, embed_dim]``. Clone of ``dtfd/dataset.py`` (identical contract).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class ABMILSlide:
    """One bag: slide id, ``[N, embed_dim]`` float32 features, int label."""

    slide_id: str
    features: torch.Tensor  # [N, embed_dim], float32, CPU
    label: int


@dataclass(frozen=True)
class ABMILSurvivalSlide:
    """One survival bag: slide id, H5 path, event status, time, patient id.

    Holds the H5 path, not the loaded tensor -- features are read on demand
    (``_read_bag``) in the trainer so a fold never has every bag resident in
    host RAM at once.
    """

    slide_id: str
    h5_path: str
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
    # Undo pandas float coercion of purely-numeric ids ("1234.0" -> "1234"),
    # but keep real ids intact: TCGA slide_ids legitimately contain a dot
    # ("...-DX1.<uuid>"), so a blanket split(".")[0] truncates them and orphans
    # every slide against the task CSV.
    def _bare(x: str) -> str:
        s = str(x)
        return s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s
    return [_bare(x) for x in ids]


def load_abmil_split(
    task_csv: str,
    split_csv: str,
    h5_dir: str,
    label_dict: dict[str, int],
    split: str,
) -> list[ABMILSlide]:
    """Load one split (train/val/test) as a list of ``ABMILSlide`` bags.

    Slides missing an H5 file are dropped with a warning (mirrors the other
    arms' behaviour), so a partially-extracted cohort still runs.
    """
    task_df = pd.read_csv(task_csv, dtype=str)
    label_lookup: dict[str, int] = {}
    for _, row in task_df.iterrows():
        raw = row["label"]
        label_lookup[str(row["slide_id"])] = int(label_dict.get(raw, raw))

    slides: list[ABMILSlide] = []
    missing: list[str] = []
    for sid in _load_split_ids(split_csv, split):
        if sid not in label_lookup:
            continue
        h5_path = os.path.join(h5_dir, f"{sid}.h5")
        if not os.path.exists(h5_path):
            missing.append(sid)
            continue
        slides.append(
            ABMILSlide(slide_id=sid, features=_read_bag(h5_path), label=label_lookup[sid])
        )

    if missing:
        print(
            f"  [ABMIL] {split}: dropping {len(missing)} slide(s) missing H5 "
            f"features (first: {missing[:3]!r})"
        )
    return slides


def load_abmil_survival_split(
    task_csv: str,
    split_csv: str,
    h5_dir: str,
    split: str,
) -> list[ABMILSurvivalSlide]:
    """Load one split (train/val/test) as a list of ``ABMILSurvivalSlide`` bags."""
    task_df = pd.read_csv(task_csv, dtype=str)
    status_map = dict(zip(task_df["slide_id"], task_df["status"]))
    time_map = dict(zip(task_df["slide_id"], task_df["time"]))
    case_map = dict(zip(task_df["slide_id"], task_df["case_id"]))

    slides: list[ABMILSurvivalSlide] = []
    missing: list[str] = []
    for sid in _load_split_ids(split_csv, split):
        if sid not in status_map:
            continue
        h5_path = os.path.join(h5_dir, f"{sid}.h5")
        if not os.path.exists(h5_path):
            missing.append(sid)
            continue
        slides.append(
            ABMILSurvivalSlide(
                slide_id=sid,
                h5_path=h5_path,
                status=int(status_map[sid]),
                time=float(time_map[sid]),
                patient_id=str(case_map[sid]),
            )
        )

    if missing:
        print(
            f"  [ABMIL-surv] {split}: dropping {len(missing)} slide(s) missing H5 "
            f"features (first: {missing[:3]!r})"
        )
    return slides
