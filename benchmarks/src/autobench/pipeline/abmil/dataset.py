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
    # Strip only a trailing ".0" (float-coercion artifact for purely-numeric
    # ids). Do NOT split on "." — TCGA slide_ids carry a "." before the UUID
    # suffix (e.g. "TCGA-..-DX1.<uuid>"); splitting drops the UUID so the id
    # matches no H5 file / label and the split silently empties.
    return [s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s
            for s in map(str, ids)]


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
