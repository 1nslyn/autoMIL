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
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import h5py
import numpy as np
import pandas as pd
import torch

from autobench.pipeline.dataset_guards import check_split_retention


@dataclass(frozen=True)
class DTFDSlide:
    """One bag: slide id, H5 path, int label.

    Holds the H5 *path*, not the loaded tensor -- features are read on demand
    (``_read_bag``) in the trainer, so a fold never has every bag resident in
    host RAM at once. Same contract as ``DTFDSurvivalSlide`` below, and as CLAM
    and nnMIL in both task types.

    This used to hold the tensor. At virchow2 (2560-dim, ~7.4k patches/slide)
    that is ~76 MB per slide, so a 372-slide train split sat at ~28 GB for the
    whole fold -- and the runner keeps train+val+test alive at once, then
    builds the next fold's split before rebinding, doubling it at the fold
    boundary. Concurrent cells were OOM-killed by the cgroup exactly there
    (``oom-kill: constraint=CONSTRAINT_MEMCG``, no traceback, worker log
    stopping mid-fold). Reading on demand also lets the OS page cache serve
    every worker on the node from ONE copy instead of each holding its own.
    """

    slide_id: str
    h5_path: str
    label: int


@dataclass(frozen=True)
class DTFDSurvivalSlide:
    """One survival bag: slide id, H5 path, event status, time, patient id.

    Holds the H5 *path*, not the loaded tensor -- features are read on demand
    (``_read_bag``) in the trainer so a fold never has every bag resident in
    host RAM at once (mirrors CLAM's lazy ``pt_path`` survival contract).
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
    # Strip only a trailing ".0" (float-coercion artifact for purely-numeric
    # ids). Do NOT split on "." — TCGA slide_ids carry a "." before the UUID
    # suffix (e.g. "TCGA-..-DX1.<uuid>"); splitting drops the UUID so the id
    # matches no H5 file / label and the split silently empties.
    return [s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s
            for s in map(str, ids)]


def load_dtfd_split(
    task_csv: str,
    split_csv: str,
    h5_dir: str,
    label_dict: dict[str, int],
    split: str,
) -> list[DTFDSlide]:
    """Load one split (train/val/test) as a list of ``DTFDSlide`` bags.

    Slides missing an H5 file are dropped with a warning (mirrors the other
    arms' behaviour), so a partially-extracted cohort still runs -- but M-9's
    retained-fraction / per-class-floor guard (``check_split_retention``)
    fails loudly on val/test if the drop is severe enough to corrupt the
    split. Train only warns (see that function's docstring for why).
    """
    task_df = pd.read_csv(task_csv, dtype=str)
    label_lookup: dict[str, int] = {}
    for _, row in task_df.iterrows():
        raw = row["label"]
        label_lookup[str(row["slide_id"])] = int(label_dict.get(raw, raw))

    expected_ids = _load_split_ids(split_csv, split)
    slides: list[DTFDSlide] = []
    missing: list[str] = []
    for sid in expected_ids:
        if sid not in label_lookup:
            continue
        h5_path = os.path.join(h5_dir, f"{sid}.h5")
        if not os.path.exists(h5_path):
            missing.append(sid)
            continue
        slides.append(
            DTFDSlide(slide_id=sid, h5_path=h5_path, label=label_lookup[sid])
        )

    if missing:
        print(
            f"  [DTFD] {split}: dropping {len(missing)} slide(s) missing H5 "
            f"features (first: {missing[:3]!r})"
        )

    check_split_retention(
        context=f"DTFD task_csv={task_csv} split_csv={split_csv}",
        split=split,
        expected_total=len(expected_ids),
        retained_total=len(slides),
        expected_by_class=Counter(
            label_lookup[sid] for sid in expected_ids if sid in label_lookup
        ),
        retained_by_class=Counter(s.label for s in slides),
        warn_only=(split == "train"),
    )
    return slides


def min_bag_size(
    slides: Sequence[DTFDSlide | DTFDSurvivalSlide],
) -> int:
    """Smallest patch count across a split (used to guard ``numGroup``).

    Reads only the H5 dataset's shape metadata, never its contents, so guarding
    a split costs no RAM. Both slide types are lazy now, so there is no longer
    an ``.features`` branch to fall back to.
    """
    if not slides:
        return 0

    def _n(s: DTFDSlide | DTFDSurvivalSlide) -> int:
        with h5py.File(s.h5_path, "r") as f:
            return int(f["features"].shape[0])

    return min(_n(s) for s in slides)


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
    ``clam/dataset.py::load_survival_fold_splits``). Same retained-fraction
    guard (M-9); no per-class floor since survival tasks have no classes.
    """
    task_df = pd.read_csv(task_csv, dtype=str)
    status_map = dict(zip(task_df["slide_id"], task_df["status"]))
    time_map = dict(zip(task_df["slide_id"], task_df["time"]))
    case_map = dict(zip(task_df["slide_id"], task_df["case_id"]))

    expected_ids = _load_split_ids(split_csv, split)
    slides: list[DTFDSurvivalSlide] = []
    missing: list[str] = []
    for sid in expected_ids:
        if sid not in status_map:
            continue
        h5_path = os.path.join(h5_dir, f"{sid}.h5")
        if not os.path.exists(h5_path):
            missing.append(sid)
            continue
        slides.append(
            DTFDSurvivalSlide(
                slide_id=sid,
                h5_path=h5_path,
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

    check_split_retention(
        context=f"DTFD-surv task_csv={task_csv} split_csv={split_csv}",
        split=split,
        expected_total=len(expected_ids),
        retained_total=len(slides),
        warn_only=(split == "train"),
    )
    return slides
