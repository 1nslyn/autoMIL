"""Bags are read on demand, never held for a whole split.

Loading a split must cost O(1) in feature bytes. When ABMIL and DTFD
classification held ``features: torch.Tensor`` in the slide dataclass instead,
one worker carried its entire train split: at virchow2 (2560-dim, ~7.4k
patches/slide) that is ~76 MB per slide, so a 372-slide split sat at ~28 GB for
the fold. The runner keeps train+val+test alive at once and builds the next
fold's split before rebinding, so the fold boundary doubled it.

Concurrent cells were then OOM-killed by the cgroup right there --
``oom-kill: constraint=CONSTRAINT_MEMCG``, no Python traceback, the worker log
simply stopping mid-fold. Because the orchestrator raises on the resulting
"returned None", one kill tore down the whole block and stranded every cell
still queued behind it.

CLAM, nnMIL and both survival paths were already lazy; classification for these
two arms was the outlier. These tests keep it that way.
"""

from __future__ import annotations

import dataclasses
import re
import typing

import h5py
import numpy as np
import pandas as pd
import pytest

from autobench.pipeline.abmil import dataset as abmil_ds
from autobench.pipeline.dtfd import dataset as dtfd_ds

_SLIDE_TYPES = [
    pytest.param(abmil_ds.ABMILSlide, id="ABMILSlide"),
    pytest.param(abmil_ds.ABMILSurvivalSlide, id="ABMILSurvivalSlide"),
    pytest.param(dtfd_ds.DTFDSlide, id="DTFDSlide"),
    pytest.param(dtfd_ds.DTFDSurvivalSlide, id="DTFDSurvivalSlide"),
]


@pytest.mark.parametrize("slide_type", _SLIDE_TYPES)
def test_slide_holds_a_path_not_a_tensor(slide_type) -> None:
    """No slide dataclass may carry the bag itself.

    Matches the rendered annotation rather than ``__name__``: a union such as
    ``torch.Tensor | None`` has no ``__name__``, so an identity check on it
    silently passes and the field slips back in.
    """
    hints = typing.get_type_hints(slide_type)
    tensor_fields = sorted(
        f.name for f in dataclasses.fields(slide_type)
        if re.search(r"Tensor|ndarray", str(hints.get(f.name, "")))
    )
    assert not tensor_fields, (
        f"{slide_type.__name__} holds {tensor_fields} in the dataclass, so every "
        "bag in a split stays resident for the whole fold. Hold h5_path and read "
        "through _read_bag instead."
    )
    assert "h5_path" in hints, f"{slide_type.__name__} has no h5_path to read from"


def _cohort(tmp_path, n_slides: int, n_patches: int, dim: int):
    """A tiny on-disk cohort: task csv, split csv, and real H5 bags."""
    rng = np.random.default_rng(0)
    tmp_path.mkdir(parents=True, exist_ok=True)
    h5_dir = tmp_path / "h5"
    h5_dir.mkdir()
    sids = [f"s{i}" for i in range(n_slides)]
    for i, sid in enumerate(sids):
        feats = rng.standard_normal((n_patches, dim)).astype("float32") + (i % 2)
        with h5py.File(h5_dir / f"{sid}.h5", "w") as f:
            f.create_dataset("features", data=feats)

    task_csv = tmp_path / "task.csv"
    pd.DataFrame({"slide_id": sids, "label": [str(i % 2) for i in range(n_slides)]}).to_csv(
        task_csv, index=False,
    )
    split_csv = tmp_path / "splits_0.csv"
    pd.DataFrame({"train": sids, "val": sids, "test": sids}).to_csv(split_csv, index=False)
    return str(task_csv), str(split_csv), str(h5_dir)


@pytest.mark.parametrize(
    "module,loader",
    [
        pytest.param(abmil_ds, "load_abmil_split", id="abmil"),
        pytest.param(dtfd_ds, "load_dtfd_split", id="dtfd"),
    ],
)
def test_loading_a_split_reads_no_feature_bytes(module, loader, tmp_path, monkeypatch):
    """The loader must not touch bag CONTENTS -- that is the whole bug.

    Spying on ``_read_bag`` is the precise check: the eager version called it
    once per slide while building the split, which is what put the entire split
    in RAM before a single training step ran.
    """
    task_csv, split_csv, h5_dir = _cohort(tmp_path, n_slides=6, n_patches=8, dim=4)

    calls: list[str] = []
    real = module._read_bag
    monkeypatch.setattr(
        module, "_read_bag", lambda p: (calls.append(p), real(p))[1],
    )

    slides = getattr(module, loader)(
        task_csv, split_csv, h5_dir, {"0": 0, "1": 1}, "train",
    )

    assert len(slides) == 6
    assert calls == [], (
        f"{loader} read {len(calls)} bag(s) while building the split; loading a "
        "split must be O(1) in feature bytes"
    )
    assert all(isinstance(s.h5_path, str) for s in slides)


@pytest.mark.parametrize(
    "module,loader",
    [
        pytest.param(abmil_ds, "load_abmil_split", id="abmil"),
        pytest.param(dtfd_ds, "load_dtfd_split", id="dtfd"),
    ],
)
def test_split_footprint_is_independent_of_bag_size(module, loader, tmp_path):
    """A 100x bigger bag must not make the loaded split bigger.

    The structural test says the field is a path; this says the consequence
    holds -- split size is decoupled from feature size, which is the property
    that lets many cells share a node.
    """
    import pickle

    small = _cohort(tmp_path / "small", 4, 8, 4)
    big = _cohort(tmp_path / "big", 4, 800, 4)

    sizes = []
    for args in (small, big):
        slides = getattr(module, loader)(*args, {"0": 0, "1": 1}, "train")
        sizes.append(len(pickle.dumps(slides)))

    small_size, big_size = sizes
    assert big_size < small_size * 2, (
        f"split grew {small_size} -> {big_size} bytes when bags got 100x bigger, "
        "so the bags are travelling with the split"
    )


def test_min_bag_size_reads_only_shape_metadata(tmp_path, monkeypatch):
    """Guarding numGroup must not pull whole bags into RAM either."""
    task_csv, split_csv, h5_dir = _cohort(tmp_path, n_slides=4, n_patches=16, dim=4)
    slides = dtfd_ds.load_dtfd_split(task_csv, split_csv, h5_dir, {"0": 0, "1": 1}, "train")

    calls: list[str] = []
    monkeypatch.setattr(dtfd_ds, "_read_bag", lambda p: calls.append(p))

    assert dtfd_ds.min_bag_size(slides) == 16
    assert calls == [], "min_bag_size read bag contents instead of the H5 shape"


def _count_h5_opens(monkeypatch, module) -> list[str]:
    """Spy on real file opens, not on a function name.

    Watching ``_read_bag`` is not enough. Memoizing it
    (``@functools.lru_cache``) leaves the call site intact while the bag stops
    being re-read, and an inline ``h5py.File`` in the loader stops calling it
    entirely. Both restore whole-split residency and both pass a name-level
    spy. File opens are the property.
    """
    opens: list[str] = []
    real = module.h5py.File

    def spy(path, *args, **kwargs):
        opens.append(str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(module.h5py, "File", spy)
    return opens


@pytest.mark.parametrize(
    "module,loader",
    [
        pytest.param(abmil_ds, "load_abmil_split", id="abmil"),
        pytest.param(dtfd_ds, "load_dtfd_split", id="dtfd"),
    ],
)
def test_loading_a_split_opens_no_bag_files(module, loader, tmp_path, monkeypatch):
    """Building a split must not open a single bag.

    Stronger than spying on ``_read_bag``: this also catches an inline
    ``h5py.File`` read that stashes tensors in a module-level dict.
    """
    task_csv, split_csv, h5_dir = _cohort(tmp_path, n_slides=6, n_patches=8, dim=4)
    opens = _count_h5_opens(monkeypatch, module)

    slides = getattr(module, loader)(
        task_csv, split_csv, h5_dir, {"0": 0, "1": 1}, "train",
    )

    assert len(slides) == 6
    assert opens == [], (
        f"{loader} opened {len(opens)} bag file(s) while building the split; "
        "loading a split must not touch bag contents at all"
    )


@pytest.mark.parametrize("arm", ["abmil", "dtfd"])
def test_bags_are_reread_every_epoch(arm, tmp_path, monkeypatch):
    """Training longer must read more.

    This is the half the structural tests cannot express. Under any memoization
    the open count is flat in epochs, so the split becomes resident again from
    epoch two onward -- worse than the original bug, because a module-level
    cache is keyed by path and survives the fold boundary that used to drop it.
    """
    import torch

    if arm == "abmil":
        from autobench.pipeline.abmil.config import ABMILConfig
        from autobench.pipeline.abmil.train import train_abmil_fold
        module = abmil_ds
        loader, label_dict = abmil_ds.load_abmil_split, {"0": 0, "1": 1}

        def run(epochs, slides):
            return train_abmil_fold(
                "abmil", *slides, embed_dim=4, num_classes=2,
                cfg=ABMILConfig(max_epochs=epochs, early_stopping=False),
                device=torch.device("cpu"), seed=0,
            )
    else:
        from autobench.pipeline.dtfd.config import DTFDConfig
        from autobench.pipeline.dtfd.train import train_dtfd_fold
        module = dtfd_ds
        loader, label_dict = dtfd_ds.load_dtfd_split, {"0": 0, "1": 1}

        def run(epochs, slides):
            return train_dtfd_fold(
                *slides, embed_dim=4, num_classes=2,
                cfg=DTFDConfig(numGroup=2, mDim=8, max_epochs=epochs, lr=1e-3,
                               early_stopping=False),
                device=torch.device("cpu"), seed=0,
            )

    args = _cohort(tmp_path, n_slides=6, n_patches=8, dim=4)
    splits = [loader(*args, label_dict, s) for s in ("train", "val", "test")]

    counts = []
    for epochs in (1, 3):
        opens = _count_h5_opens(monkeypatch, module)
        run(epochs, splits)
        counts.append(len(opens))
        monkeypatch.undo()

    one_epoch, three_epochs = counts
    assert one_epoch > 0, "no bag was ever read"
    assert three_epochs > one_epoch, (
        f"{arm} opened {one_epoch} bag file(s) for 1 epoch and {three_epochs} for "
        "3 -- the count is flat in epochs, so bags are being cached and the whole "
        "split is resident again"
    )
