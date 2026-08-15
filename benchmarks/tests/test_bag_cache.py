"""The derived bag cache must be faster without ever being different.

Measured motivation (H100, real bag, real model): the H5 read is 95% of a
training step while the GPU does ~1 ms of it, because the feature store is
chunked ``(1, dim)`` -- one chunk per patch, thousands of B-tree lookups per
bag. A contiguous memory-mapped copy makes that ~0.

Everything here is about the cache being *invisible*: same bytes, off by
default, and degrading to the plain H5 read on any failure rather than taking
the run down with it.
"""

from __future__ import annotations

import os

import h5py
import numpy as np
import pytest
import torch

from autobench.pipeline import bag_cache


@pytest.fixture()
def bag(tmp_path):
    """One real H5 bag, chunked the way the frozen feature store chunks them."""
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((37, 16)).astype("float32")
    path = tmp_path / "slide.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=feats, chunks=(1, 16))
    return str(path), torch.from_numpy(feats)


def _enable(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    monkeypatch.setenv("AUTOBENCH_BAG_CACHE", str(cache))
    bag_cache._warned.clear()
    return cache


class TestBytesAreIdentical:
    def test_disabled_matches_enabled_first_read(self, bag, tmp_path, monkeypatch):
        path, expected = bag
        monkeypatch.delenv("AUTOBENCH_BAG_CACHE", raising=False)
        assert torch.equal(bag_cache.read_bag(path), expected)

        _enable(monkeypatch, tmp_path)
        assert torch.equal(bag_cache.read_bag(path), expected)

    def test_cached_read_matches_uncached(self, bag, tmp_path, monkeypatch):
        """The second read comes off the mapping -- it must still be identical."""
        path, expected = bag
        _enable(monkeypatch, tmp_path)
        bag_cache.read_bag(path)                      # populate
        got = bag_cache.read_bag(path)                # serve from mmap
        assert torch.equal(got, expected)
        assert got.dtype == expected.dtype
        assert got.shape == expected.shape


class TestItActuallyCaches:
    def test_second_read_opens_no_h5(self, bag, tmp_path, monkeypatch):
        """The whole point: the H5 is touched once, not once per epoch."""
        path, _ = bag
        _enable(monkeypatch, tmp_path)
        bag_cache.read_bag(path)

        opens: list[str] = []
        real = bag_cache.h5py.File

        def spy(p, *a, **k):
            opens.append(str(p))
            return real(p, *a, **k)

        monkeypatch.setattr(bag_cache.h5py, "File", spy)
        bag_cache.read_bag(path)
        assert opens == [], "a cached bag must not reopen the H5"

    def test_disabled_by_default_reads_h5_every_time(self, bag, tmp_path, monkeypatch):
        """Unset means today's behaviour exactly -- no surprise caching."""
        path, _ = bag
        monkeypatch.delenv("AUTOBENCH_BAG_CACHE", raising=False)

        opens: list[str] = []
        real = bag_cache.h5py.File
        monkeypatch.setattr(
            bag_cache.h5py, "File",
            lambda p, *a, **k: (opens.append(str(p)), real(p, *a, **k))[1],
        )
        bag_cache.read_bag(path)
        bag_cache.read_bag(path)
        assert len(opens) == 2


class TestStalenessAndFailure:
    def test_regenerated_source_invalidates_the_entry(self, bag, tmp_path, monkeypatch):
        """Lazy reading made feature immutability load-bearing.

        The key carries size and mtime, so a regenerated feature file is
        DETECTED rather than trusted -- the alternative is silently training on
        yesterday's bytes.
        """
        path, _ = bag
        _enable(monkeypatch, tmp_path)
        bag_cache.read_bag(path)
        first_key = bag_cache._cache_key(path)

        replacement = np.full((37, 16), 7.0, dtype="float32")
        with h5py.File(path, "w") as f:
            f.create_dataset("features", data=replacement, chunks=(1, 16))
        os.utime(path, (0, 0))  # force a different mtime, not merely a later one

        assert bag_cache._cache_key(path) != first_key
        assert torch.equal(bag_cache.read_bag(path), torch.from_numpy(replacement))

    def test_corrupt_entry_is_rebuilt_not_fatal(self, bag, tmp_path, monkeypatch):
        path, expected = bag
        cache = _enable(monkeypatch, tmp_path)
        bag_cache.read_bag(path)

        entry = os.path.join(str(cache), bag_cache._cache_key(path) + ".npy")
        with open(entry, "wb") as fh:
            fh.write(b"not an npy file")

        assert torch.equal(bag_cache.read_bag(path), expected)

    def test_unwritable_cache_degrades_to_h5(self, bag, tmp_path, monkeypatch):
        """A full disk must slow the run down, not end it."""
        path, expected = bag
        _enable(monkeypatch, tmp_path)

        def boom(*_a, **_k):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(bag_cache, "_write_atomically", boom)
        assert torch.equal(bag_cache.read_bag(path), expected)

    def test_no_partial_file_survives_a_failed_write(self, bag, tmp_path, monkeypatch):
        """A reader must never map a half-written entry."""
        path, _ = bag
        cache = _enable(monkeypatch, tmp_path)
        target = os.path.join(str(cache), "x.npy")

        def explode(*_a, **_k):
            raise OSError("disk died mid-write")

        monkeypatch.setattr(bag_cache.np, "save", explode)
        with pytest.raises(OSError):
            bag_cache._write_atomically(target, np.zeros((2, 2), dtype="float32"))

        assert not os.path.exists(target)
        leftovers = [f for f in os.listdir(str(cache)) if f.startswith(".bag-")]
        assert leftovers == [], f"temp files left behind: {leftovers}"


def test_arms_read_through_the_cache():
    """ABMIL and DTFD must route through it, or the fix reaches nothing.

    Both are the arms that read chunked H5. CLAM reads contiguous .pt and nnMIL
    reads through its vendored loader, which is why neither is covered here.
    """
    import inspect

    from autobench.pipeline.abmil import dataset as abmil_ds
    from autobench.pipeline.dtfd import dataset as dtfd_ds

    for module in (abmil_ds, dtfd_ds):
        source = inspect.getsource(module._read_bag)
        assert "read_bag(" in source, (
            f"{module.__name__}._read_bag bypasses the shared cache"
        )
