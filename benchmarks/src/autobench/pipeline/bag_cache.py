"""Memory-mapped derived cache for feature bags.

MIL training on precomputed features is data-movement bound, not compute bound.
Measured on an H100 with a real bag and the real model:

    h5 read + float32 cast   62.9 ms   95.0%   [CPU]
    host->device transfer     2.3 ms    3.5%
    forward+backward+step     1.0 ms    1.5%   [GPU]

The GPU does about a millisecond of work per slide while the CPU spends sixty
reading the bag. The cause is the feature store's chunking -- ``chunks=(1, dim)``
means one HDF5 chunk per patch, so a single bag costs thousands of B-tree
lookups. Neither ``read_direct`` nor a 256 MB chunk cache helps (48.5 ms and
52.3 ms against a 50.0 ms baseline, measured): the cost is per-chunk, not
per-allocation. Rewriting the store with sane chunking would fix it at the
source, but that store is frozen substrate.

So convert once into a contiguous ``.npy`` and memory-map it thereafter:

    derived .npy MEMMAP (zero-copy)    0.1 ms
    mmap -> GPU (what a step needs)    4.3 ms
    current -> GPU (today)            53.3 ms      => 12.5x

Why a page-cache mapping rather than an in-process cache:

* It is **shared per node**, not per worker. Five arms read the same three
  encoders; today each worker pays for its own copy of identical bytes.
* It **cannot OOM**. Page cache is reclaimable -- under pressure the kernel
  evicts and the next read faults back in. There is no budget to size wrong,
  which is precisely the failure that made the eager loader dangerous.
* There is **no eviction policy to get wrong**. Each epoch walks every slide in
  shuffled order, so any application-level cache smaller than the split thrashes
  to a ~0% hit rate. The kernel sidesteps the question.

Opt-in and fail-safe by construction: with ``AUTOBENCH_BAG_CACHE`` unset the
behaviour is byte-for-byte today's, and any cache failure falls back to reading
the H5 rather than failing the run.
"""

from __future__ import annotations

import hashlib
import os
import tempfile

import h5py
import numpy as np
import torch

#: Directory for derived bags. Unset disables the cache entirely.
#: ``$SLURM_TMPDIR`` is the natural home on a cluster: node-local, fast, and
#: reclaimed by the scheduler when the job ends.
_ENV_VAR = "AUTOBENCH_BAG_CACHE"

#: Warn once per process, not once per bag.
_warned: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        print(f"[bag-cache] {message}")


def read_bag_from_h5(h5_path: str) -> np.ndarray:
    """The uncached read -- the exact bytes every arm has always trained on."""
    with h5py.File(h5_path, "r") as f:
        return np.asarray(f["features"][:], dtype=np.float32)


def _cache_key(h5_path: str) -> str:
    """Identity of the SOURCE file, not just its name.

    Size and mtime are in the key, so regenerating features invalidates the
    derived copy instead of silently serving stale bytes. Lazy reading already
    made feature immutability load-bearing for a run's correctness; this makes a
    change detectable rather than trusted.
    """
    stat = os.stat(h5_path)
    payload = f"{os.path.realpath(h5_path)}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _write_atomically(target: str, array: np.ndarray) -> None:
    """Publish via rename so a concurrent reader never sees a partial file.

    Workers race to build the same entry; ``os.replace`` makes the loser
    harmless rather than corrupting the winner, since both wrote identical
    bytes.
    """
    directory = os.path.dirname(target)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".bag-", suffix=".npy.tmp")
    try:
        # Write through the OPEN handle. Passing np.save a *path* that does not
        # end in .npy makes it append the suffix, so it would write beside the
        # temp file and the rename would publish an empty one.
        with os.fdopen(fd, "wb") as fh:
            np.save(fh, array, allow_pickle=False)
        os.replace(tmp, target)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_bag(h5_path: str) -> torch.Tensor:
    """Return one ``[N, embed_dim]`` float32 bag, memory-mapped when cached.

    Identical bytes either way -- only where they are read from changes.
    """
    cache_dir = os.environ.get(_ENV_VAR)
    if not cache_dir:
        return torch.from_numpy(read_bag_from_h5(h5_path))

    try:
        cached = os.path.join(cache_dir, _cache_key(h5_path) + ".npy")
    except OSError as exc:  # source vanished mid-run; let the H5 read report it
        _warn_once("stat", f"cannot stat sources ({exc}); cache disabled")
        return torch.from_numpy(read_bag_from_h5(h5_path))

    if os.path.exists(cached):
        try:
            # mmap_mode="c": copy-on-write. Reads share the page cache exactly
            # like "r", but the array is writable, so torch.from_numpy stays
            # zero-copy without emitting its non-writable-array warning on
            # every single bag.
            return torch.from_numpy(np.load(cached, mmap_mode="c"))
        except (OSError, ValueError, EOFError) as exc:
            # Truncated or unreadable entry: rebuild rather than fail the run.
            _warn_once("reread", f"unreadable cache entry ({exc}); rebuilding")

    array = read_bag_from_h5(h5_path)
    try:
        _write_atomically(cached, array)
    except OSError as exc:
        # Full disk, read-only mount, vanished directory -- degrade to the
        # uncached path. A slow run beats a failed one.
        _warn_once("write", f"cannot populate cache ({exc}); using H5 reads")
    return torch.from_numpy(array)
