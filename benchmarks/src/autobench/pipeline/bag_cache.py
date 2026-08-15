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
* It **cannot OOM -- on local disk**. Page cache is reclaimable, so under
  pressure the kernel evicts and the next read faults back in. There is no
  budget to size wrong, which is precisely the failure that made the eager
  loader dangerous. That guarantee is conditional, not absolute: on tmpfs the
  pages are cgroup-charged and unreclaimable, and on a network filesystem a
  replaced inode can fault as SIGBUS. Both are refused outright rather than
  documented as somebody else's problem -- see ``_UNSAFE_FSTYPES``.
* There is **no eviction policy to get wrong**. Each epoch walks every slide in
  shuffled order, so any application-level cache smaller than the split thrashes
  to a ~0% hit rate. The kernel sidesteps the question.

Opt-in and fail-safe by construction: with ``AUTOBENCH_BAG_CACHE`` unset the
behaviour is byte-for-byte today's, and any cache failure falls back to reading
the H5 rather than failing the run. A failure that will recur -- a full disk, an
unsafe filesystem -- latches the cache off for the process, because retrying it
per bag would leave the run *slower* than never enabling the cache at all.

The cache never evicts and is not size-capped. That is deliberate for a
node-local, scheduler-reclaimed directory holding at most the working set of one
job; it is another reason a shared or persistent location is the wrong home.
"""

from __future__ import annotations

import hashlib
import os
import tempfile

import h5py
import numpy as np
import torch

#: Directory for derived bags. Unset disables the cache entirely.
#: ``$SLURM_TMPDIR`` is the natural home on a cluster: node-local disk, fast,
#: and reclaimed by the scheduler when the job ends.
_ENV_VAR = "AUTOBENCH_BAG_CACHE"

#: Identity of the DERIVATION, not just the source bytes. ``read_bag_from_h5``
#: pins a dataset key and a dtype; change either and the source file's identity
#: does not move, so every warm cache would keep serving the OLD derivation
#: while the code claims the new one. That is precisely how the per-fold results
#: cache burned this project once already -- it fingerprints configuration and
#: never the code version, so a code-motivated re-run silently returned stale
#: numbers. BUMP THIS whenever read_bag_from_h5 changes what it produces.
_DERIVATION = "features|float32|v1"

#: Filesystems where this cache is actively harmful, not merely slow.
#:
#: tmpfs/ramfs: pages are NOT reclaimable page cache -- they are charged to the
#:   job's cgroup and can only be swapped, so a cache there reproduces the exact
#:   ``oom-kill: constraint=CONSTRAINT_MEMCG`` this design exists to avoid.
#: network filesystems: a mapping whose inode is replaced on another client can
#:   fault with ESTALE, which the kernel delivers as SIGBUS -- uncatchable, no
#:   traceback, dead worker. mmap over them is also slow enough to erase the win.
#:
#: This matters because the operator picks the directory by hand when there is
#: no scheduler to supply $SLURM_TMPDIR, and /dev/shm is the obvious "fast" guess.
_UNSAFE_FSTYPES = frozenset({
    "tmpfs", "ramfs", "nfs", "nfs4", "cifs", "smb3", "smbfs", "lustre",
    "fuse.sshfs", "gpfs", "beegfs", "ceph", "glusterfs",
})

#: Warn once per process, not once per bag.
_warned: set[str] = set()

#: Latched off after a failure that will recur for every remaining bag.
_disabled = False

#: Cache dir whose filesystem has already been vetted, so the mount table is
#: parsed once per process rather than once per bag.
_checked_dir: str | None = None


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        print(f"[bag-cache] {message}")


def _disable(message: str) -> None:
    """Stop trying, permanently, for this process.

    Without this a broken cache is SLOWER than no cache: every bag of every
    epoch would read the H5, write ~76 MB, fail to publish, and repeat -- while
    printing a single line for the whole run. A degraded run must be
    baseline-speed, and it must say so.
    """
    global _disabled
    _disabled = True
    _warn_once("disabled", f"{message}; falling back to direct H5 reads")


def filesystem_type(path: str, mounts_file: str = "/proc/self/mounts") -> str | None:
    """Filesystem backing ``path``, or None when it cannot be determined.

    Takes the longest matching mount point, so ``/scratch`` wins over ``/``.
    Returns None off Linux, where the caller treats the filesystem as acceptable
    rather than refusing to run.

    ``mounts_file`` is injectable so the parser itself is testable. Faking this
    function wholesale would leave the real longest-match logic unexecuted by
    any test, and a typo in it fails open -- straight back to caching on tmpfs.
    """
    try:
        with open(mounts_file) as fh:
            rows = [line.split()[:3] for line in fh]
    except OSError:
        return None

    target = os.path.realpath(path)
    best_type, best_len = None, -1
    for row in rows:
        if len(row) < 3:
            continue
        point, fstype = row[1].replace("\\040", " "), row[2]
        if (target == point or target.startswith(point.rstrip("/") + "/")) \
                and len(point) > best_len:
            best_type, best_len = fstype, len(point)
    return best_type


def _cache_dir_is_usable(cache_dir: str) -> bool:
    """Check the filesystem ONCE per process, not once per bag.

    This re-opens and re-parses the mount table; on a host with many mounts
    (containers, autofs, NFS) that is a measurable per-read cost against a
    ~4 ms/step target, and it is the same answer every time.
    """
    global _checked_dir
    if _checked_dir == cache_dir:
        return True
    fstype = filesystem_type(cache_dir)
    if fstype in _UNSAFE_FSTYPES:
        _disable(
            f"{_ENV_VAR}={cache_dir!r} is on {fstype!r}, where this cache is "
            "harmful rather than helpful (see _UNSAFE_FSTYPES)"
        )
        return False
    _checked_dir = cache_dir
    return True


def read_bag_from_h5(h5_path: str) -> np.ndarray:
    """The uncached read -- the exact bytes every arm has always trained on.

    Changing the dataset key or the dtype here changes the derivation; bump
    ``_DERIVATION`` in the same commit or warm caches will serve the old one.
    """
    with h5py.File(h5_path, "r") as f:
        return np.asarray(f["features"][:], dtype=np.float32)


def _cache_key(h5_path: str) -> str:
    """Identity of the SOURCE file and of the DERIVATION applied to it.

    Size, inode and mtime are in the key, so regenerating features invalidates
    the derived copy instead of silently serving stale bytes. Lazy reading
    already made feature immutability load-bearing for a run's correctness; this
    makes a change detectable rather than trusted. The inode also catches
    update-by-rename, the standard atomic-update idiom, which can preserve both
    size and mtime.

    ``_DERIVATION`` is in the payload for the reason described at its
    definition: a key that fingerprints only the input is the same mistake the
    per-fold results cache made.
    """
    stat = os.stat(h5_path)
    payload = (
        f"{os.path.realpath(h5_path)}|{stat.st_dev}|{stat.st_ino}"
        f"|{stat.st_size}|{stat.st_mtime_ns}|{_DERIVATION}"
    )
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
    if not cache_dir or _disabled:
        return torch.from_numpy(read_bag_from_h5(h5_path))

    if not _cache_dir_is_usable(cache_dir):
        return torch.from_numpy(read_bag_from_h5(h5_path))

    try:
        cached = os.path.join(cache_dir, _cache_key(h5_path) + ".npy")
    except OSError as exc:  # source vanished mid-run; let the H5 read report it
        _warn_once("stat", f"cannot stat {h5_path} ({exc})")
        return torch.from_numpy(read_bag_from_h5(h5_path))

    if os.path.exists(cached):
        try:
            # mmap_mode="c": copy-on-write. Reads share the page cache exactly
            # like "r", but the array is writable, so torch.from_numpy stays
            # zero-copy without emitting its non-writable-array warning on
            # every single bag. Under "r" the pages are PROT_READ while the
            # tensor claims to be writable, so an in-place op would SIGSEGV.
            return torch.from_numpy(np.load(cached, mmap_mode="c"))
        except Exception as exc:
            # Truncated, unreadable, or not an .npy at all: rebuild rather than
            # fail the run. Deliberately broad -- a foreign file at this path
            # can raise from zipfile or torch, and no read failure justifies
            # taking down a training worker.
            _warn_once("reread", f"unreadable cache entry ({exc}); rebuilding")

    array = read_bag_from_h5(h5_path)
    try:
        _write_atomically(cached, array)
    except OSError as exc:
        # Full disk, read-only mount, vanished directory: conditions that will
        # recur for every remaining bag. Latch off rather than paying an H5 read
        # AND a doomed multi-megabyte write for each one, which would leave the
        # run slower than never having enabled the cache.
        _disable(f"cannot populate {cache_dir!r} ({exc})")
    return torch.from_numpy(array)
