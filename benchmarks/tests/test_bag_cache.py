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


# Process-global state (_disabled, _warned, _checked_dir) and the env var are
# reset for every test by the autouse fixture in conftest.py, so that reset
# applies to the whole suite rather than only this file.


def _enable(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    monkeypatch.setenv("AUTOBENCH_BAG_CACHE", str(cache))
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


class TestMappingSemantics:
    """A mapped tensor must be indistinguishable from an in-memory one."""

    def test_tensor_outlives_the_numpy_handle(self, bag, tmp_path, monkeypatch):
        """read_bag drops its local reference the moment it returns.

        If torch did not keep the mapping alive through the array it wraps,
        every cached read would be a use-after-free waiting for a GC pass.
        """
        import gc

        path, expected = bag
        _enable(monkeypatch, tmp_path)
        bag_cache.read_bag(path)

        got = bag_cache.read_bag(path)
        gc.collect()
        assert torch.equal(got, expected)

    def test_a_rogue_write_cannot_corrupt_the_shared_cache(
        self, bag, tmp_path, monkeypatch,
    ):
        """This cache is shared by every worker on the node.

        mmap_mode="c" is copy-on-write, so an in-place write by one reader --
        none exists today, but nothing structurally forbids one appearing --
        stays private to that process instead of poisoning the file and every
        other worker's view of it.
        """
        path, expected = bag
        _enable(monkeypatch, tmp_path)
        bag_cache.read_bag(path)

        first = bag_cache.read_bag(path)
        first[0, 0] = 12345.0

        second = bag_cache.read_bag(path)
        assert torch.equal(second, expected), (
            "an in-place write leaked into the shared cache; every other "
            "worker on the node would now read corrupted features"
        )


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

    def test_a_failed_write_latches_the_cache_off(self, bag, tmp_path, monkeypatch):
        """Degraded must mean baseline-speed, not slower than baseline.

        Without a latch, a full disk costs an H5 read AND a doomed
        multi-megabyte write for every bag of every epoch -- strictly worse
        than never enabling the cache, and announced by a single line of output
        for the whole run.
        """
        path, expected = bag
        _enable(monkeypatch, tmp_path)

        attempts = []

        def boom(*_a, **_k):
            attempts.append(1)
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(bag_cache, "_write_atomically", boom)
        for _ in range(5):
            assert torch.equal(bag_cache.read_bag(path), expected)

        assert len(attempts) == 1, (
            f"kept retrying a doomed write {len(attempts)} times; the cache "
            "must latch off after the first failure"
        )
        assert bag_cache._disabled

    def test_unsafe_filesystem_is_refused(self, bag, tmp_path, monkeypatch):
        """tmpfs pages are cgroup-charged and unreclaimable.

        Caching there reproduces the exact OOM kill this design exists to
        avoid, so it must refuse rather than "work". Reachable in practice:
        with no scheduler to supply $SLURM_TMPDIR the operator picks the
        directory by hand, and /dev/shm is the obvious fast-looking guess.
        """
        path, expected = bag
        _enable(monkeypatch, tmp_path)
        monkeypatch.setattr(bag_cache, "filesystem_type", lambda _p: "tmpfs")

        assert torch.equal(bag_cache.read_bag(path), expected)
        assert bag_cache._disabled, "a tmpfs cache dir must latch the cache off"
        assert not list(tmp_path.glob("cache/*.npy")), "wrote to tmpfs anyway"

    def test_unknown_filesystem_is_allowed(self, bag, tmp_path, monkeypatch):
        """Off Linux the type is undeterminable; that must not disable caching."""
        path, _ = bag
        cache = _enable(monkeypatch, tmp_path)
        monkeypatch.setattr(bag_cache, "filesystem_type", lambda _p: None)

        bag_cache.read_bag(path)
        assert not bag_cache._disabled
        assert list(cache.glob("*.npy"))

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


def test_arms_call_the_cache(bag, monkeypatch):
    """ABMIL and DTFD must route through it, or the fix reaches nothing.

    Asserts the CALL, not the source text. A substring check on
    ``inspect.getsource`` is vacuous here: the function's own ``def _read_bag(``
    line contains "read_bag(", so it holds for any function of that name --
    including one that reads the H5 directly and never imports this module.

    Both are the arms that read chunked H5. CLAM reads contiguous .pt and nnMIL
    reads through its vendored loader, which is why neither is covered here.
    """
    from autobench.pipeline.abmil import dataset as abmil_ds
    from autobench.pipeline.dtfd import dataset as dtfd_ds

    path, _ = bag
    calls: list[str] = []
    monkeypatch.setattr(
        bag_cache, "read_bag", lambda p: calls.append(p) or torch.zeros(1, 1),
    )

    abmil_ds._read_bag(path)
    dtfd_ds._read_bag(path)
    assert calls == [path, path], (
        "an arm's _read_bag did not go through bag_cache.read_bag"
    )


def test_changing_the_derivation_moves_every_key(bag, monkeypatch):
    """The key must fingerprint the TRANSFORM, not only the input.

    read_bag_from_h5 pins a dataset key and a dtype. If those change and the
    key does not, every warm cache serves the old derivation while the code
    claims the new one -- the same failure the per-fold results cache has
    already shipped once, where it fingerprints configuration and never the
    code version.

    Asserts the behaviour rather than the presence of the token in the source:
    ``_cache_key``'s own docstring names ``_DERIVATION`` twice, so a text check
    passes even after the payload drops it.
    """
    path, _ = bag
    before = bag_cache._cache_key(path)
    monkeypatch.setattr(bag_cache, "_DERIVATION", "features|float16|v99")
    assert bag_cache._cache_key(path) != before, (
        "changing _DERIVATION left the key unchanged, so a warm cache would "
        "keep serving bags built by the old derivation"
    )


def test_derivation_names_what_the_read_actually_does():
    """A version token that does not track the transform is decoration."""
    for token in ("features", "float32"):
        assert token in bag_cache._DERIVATION, (
            f"_DERIVATION must name what read_bag_from_h5 actually does; "
            f"{token!r} is missing"
        )


class TestTheMappingIsTheMechanism:
    """The win comes from mapping, not from remembering."""

    def test_every_cached_read_goes_back_to_the_mapping(
        self, bag, tmp_path, monkeypatch,
    ):
        """A process-level dict of arrays would pass every other test here.

        It would also put the whole split back in each worker's RAM -- the
        ~28 GB/worker that lazy loading removed. Spying on the resource rather
        than on a name is what distinguishes the two.
        """
        path, _ = bag
        _enable(monkeypatch, tmp_path)
        bag_cache.read_bag(path)  # populate

        modes: list[str | None] = []
        real = bag_cache.np.load
        monkeypatch.setattr(
            bag_cache.np, "load",
            lambda p, **k: (modes.append(k.get("mmap_mode")), real(p, **k))[1],
        )
        for _ in range(3):
            bag_cache.read_bag(path)

        assert modes == ["c", "c", "c"], (
            f"reads were served from process memory ({modes}); the split is "
            "resident per worker again"
        )

    def test_the_entry_is_invisible_until_it_is_complete(self, tmp_path, monkeypatch):
        """Publish must be atomic as a CONCURRENT READER sees it.

        Cleaning up after an exception is not the same property: writing
        straight to the final path passes that check while exposing a
        partially-written entry from the first byte, and a SIGKILL mid-write --
        this codebase's documented failure mode -- would publish it permanently.
        """
        cache = tmp_path / "cache"
        target = os.path.join(str(cache), "x.npy")
        seen: list[bool] = []
        real = bag_cache.np.save

        def observe(fh, arr, **kw):
            real(fh, arr, **kw)
            fh.flush()
            seen.append(os.path.exists(target))

        monkeypatch.setattr(bag_cache.np, "save", observe)
        bag_cache._write_atomically(target, np.zeros((2, 2), dtype="float32"))

        assert seen == [False], "entry was visible at its final path mid-write"
        assert os.path.exists(target)


@pytest.mark.parametrize("impostor", [None, "SLURM_TMPDIR", "TMPDIR", "XDG_CACHE_HOME"])
def test_only_the_opt_in_variable_turns_it_on(bag, tmp_path, monkeypatch, impostor):
    """Opt-in means one variable, not any plausible-looking one.

    ``$SLURM_TMPDIR`` is documented as the natural home, which makes falling
    back to it a tempting one-line change -- and one that would silently cache
    on every scheduler node for runs that never asked, voiding the
    byte-for-byte promise exactly where it matters.
    """
    path, _ = bag
    if impostor:
        monkeypatch.setenv(impostor, str(tmp_path / "elsewhere"))
    monkeypatch.chdir(tmp_path)

    opens: list[str] = []
    real = bag_cache.h5py.File
    monkeypatch.setattr(
        bag_cache.h5py, "File",
        lambda p, *a, **k: (opens.append(str(p)), real(p, *a, **k))[1],
    )
    bag_cache.read_bag(path)
    bag_cache.read_bag(path)

    assert len(opens) == 2, "something cached without the opt-in variable"
    assert list(tmp_path.rglob("*.npy")) == [], "wrote a cache entry anyway"


class TestFilesystemParser:
    """The real longest-match parser, not a stand-in for it.

    Every other filesystem test fakes ``filesystem_type`` wholesale, which
    leaves this logic unexecuted -- and a typo in it fails OPEN, straight back
    to caching on tmpfs.
    """

    @staticmethod
    def _mounts(tmp_path, body: str) -> str:
        path = tmp_path / "mounts"
        path.write_text(body)
        return str(path)

    def test_longest_mount_point_wins(self, tmp_path):
        mounts = self._mounts(tmp_path, "/dev/a / ext4 rw 0 0\ntmp /scratch tmpfs rw 0 0\n")
        assert bag_cache.filesystem_type("/scratch/x/y", mounts) == "tmpfs"
        assert bag_cache.filesystem_type("/home/x", mounts) == "ext4"

    def test_a_prefix_that_is_not_a_path_component_does_not_match(self, tmp_path):
        mounts = self._mounts(tmp_path, "/dev/a / ext4 rw 0 0\ntmp /scratch tmpfs rw 0 0\n")
        assert bag_cache.filesystem_type("/scratchpad/x", mounts) == "ext4"

    def test_escaped_space_in_mount_point(self, tmp_path):
        mounts = self._mounts(tmp_path, "/dev/a /my\\040disk ext4 rw 0 0\n")
        assert bag_cache.filesystem_type("/my disk/x", mounts) == "ext4"

    def test_malformed_line_is_skipped(self, tmp_path):
        mounts = self._mounts(tmp_path, "garbage\n/dev/a / ext4 rw 0 0\n")
        assert bag_cache.filesystem_type("/x", mounts) == "ext4"

    def test_missing_mounts_file_is_undeterminable(self, tmp_path):
        assert bag_cache.filesystem_type("/x", str(tmp_path / "nope")) is None

    def test_read_bag_refuses_a_tmpfs_dir_through_the_real_parser(
        self, bag, tmp_path, monkeypatch,
    ):
        """End to end, faking the mount TABLE rather than the function.

        This is the only test in which the real parser decides the outcome, so
        it is the only one that would notice the parser breaking.
        """
        path, expected = bag
        cache = _enable(monkeypatch, tmp_path)
        mounts = self._mounts(
            tmp_path, f"/dev/a / ext4 rw 0 0\ntmp {cache} tmpfs rw 0 0\n",
        )
        real = bag_cache.filesystem_type
        monkeypatch.setattr(
            bag_cache, "filesystem_type", lambda p, _f=None: real(p, mounts),
        )

        assert torch.equal(bag_cache.read_bag(path), expected)
        assert bag_cache._disabled, "real parser saw tmpfs and did not refuse"
        assert not list(cache.glob("*.npy"))


def test_the_filesystem_is_checked_once_not_per_bag(bag, tmp_path, monkeypatch):
    """Re-parsing the mount table per read is a real cost at 4 ms/step.

    On a host with many mounts (containers, autofs, NFS) this measured ~0.33 ms
    per bag, which is ~12 s per fold spent answering the same question.
    """
    path, _ = bag
    _enable(monkeypatch, tmp_path)

    checks: list[str] = []
    monkeypatch.setattr(
        bag_cache, "filesystem_type",
        lambda p, mounts_file="/proc/self/mounts": (checks.append(p), "ext4")[1],
    )
    for _ in range(5):
        bag_cache.read_bag(path)

    assert len(checks) == 1, (
        f"parsed the mount table {len(checks)} times for 5 reads; it is the "
        "same answer every time and belongs behind the latch"
    )


def test_a_stat_failure_degrades_instead_of_killing_the_worker(
    bag, tmp_path, monkeypatch,
):
    """With the cache ON, a transient stat error must not be fatal.

    A retrying ``h5py.File`` open survives ESTALE/EACCES that ``os.stat`` does
    not, so without this branch enabling the cache would make a run strictly
    more fragile than leaving it off -- the one asymmetry it must never have.
    """
    path, expected = bag
    _enable(monkeypatch, tmp_path)

    def boom(*_a, **_k):
        raise OSError(116, "Stale file handle")

    monkeypatch.setattr(bag_cache.os, "stat", boom)
    assert torch.equal(bag_cache.read_bag(path), expected)


def test_degradation_is_announced(bag, tmp_path, monkeypatch, capsys):
    """A silently degraded run looks exactly like a healthy one.

    The latch keeps a broken cache from being slower than no cache; this keeps
    it from being invisible, which is how a cluster-wide misconfiguration
    survives for a whole campaign.
    """
    path, _ = bag
    _enable(monkeypatch, tmp_path)

    def boom(*_a, **_k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(bag_cache, "_write_atomically", boom)
    bag_cache.read_bag(path)

    assert "bag-cache" in capsys.readouterr().out, (
        "degraded to uncached reads without telling anyone"
    )


def test_no_consumer_mutates_a_returned_bag():
    """The 'cannot OOM' guarantee depends on this, so pin it.

    Under copy-on-write a write faults the touched pages into ANONYMOUS,
    cgroup-charged, non-reclaimable memory -- turning the shared reclaimable
    page cache back into exactly the kind of allocation that OOM-killed this
    campaign before. Nothing structurally forbids an in-place op appearing, so
    assert its absence.
    """
    import re

    from autobench import BENCHMARKS_ROOT

    inplace = re.compile(
        r"\b(features|feats|bag)\s*(\[[^\]]*\]\s*=[^=]|\.(add_|mul_|sub_|div_|"
        r"copy_|clamp_|zero_|fill_|normal_|masked_fill_|scatter_)\()"
    )
    offenders = []
    for rel in (
        "abmil/train.py", "abmil/survival_train.py",
        "dtfd/train.py", "dtfd/eval.py", "dtfd/survival_train.py",
    ):
        path = BENCHMARKS_ROOT / "src" / "autobench" / "pipeline" / rel
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if inplace.search(line) and "device" not in line:
                offenders.append(f"{rel}:{n}: {line.strip()}")

    assert not offenders, (
        "in-place write to a bag; under copy-on-write this converts shared "
        "reclaimable page cache into unreclaimable cgroup-charged memory:\n  "
        + "\n  ".join(offenders)
    )
