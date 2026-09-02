"""Coverage for the umask-honoring atomic-write primitives (D-2xx).

Group-shared preprint campaign context: five Unix users write into one
tree with umask 0o007 under setgid directories. Every mkstemp-based writer
was previously born 0600/0700 regardless of umask (mkstemp/mkdtemp both
ignore umask by design), so a file written by user A was unreadable to
user B. These tests assert the fixed permission behavior directly, not
the presence of any particular line of source.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from automil.runtime_helpers import atomic_write_text, group_mkdtemp


@pytest.fixture
def restore_umask():
    """Snapshot and restore the process umask around a test.

    os.umask() has no "peek" form -- reading it requires setting it, so
    tests that pin a umask must restore whatever was there before, or a
    process-wide umask leak would poison every later test in the run.
    """
    original = os.umask(0)
    os.umask(original)
    yield
    os.umask(original)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


class TestAtomicWriteTextUmask:
    def test_umask_007_yields_0660(self, tmp_path, restore_umask):
        os.umask(0o007)
        target = tmp_path / "out.json"
        atomic_write_text(target, "hello\n")
        assert _mode(target) == 0o660

    def test_umask_022_yields_0644(self, tmp_path, restore_umask):
        os.umask(0o022)
        target = tmp_path / "out.json"
        atomic_write_text(target, "hello\n")
        assert _mode(target) == 0o644

    def test_content_is_written_exactly(self, tmp_path, restore_umask):
        os.umask(0o007)
        target = tmp_path / "out.json"
        atomic_write_text(target, '{"a": 1}\n')
        assert target.read_text() == '{"a": 1}\n'

    def test_no_temp_file_left_behind(self, tmp_path, restore_umask):
        os.umask(0o007)
        target = tmp_path / "out.json"
        atomic_write_text(target, "content\n")
        leftovers = [p for p in tmp_path.iterdir() if p != target]
        assert leftovers == []

    def test_overwrite_replaces_existing_file_content(self, tmp_path, restore_umask):
        os.umask(0o007)
        target = tmp_path / "out.json"
        atomic_write_text(target, "first\n")
        atomic_write_text(target, "second\n")
        assert target.read_text() == "second\n"
        assert _mode(target) == 0o660

    def test_creates_parent_directory(self, tmp_path, restore_umask):
        os.umask(0o007)
        target = tmp_path / "nested" / "dir" / "out.json"
        atomic_write_text(target, "content\n")
        assert target.read_text() == "content\n"

    def test_failure_cleans_up_temp_file(self, tmp_path, restore_umask, monkeypatch):
        os.umask(0o007)
        target = tmp_path / "out.json"

        def _boom(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(os, "fsync", _boom)
        with pytest.raises(OSError):
            atomic_write_text(target, "content\n")
        assert not target.exists()
        assert list(tmp_path.iterdir()) == []


class TestGroupMkdtempUmask:
    def test_umask_007_yields_0770(self, tmp_path, restore_umask):
        os.umask(0o007)
        created = Path(group_mkdtemp(dir=str(tmp_path), prefix=".attempt-"))
        assert created.is_dir()
        # Mask off any special bits (setuid/setgid/sticky) before comparing
        # the base rwx permissions -- this directory's parent (tmp_path) has
        # no setgid bit in a plain pytest tmp dir, so none should appear
        # here either; the base permissions are what umask governs.
        assert _mode(created) & 0o777 == 0o770

    def test_umask_022_yields_0755(self, tmp_path, restore_umask):
        os.umask(0o022)
        created = Path(group_mkdtemp(dir=str(tmp_path), prefix=".attempt-"))
        assert _mode(created) & 0o777 == 0o755

    def test_preserves_inherited_setgid_bit(self, tmp_path, restore_umask):
        """A setgid parent directory must still be setgid-inherited by the child.

        group_mkdtemp widens the rwx bits to honor umask but must not
        clobber a setgid bit the OS already applied via inheritance from a
        setgid parent -- only the caller's own umask governs the base rwx
        bits, never the special bits.
        """
        os.umask(0o007)
        setgid_parent = tmp_path / "shared"
        setgid_parent.mkdir()
        os.chmod(setgid_parent, 0o2770)
        if not (stat.S_IMODE(setgid_parent.stat().st_mode) & stat.S_ISGID):
            pytest.skip("filesystem does not support setgid on directories here")

        # Probe: does a bare mkdir() under this setgid parent actually
        # propagate the bit to a freshly created child on this OS/filesystem?
        # Linux's VFS does this unconditionally at mkdir() time regardless of
        # the requested mode; some platforms (observed: macOS/APFS) do not,
        # which would make this test's premise false rather than the
        # implementation wrong.
        probe = setgid_parent / "probe"
        os.mkdir(probe, 0o700)
        propagates = bool(stat.S_IMODE(probe.stat().st_mode) & stat.S_ISGID)
        probe.rmdir()
        if not propagates:
            pytest.skip("this OS/filesystem does not propagate setgid to new subdirectories")

        created = Path(group_mkdtemp(dir=str(setgid_parent), prefix=".attempt-"))
        mode = stat.S_IMODE(created.stat().st_mode)
        assert mode & stat.S_ISGID, "inherited setgid bit must survive group_mkdtemp"
        assert mode & 0o777 == 0o770

    def test_returns_str_like_tempfile_mkdtemp(self, tmp_path, restore_umask):
        os.umask(0o007)
        result = group_mkdtemp(dir=str(tmp_path), prefix=".x-")
        assert isinstance(result, str)

    def test_prefix_is_applied(self, tmp_path, restore_umask):
        os.umask(0o007)
        created = Path(group_mkdtemp(dir=str(tmp_path), prefix=".marker-"))
        assert created.name.startswith(".marker-")
