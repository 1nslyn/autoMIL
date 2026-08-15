"""The project-root walk must treat unreadable probe targets as absent.

A host with an ``automil`` service user makes ``/home/automil`` unreadable to
other users, and every ``_find_automil_dir`` walk from a cwd under ``/home``
probed ``/home/automil/config.yaml`` and crashed with PermissionError —
breaking every CLI command run outside a project on that host.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

needs_perms = pytest.mark.skipif(
    os.geteuid() == 0, reason="root bypasses file permissions"
)


def test_probe_exists_true_false(tmp_path: Path) -> None:
    from automil.paths import probe_exists

    present = tmp_path / "config.yaml"
    present.write_text("x")
    assert probe_exists(present)
    assert not probe_exists(tmp_path / "missing.yaml")


@needs_perms
def test_probe_exists_unreadable_reads_as_absent(tmp_path: Path) -> None:
    from automil.paths import probe_exists

    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "config.yaml").write_text("x")
    locked.chmod(0o000)
    try:
        assert probe_exists(locked / "config.yaml") is False
    finally:
        locked.chmod(0o755)


@needs_perms
def test_find_automil_dir_walks_past_an_unreadable_automil_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression shape: an ancestor carries an unreadable ``automil``
    entry (the /home/automil case); the walk must pass it and find the real
    project above, not crash."""
    from automil.cli._helpers import _find_automil_dir

    # tmp_path/automil -> the real project root's automil dir.
    real = tmp_path / "automil"
    real.mkdir()
    (real / "config.yaml").write_text("scoring: {}\n")
    # tmp_path/mid/automil -> unreadable decoy between cwd and the project.
    mid = tmp_path / "mid"
    decoy = mid / "automil"
    decoy.mkdir(parents=True)
    decoy.chmod(0o000)
    cwd = mid / "work"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    try:
        assert _find_automil_dir() == real
    finally:
        decoy.chmod(0o755)
