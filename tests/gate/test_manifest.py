"""Tests for gate/manifest.py — GateManifest persistence layer (GTE-01, GTE-02 / D-137, D-138).

Tests 1-12 cover Task 1 (dataclass + read/write/load/retire).
Tests 13-16 cover Task 2 (write_manifest_committed + git-commit + path.unlink rollback).

Note Test 11b (test_retire_manifest_rollback_on_git_failure) added during plan-checker iter-1.
"""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from automil.gate.manifest import (
    GateManifest,
    load_manifest,
    read_manifest,
    retire_manifest,
    validate_manifest_dict,
    write_manifest,
    write_manifest_committed,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_CELLS = (
    ("abc12345", "ccrcc", "uni_v2", "high_grade"),
    ("def67890", "clwd", "ctranspath", "subtype"),
)

_VALID_DICT = {
    "parent_id": "node_0001",
    "created_at": "2026-05-05T00:00:00+00:00",
    "git_committed_at_sha": "PENDING",
    "held_out_cells": list(_CELLS),
    "K": 2,
    "p_threshold": 0.05,
    "bootstrap_reps": 100,
    "win_definition": "delta_primary_value > 0 AND p < p_threshold",
    "schema_version": "gate-v1",
}


def _make_manifest(**overrides) -> GateManifest:
    defaults = dict(
        parent_id="node_0001",
        created_at="2026-05-05T00:00:00+00:00",
        git_committed_at_sha="PENDING",
        held_out_cells=_CELLS,
        K=2,
        p_threshold=0.05,
        bootstrap_reps=100,
        win_definition="delta_primary_value > 0 AND p < p_threshold",
        schema_version="gate-v1",
    )
    defaults.update(overrides)
    return GateManifest(**defaults)


@pytest.fixture
def git_repo(tmp_path) -> Path:
    """A minimal git repo with an initial commit — needed for write_manifest_committed."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    # Initial commit so there is a HEAD
    (tmp_path / "README.md").write_text("initial")
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True)
    return tmp_path


# ---------------------------------------------------------------------------
# Task 1 tests — dataclass + read/write/load/retire
# ---------------------------------------------------------------------------

# Test 1
def test_manifest_frozen():
    """GateManifest is a frozen dataclass — mutations raise FrozenInstanceError."""
    m = _make_manifest()
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.K = 99  # type: ignore[misc]


# Test 2
def test_manifest_round_trip(tmp_path):
    """write_manifest then read_manifest preserves all fields including nested tuples."""
    m = _make_manifest()
    manifests_dir = tmp_path / "gate"
    write_manifest(m, manifests_dir)
    path = manifests_dir / "node_0001.gate_manifest.json"
    assert path.exists()
    m2 = read_manifest(path)
    assert m2 == m
    assert isinstance(m2.held_out_cells, tuple)
    for cell in m2.held_out_cells:
        assert isinstance(cell, tuple)
        assert len(cell) == 4


# Test 3
def test_manifest_atomic_write_uses_tempfile():
    """Source code uses tempfile.mkstemp(dir=str( and os.replace(tmp_path, str(."""
    src = pathlib.Path("src/automil/gate/manifest.py").read_text()
    assert "tempfile.mkstemp(dir=str(" in src, "must use tempfile.mkstemp with dir= for atomicity"
    assert "os.replace(tmp_path, str(" in src, "must use os.replace for atomic rename"


# Test 4
def test_validate_K_must_be_positive():
    """K < 1 raises ValueError with 'K must be >= 1' in message."""
    d = dict(_VALID_DICT, K=0)
    with pytest.raises(ValueError, match="K must be >= 1"):
        validate_manifest_dict(d)


# Test 5
def test_validate_K_le_held_out_count():
    """K=5 with len(held_out_cells)=2 raises ValueError containing 'K'."""
    d = dict(_VALID_DICT, K=5)  # only 2 cells
    with pytest.raises(ValueError, match="K"):
        validate_manifest_dict(d)


# Test 6
def test_validate_p_threshold_range():
    """p_threshold <= 0 or > 1 raises; 0.05 passes."""
    with pytest.raises(ValueError, match="p_threshold"):
        validate_manifest_dict(dict(_VALID_DICT, p_threshold=0.0))
    with pytest.raises(ValueError, match="p_threshold"):
        validate_manifest_dict(dict(_VALID_DICT, p_threshold=1.5))
    # Should not raise
    validate_manifest_dict(dict(_VALID_DICT, p_threshold=0.05))
    validate_manifest_dict(dict(_VALID_DICT, p_threshold=1.0))  # boundary: 1.0 is valid


# Test 7
def test_validate_bootstrap_reps_minimum():
    """bootstrap_reps < 100 raises; 100 and 1000 pass."""
    with pytest.raises(ValueError, match="bootstrap_reps"):
        validate_manifest_dict(dict(_VALID_DICT, bootstrap_reps=50))
    # Should not raise
    validate_manifest_dict(dict(_VALID_DICT, bootstrap_reps=100))
    validate_manifest_dict(dict(_VALID_DICT, bootstrap_reps=1000))


# Test 8
def test_validate_held_out_cells_non_empty():
    """Empty held_out_cells raises ValueError."""
    with pytest.raises(ValueError, match="held_out_cells"):
        validate_manifest_dict(dict(_VALID_DICT, held_out_cells=[]))


# Test 9
def test_validate_schema_version_must_match():
    """schema_version != 'gate-v1' raises; 'gate-v1' passes."""
    with pytest.raises(ValueError, match="schema_version"):
        validate_manifest_dict(dict(_VALID_DICT, schema_version="gate-v0"))
    with pytest.raises(ValueError, match="schema_version"):
        d = dict(_VALID_DICT)
        del d["schema_version"]
        validate_manifest_dict(d)
    # Should not raise
    validate_manifest_dict(dict(_VALID_DICT, schema_version="gate-v1"))


# Test 10
def test_load_manifest_by_parent_id(tmp_path):
    """load_manifest('node_0001', dir) reads node_0001.gate_manifest.json."""
    m = _make_manifest()
    manifests_dir = tmp_path / "gate"
    write_manifest(m, manifests_dir)
    m2 = load_manifest("node_0001", manifests_dir)
    assert m2 == m
    # FileNotFoundError when absent
    with pytest.raises(FileNotFoundError):
        load_manifest("node_9999", manifests_dir)


# Test 11
def test_retire_manifest_renames_file(git_repo):
    """retire_manifest renames active -> retired; retired JSON has reason + timestamp."""
    manifests_dir = git_repo / "gate"
    m = _make_manifest()
    # Stage + commit the active manifest so git knows about it
    write_manifest(m, manifests_dir)
    active = manifests_dir / "node_0001.gate_manifest.json"
    subprocess.run(["git", "add", str(active)], cwd=git_repo, check=True)
    subprocess.run(["git", "commit", "-m", "add manifest"], cwd=git_repo, check=True)

    sha = retire_manifest("node_0001", "K too generous", manifests_dir, git_repo)

    # Active file should be gone
    assert not active.exists()
    # Retired file should exist
    retired = manifests_dir / "node_0001.retired.gate_manifest.json"
    assert retired.exists()
    d = json.loads(retired.read_text())
    assert d["retired_reason"] == "K too generous"
    assert "retired_at" in d
    # SHA is returned
    assert len(sha) == 40


# Test 11b
def test_retire_manifest_rollback_on_git_failure(tmp_path):
    """On git commit failure: RuntimeError raised with 'working tree restored';
    active file restored with original payload; retired file absent;
    NO git checkout in mock call history (Leo memory feedback_never_blind_checkout)."""
    manifests_dir = tmp_path / "gate"
    m = _make_manifest()
    write_manifest(m, manifests_dir)
    active = manifests_dir / "node_0001.gate_manifest.json"
    original_payload = active.read_text()

    called_cmds = []

    def mock_run(cmd, **kwargs):
        called_cmds.append(list(cmd))
        check = kwargs.get("check", False)
        # Raise on git commit (but not on git add or git rev-parse)
        if cmd[1] == "commit":
            import subprocess as sp
            raise sp.CalledProcessError(1, cmd, stderr="mock commit failure")
        # Return a mock result for other calls
        import subprocess as sp
        result = sp.CompletedProcess(cmd, 0, stdout="", stderr="")
        return result

    with patch("subprocess.run", side_effect=mock_run):
        with pytest.raises(RuntimeError, match="working tree restored"):
            retire_manifest("node_0001", "some reason", manifests_dir, tmp_path)

    # (a) active manifest should be restored with original payload
    assert active.exists(), "active manifest must be restored after git failure"
    assert active.read_text() == original_payload

    # (b) retired file must NOT exist
    retired = manifests_dir / "node_0001.retired.gate_manifest.json"
    assert not retired.exists(), "retired file must be cleaned up on rollback"

    # (c) No git checkout invocation (Leo memory feedback_never_blind_checkout)
    git_checkout_calls = [c for c in called_cmds if len(c) >= 2 and c[1] == "checkout"]
    assert git_checkout_calls == [], (
        f"git checkout must NEVER be called for rollback; found: {git_checkout_calls}"
    )


# Test 12
def test_atomic_write_cleans_up_tmp_on_exception(tmp_path):
    """If write fails mid-way, no *.tmp files remain in manifests_dir."""
    import os

    manifests_dir = tmp_path / "gate"
    manifests_dir.mkdir()
    m = _make_manifest()

    original_fdopen = os.fdopen

    def broken_fdopen(fd, *args, **kwargs):
        os.close(fd)  # release the fd to avoid leak
        raise OSError("injected write failure")

    with patch("os.fdopen", side_effect=broken_fdopen):
        with pytest.raises(OSError, match="injected write failure"):
            write_manifest(m, manifests_dir)

    tmp_files = list(manifests_dir.glob("*.tmp"))
    assert tmp_files == [], f"tmp files should be cleaned up; found: {tmp_files}"


# ---------------------------------------------------------------------------
# Task 2 tests — write_manifest_committed (git-commit + path.unlink rollback)
# ---------------------------------------------------------------------------

# Test 13
def test_write_manifest_committed_returns_sha(git_repo):
    """write_manifest_committed returns 40-char SHA; git log shows new commit."""
    manifests_dir = git_repo / "gate"
    m = _make_manifest()
    sha = write_manifest_committed(m, manifests_dir, git_repo)

    # 40-char hex SHA
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha)

    # git log shows commit with expected message components
    log = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=git_repo, capture_output=True, text=True, check=True,
    ).stdout
    assert "node_0001" in log
    assert "gate: register manifest" in log or "gate:" in log

    # Registration commit message (either the last or second-to-last commit) should
    # contain held-out count, K=N, and p< (the backfill commit is a second commit).
    all_msgs = subprocess.run(
        ["git", "log", "--format=%s"],
        cwd=git_repo, capture_output=True, text=True, check=True,
    ).stdout
    # At least one commit in history should have K=2 and p<0.05
    assert "K=2" in all_msgs, f"K=2 not found in any commit message: {all_msgs!r}"
    assert "p<0.05" in all_msgs, f"p<0.05 not found in any commit message: {all_msgs!r}"


# Test 14
def test_write_manifest_committed_refuses_overwrite(git_repo):
    """Second call with same parent_id raises FileExistsError with 'retire-manifest'."""
    manifests_dir = git_repo / "gate"
    m = _make_manifest()
    write_manifest_committed(m, manifests_dir, git_repo)
    with pytest.raises(FileExistsError, match="retire-manifest"):
        write_manifest_committed(m, manifests_dir, git_repo)


# Test 15
def test_write_manifest_committed_rollback_on_git_failure(tmp_path):
    """On git failure: RuntimeError; manifest file does NOT exist; no git checkout."""
    manifests_dir = tmp_path / "gate"
    m = _make_manifest()

    called_cmds = []

    def mock_run(cmd, **kwargs):
        called_cmds.append(list(cmd))
        check = kwargs.get("check", False)
        if cmd[1] == "commit":
            import subprocess as sp
            raise sp.CalledProcessError(1, cmd, stderr="mock commit failure")
        import subprocess as sp
        return sp.CompletedProcess(cmd, 0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=mock_run):
        with pytest.raises(RuntimeError):
            write_manifest_committed(m, manifests_dir, tmp_path)

    # Manifest file must not exist (rolled back via path.unlink())
    manifest_path = manifests_dir / "node_0001.gate_manifest.json"
    assert not manifest_path.exists(), (
        "manifest must be removed via path.unlink() on git failure"
    )

    # No git checkout in call history
    checkout_calls = [c for c in called_cmds if len(c) >= 2 and c[1] == "checkout"]
    assert checkout_calls == [], (
        f"git checkout must NEVER be used for rollback; found: {checkout_calls}"
    )


# Test 16
def test_write_manifest_committed_uses_path_unlink_not_checkout():
    """Source-level: manifest.py contains path.unlink() and NO 'git', 'checkout' literal."""
    src = pathlib.Path("src/automil/gate/manifest.py").read_text()
    # Leo memory feedback_never_blind_checkout — NEVER checkout for rollback
    assert '"git", "checkout"' not in src, (
        "manifest.py must not invoke `git checkout` for rollback"
    )
    assert "git checkout --" not in src, (
        "manifest.py must not use `git checkout --` for rollback"
    )
    # Rollback must use path.unlink()
    assert "path.unlink()" in src, "rollback must use path.unlink() per Leo memory"
