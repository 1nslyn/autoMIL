"""Unit tests for automil.runtime_helpers (CAP-03 / D-121, D-122).

Tests:
    test_get_fold_count_default                              -- env var unset → returns 5
    test_get_fold_count_from_env                             -- env var set → returns int
    test_register_sigterm_flush_idempotent                   -- second call is no-op
    test_register_sigterm_flush_subprocess_writes_result_json_on_sigterm
                                                             -- SIGTERM → result.json + returncode 0
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest


def test_get_fold_count_default(monkeypatch):
    """AUTOMIL_FOLD_COUNT unset → get_fold_count() returns 5 (int)."""
    monkeypatch.delenv("AUTOMIL_FOLD_COUNT", raising=False)
    from automil.runtime_helpers import get_fold_count
    assert get_fold_count() == 5


def test_get_fold_count_from_env(monkeypatch):
    """AUTOMIL_FOLD_COUNT=3 → get_fold_count() returns 3 (int, not str)."""
    monkeypatch.setenv("AUTOMIL_FOLD_COUNT", "3")
    from automil.runtime_helpers import get_fold_count
    result = get_fold_count()
    assert result == 3 and isinstance(result, int)


def test_register_sigterm_flush_idempotent(monkeypatch):
    """Calling register_sigterm_flush() twice installs handler once; second call is no-op.

    Saves and restores the original SIGTERM handler to avoid polluting other tests.
    Resets _SIGTERM_REGISTERED via monkeypatch to ensure a clean slate.
    """
    from automil import runtime_helpers
    original = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(runtime_helpers, "_SIGTERM_REGISTERED", False, raising=False)
    try:
        runtime_helpers.register_sigterm_flush()
        first = signal.getsignal(signal.SIGTERM)
        runtime_helpers.register_sigterm_flush()  # second call must be no-op
        second = signal.getsignal(signal.SIGTERM)
        assert callable(first)
        assert first is second
    finally:
        signal.signal(signal.SIGTERM, original)


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM unsupported on Windows")
def test_register_sigterm_flush_subprocess_writes_result_json_on_sigterm(tmp_path):
    """SIGTERM → handler aggregates fold files, writes result.json, exits 0.

    Pre-populates fold_0_result.json and fold_1_result.json in tmp_path.
    Spawns a child process that registers the SIGTERM handler then sleeps.
    Sends SIGTERM; asserts child exits 0 and result.json has status=partial.
    """
    # Pre-populate two completed fold files
    (tmp_path / "fold_0_result.json").write_text(json.dumps({
        "fold_index": 0,
        "fold_count": 5,
        "status": "completed",
        "metrics": {"val_auc": 0.80},
        "primary_value": 0.80,
        "elapsed_seconds": 100,
        "peak_vram_mb": 4000,
    }))
    (tmp_path / "fold_1_result.json").write_text(json.dumps({
        "fold_index": 1,
        "fold_count": 5,
        "status": "completed",
        "metrics": {"val_auc": 0.82},
        "primary_value": 0.82,
        "elapsed_seconds": 110,
        "peak_vram_mb": 4100,
    }))

    script = textwrap.dedent("""
        import time
        from automil.runtime_helpers import register_sigterm_flush
        register_sigterm_flush()
        time.sleep(30)
    """)

    env = os.environ.copy()
    env["AUTOMIL_FOLD_COUNT"] = "5"

    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        cwd=str(tmp_path),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(1.0)  # give child time to install handler
    proc.send_signal(signal.SIGTERM)
    stdout, stderr = proc.communicate(timeout=10)

    assert proc.returncode == 0, (
        f"Expected returncode 0, got {proc.returncode}. "
        f"stderr={stderr.decode(errors='replace')}"
    )

    result_path = tmp_path / "result.json"
    assert result_path.exists(), (
        f"result.json not written by SIGTERM handler. "
        f"stderr={stderr.decode(errors='replace')}"
    )

    data = json.loads(result_path.read_text())
    assert data["status"] == "partial"
    assert data["partial_folds"] == 2
    assert data["expected_folds"] == 5
