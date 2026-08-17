"""REC-01 / D-02: SIGTERM flush writes to AUTOMIL_RESULTS_DIR, not cwd.

The bug: register_sigterm_flush() aggregates fold results and writes
result.json to Path.cwd() regardless of AUTOMIL_RESULTS_DIR. After D-02 fix,
it must write to Path(AUTOMIL_RESULTS_DIR) when that env var is set.

Tests call the handler function directly (capturing it from signal.signal
before _SIGTERM_REGISTERED guard triggers) to avoid killing the test process.
"""
from __future__ import annotations

import importlib
import json
import os
import signal
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import automil.runtime_helpers as _rh


def _fresh_handler(fold_count_env: str = "AUTOMIL_FOLD_COUNT"):
    """Return the SIGTERM handler function by temporarily resetting the registered guard
    and capturing via a signal.signal intercept.

    Resets _SIGTERM_REGISTERED to False so the handler can be re-captured per test.
    """
    _rh._SIGTERM_REGISTERED = False

    captured = {}

    original_signal = signal.signal

    def _intercept(signum, handler):
        if signum == signal.SIGTERM:
            captured["handler"] = handler
        original_signal(signum, handler)

    with patch("automil.runtime_helpers.signal") as mock_sig:
        import signal as _signal_mod
        mock_sig.SIGTERM = _signal_mod.SIGTERM

        def _intercept2(signum, handler):
            if signum == _signal_mod.SIGTERM:
                captured["handler"] = handler

        mock_sig.signal = _intercept2
        _rh._SIGTERM_REGISTERED = False
        _rh.register_sigterm_flush(fold_count_env=fold_count_env)

    return captured.get("handler")


def test_sigterm_flush_writes_to_results_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """D-02: SIGTERM handler writes result.json to AUTOMIL_RESULTS_DIR, not cwd.

    RED until D-02 fix ships in Plan 04.
    """
    results_dir = tmp_path / "archive" / "node_0001"
    results_dir.mkdir(parents=True)

    # Write a fold result in results_dir so aggregate_folds returns something
    fold_payload = {
        "fold_index": 0,
        "fold_count": 1,
        "status": "completed",
        "primary_value": 0.82,
        "metrics": {"val_auc": 0.82},
        "elapsed_seconds": 100,
        "peak_vram_mb": 4000,
    }
    (results_dir / "fold_0_result.json").write_text(json.dumps(fold_payload))

    # Also write fold result in cwd (where the bug writes today)
    cwd_dir = tmp_path / "workdir"
    cwd_dir.mkdir()
    (cwd_dir / "fold_0_result.json").write_text(json.dumps(fold_payload))

    monkeypatch.setenv("AUTOMIL_RESULTS_DIR", str(results_dir))
    monkeypatch.setenv("AUTOMIL_FOLD_COUNT", "1")
    monkeypatch.chdir(cwd_dir)

    handler = _fresh_handler()
    assert handler is not None, "SIGTERM handler was not captured — check _fresh_handler"

    # Invoke handler with SystemExit suppressed
    with pytest.raises(SystemExit):
        handler(signal.SIGTERM, None)

    # After D-02 fix: result.json in AUTOMIL_RESULTS_DIR, NOT cwd
    assert (results_dir / "result.json").exists(), (
        f"result.json not found in AUTOMIL_RESULTS_DIR={results_dir}. "
        "Bug D-02: handler writes to cwd instead of AUTOMIL_RESULTS_DIR."
    )
    assert not (cwd_dir / "result.json").exists(), (
        "result.json was written to cwd — D-02 regression: must go to AUTOMIL_RESULTS_DIR."
    )


def test_sigterm_flush_falls_back_to_cwd_when_env_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """baseline — must stay GREEN. When AUTOMIL_RESULTS_DIR is not set, flush writes to cwd.

    This documents the existing (pre-D-02) behavior. The D-02 fix must NOT break this fallback.
    """
    monkeypatch.delenv("AUTOMIL_RESULTS_DIR", raising=False)

    cwd_dir = tmp_path / "workdir"
    cwd_dir.mkdir()
    monkeypatch.chdir(cwd_dir)

    # Write fold result in cwd so aggregate_folds returns something
    fold_payload = {
        "fold_index": 0,
        "fold_count": 1,
        "status": "completed",
        "primary_value": 0.75,
        "metrics": {"val_auc": 0.75},
        "elapsed_seconds": 50,
        "peak_vram_mb": 3000,
    }
    (cwd_dir / "fold_0_result.json").write_text(json.dumps(fold_payload))
    monkeypatch.setenv("AUTOMIL_FOLD_COUNT", "1")

    handler = _fresh_handler()
    assert handler is not None, "SIGTERM handler was not captured"

    # Invoke handler with SystemExit suppressed
    with pytest.raises(SystemExit):
        handler(signal.SIGTERM, None)

    # Current behavior (GREEN): writes to cwd
    assert (cwd_dir / "result.json").exists(), (
        "Baseline broken: when AUTOMIL_RESULTS_DIR is unset, handler must write to cwd."
    )
