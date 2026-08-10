"""Tests for aggregate_folds() — per-fold result aggregation (CAP-03 / D-119).

TDD RED: These tests fail until src/automil/cells/reconcile.py is fully
implemented per the D-119 spec.  Import path is ``automil.cells.reconcile``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from automil.cells.reconcile import aggregate_folds


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _write_fold(
    node_archive: Path,
    idx: int,
    composite: float = 0.80,
    metrics: dict | None = None,
    elapsed: int = 100,
    peak_vram: int = 4000,
    fold_count: int = 5,
) -> None:
    """Write a well-formed fold_<idx>_result.json into node_archive."""
    payload = {
        "fold_index": idx,
        "fold_count": fold_count,
        "status": "completed",
        "metrics": metrics
        or {
            "val_auc": composite,
            "val_bacc": composite,
            "test_auc": composite,
            "test_bacc": composite,
        },
        "composite": composite,
        "elapsed_seconds": elapsed,
        "peak_vram_mb": peak_vram,
    }
    (node_archive / f"fold_{idx}_result.json").write_text(json.dumps(payload))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_all_folds_completed_returns_completed_status(tmp_path: Path) -> None:
    """K=5 folds present → status='completed', partial_folds==5, correct composite mean."""
    composites = [0.80, 0.82, 0.84, 0.86, 0.88]
    for i, c in enumerate(composites):
        _write_fold(tmp_path, i, composite=c)

    result = aggregate_folds(tmp_path, expected_fold_count=5)

    assert result["status"] == "completed"
    assert result["partial_folds"] == 5
    assert result["expected_folds"] == 5
    assert result["composite"] == pytest.approx(0.84, rel=1e-6)
    # mean of per-fold val_auc (same as composite in default fixture)
    assert result["metrics"]["val_auc"] == pytest.approx(0.84, rel=1e-6)


def test_partial_folds_returns_partial_status(tmp_path: Path) -> None:
    """3 of 5 folds → status='partial'; composite is mean of 3, NOT zero, NOT NaN."""
    composites = [0.80, 0.82, 0.84]
    for i, c in enumerate(composites):
        _write_fold(tmp_path, i, composite=c)

    result = aggregate_folds(tmp_path, expected_fold_count=5)

    assert result["status"] == "partial"
    assert result["partial_folds"] == 3
    assert result["expected_folds"] == 5
    # CRITICAL: composite must not be zero or NaN
    assert result["composite"] == pytest.approx(0.82, rel=1e-6)
    assert result["composite"] != 0.0
    assert result["composite"] == result["composite"]  # not NaN


def test_zero_folds_returns_crashed_status(tmp_path: Path) -> None:
    """Empty directory → status='crash', composite=0.0, partial_folds=0.

    D-06 (REC-03): canonical status is 'crash', not 'crashed' (drift value removed).
    """
    result = aggregate_folds(tmp_path, expected_fold_count=5)

    assert result["status"] == "crash"  # D-06: was 'crashed' pre-v1.1
    assert result["composite"] == 0.0
    assert result["partial_folds"] == 0
    assert result["expected_folds"] == 5


def test_single_fold_returns_partial(tmp_path: Path) -> None:
    """1 of 5 folds → status='partial', partial_folds=1, composite=0.75."""
    _write_fold(tmp_path, 0, composite=0.75)

    result = aggregate_folds(tmp_path, expected_fold_count=5)

    assert result["status"] == "partial"
    assert result["partial_folds"] == 1
    assert result["composite"] == pytest.approx(0.75, rel=1e-6)


def test_malformed_fold_skipped_with_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Malformed JSON fold file is skipped; a WARNING is logged; partial_folds reflects good files."""
    # Write 4 good folds + 1 malformed
    for i in range(5):
        _write_fold(tmp_path, i, composite=0.80)

    # Overwrite fold_2 with bad JSON
    (tmp_path / "fold_2_result.json").write_text("{bad json")

    with caplog.at_level(logging.WARNING, logger="automil.cells.reconcile"):
        result = aggregate_folds(tmp_path, expected_fold_count=5)

    assert result["partial_folds"] == 4
    assert result["status"] == "partial"
    # At least one warning logged about the malformed file
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        "malformed" in m.lower() or "skipping" in m.lower()
        for m in warning_messages
    ), f"Expected malformed/skipping warning, got: {warning_messages}"


def test_metrics_mean_across_folds(tmp_path: Path) -> None:
    """metrics dict is the per-key mean across available folds."""
    _write_fold(tmp_path, 0, composite=0.80, metrics={"val_auc": 0.80, "test_auc": 0.85})
    _write_fold(tmp_path, 1, composite=0.82, metrics={"val_auc": 0.82, "test_auc": 0.86})
    _write_fold(tmp_path, 2, composite=0.84, metrics={"val_auc": 0.84, "test_auc": 0.87})

    result = aggregate_folds(tmp_path, expected_fold_count=5)

    assert result["metrics"]["val_auc"] == pytest.approx(0.82, rel=1e-6)
    assert result["metrics"]["test_auc"] == pytest.approx(0.86, rel=1e-6)


def test_elapsed_seconds_summed_peak_vram_max(tmp_path: Path) -> None:
    """elapsed_seconds is sum across folds; peak_vram_mb is max."""
    _write_fold(tmp_path, 0, composite=0.80, elapsed=100, peak_vram=4000)
    _write_fold(tmp_path, 1, composite=0.80, elapsed=200, peak_vram=4500)
    _write_fold(tmp_path, 2, composite=0.80, elapsed=300, peak_vram=4200)

    result = aggregate_folds(tmp_path, expected_fold_count=5)

    assert result["elapsed_seconds"] == 600
    assert result["peak_vram_mb"] == 4500


def test_unexpected_extra_fold_files_handled(tmp_path: Path) -> None:
    """6 fold files when expected=5 → partial_folds=6, status='partial' (n != expected)."""
    for i in range(6):
        _write_fold(tmp_path, i, composite=0.80)

    result = aggregate_folds(tmp_path, expected_fold_count=5)

    assert result["partial_folds"] == 6
    assert result["status"] == "partial"


def test_metrics_dict_with_mixed_keys_across_folds(tmp_path: Path) -> None:
    """Folds with different metric keys; mean is per-key across folds that have the key."""
    _write_fold(tmp_path, 0, composite=0.80, metrics={"val_auc": 0.80, "extra": 1.0})
    _write_fold(tmp_path, 1, composite=0.82, metrics={"val_auc": 0.82})
    _write_fold(tmp_path, 2, composite=0.84, metrics={"val_auc": 0.84})

    result = aggregate_folds(tmp_path, expected_fold_count=5)

    # val_auc: mean of 3 values
    assert result["metrics"]["val_auc"] == pytest.approx(0.82, rel=1e-6)
    # extra: mean of 1 value (only fold_0 has it)
    assert result["metrics"]["extra"] == pytest.approx(1.0, rel=1e-6)


def test_null_composite_fold_is_skipped_not_counted_as_zero(tmp_path: Path) -> None:
    """A fold whose composite was unestimable serializes as ``null`` (never NaN).

    ``float(None)`` raises TypeError, and this aggregator runs inside the SIGTERM
    handler — an uncaught raise there loses the whole partial flush. Counting it
    as ``0.0`` would be just as wrong: the aggregator's stated contract is to
    distinguish missing data from zero-valued data (Pitfall 4).
    """
    _write_fold(tmp_path, 0, composite=0.80)
    _write_fold(tmp_path, 1, composite=0.84)
    (tmp_path / "fold_2_result.json").write_text(json.dumps({
        "fold_index": 2,
        "fold_count": 5,
        "status": "completed",
        "metrics": {"val_auc": None, "val_bacc": None},
        "composite": None,
        "elapsed_seconds": 90,
        "peak_vram_mb": 3000,
    }))

    result = aggregate_folds(tmp_path, expected_fold_count=5)

    assert result["status"] == "partial"
    assert result["partial_folds"] == 2          # not 3 — the null fold is not evidence
    assert result["composite"] == pytest.approx(0.82, rel=1e-6)
    assert result["metrics"]["val_auc"] == pytest.approx(0.82, rel=1e-6)
    # elapsed/vram are still accounted: the fold DID consume resources
    assert result["elapsed_seconds"] == 290


def test_non_finite_composite_fold_is_skipped(tmp_path: Path) -> None:
    """Defence in depth: a NaN that reached disk some other way must not average in."""
    _write_fold(tmp_path, 0, composite=0.80)
    (tmp_path / "fold_1_result.json").write_text(
        '{"fold_index": 1, "fold_count": 5, "status": "completed", '
        '"metrics": {}, "composite": NaN, "elapsed_seconds": 10, "peak_vram_mb": 1}'
    )

    result = aggregate_folds(tmp_path, expected_fold_count=5)

    assert result["partial_folds"] == 1
    assert result["composite"] == pytest.approx(0.80, rel=1e-6)
