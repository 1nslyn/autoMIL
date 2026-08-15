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


def test_null_composite_fold_contributes_no_metrics_either(tmp_path: Path) -> None:
    """`composite` and `metrics` must describe the SAME fold set (review finding).

    Keeping a dropped fold's surviving metric left `composite` averaged over N
    folds and `metrics` over N+1. CR-1b recomputes the composite from `metrics`,
    so the two disagreed past COMPOSITE_TOLERANCE — which fires terminal_writer's
    VAL-FIREWALL ERROR ("may have been computed from test") on a benign coverage
    mismatch, and then lets the mixed-denominator recompute WIN and become the
    node's authoritative selection signal.
    """
    from automil.scoring import composite_disagrees, recompute_composite

    for i in range(3):
        _write_fold(tmp_path, i, composite=0.75,
                    metrics={"val_auc": 0.80, "val_bacc": 0.70})
    # A single-class val fold: AUC unestimable, balanced accuracy perfectly fine.
    (tmp_path / "fold_3_result.json").write_text(json.dumps({
        "fold_index": 3, "fold_count": 5, "status": "completed",
        "metrics": {"val_auc": None, "val_bacc": 0.61},
        "held_out": {"test_auc": None, "test_bacc": 0.59},
        "composite": None, "elapsed_seconds": 40, "peak_vram_mb": 100,
    }))

    result = aggregate_folds(tmp_path, expected_fold_count=5)

    assert result["partial_folds"] == 3
    assert result["metrics"] == {"val_auc": pytest.approx(0.80),
                                 "val_bacc": pytest.approx(0.70)}
    assert result["held_out"] == {}  # the 3 healthy folds carried no held_out
    recomputed = recompute_composite(result["metrics"])
    assert not composite_disagrees(result["composite"], recomputed)
    # the dropped fold still ran, so its resource usage is accounted
    assert result["elapsed_seconds"] == 340


def test_a_key_missing_from_any_fold_is_dropped_not_averaged_unevenly(tmp_path: Path) -> None:
    """`composite` and every reported mean must share ONE denominator.

    Dropping a fold whole when its COMPOSITE is null was not enough: a fold can
    carry a finite composite and still be missing an individual metric, since
    _write_fold_result_json nulls each independently and the training-script
    contract makes `metrics` keys optional while `composite` is required.

    On the val side that re-opened the CR-1b divergence (false VAL-FIREWALL
    ERROR, and the mixed-denominator recompute becoming authoritative). On the
    test side it is worse: nothing recomputes `held_out`, so a test_auc averaged
    over 2 folds sat beside a test_bacc averaged over 3 under `status:
    completed` — and that block is what gets sealed into certify.json.
    """
    from automil.scoring import composite_disagrees, recompute_composite

    for i in range(2):
        (tmp_path / f"fold_{i}_result.json").write_text(json.dumps({
            "fold_index": i, "fold_count": 3, "status": "completed",
            "metrics": {"val_auc": 0.80, "val_bacc": 0.70},
            "held_out": {"test_auc": 0.70, "test_bacc": 0.60},
            "composite": 0.75, "elapsed_seconds": 100, "peak_vram_mb": 4000,
        }))
    # Finite composite, but this fold's test split missed a class.
    (tmp_path / "fold_2_result.json").write_text(json.dumps({
        "fold_index": 2, "fold_count": 3, "status": "completed",
        "metrics": {"val_auc": 0.80, "val_bacc": 0.70},
        "held_out": {"test_auc": None, "test_bacc": 0.60},
        "composite": 0.75, "elapsed_seconds": 100, "peak_vram_mb": 4000,
    }))

    result = aggregate_folds(tmp_path, expected_fold_count=3)

    # The fold with the null test_auc contributes nothing at all, so composite
    # and every reported mean describe the same 2 folds.
    assert result["status"] == "partial"
    assert result["partial_folds"] == 2
    assert result["held_out"] == {"test_auc": pytest.approx(0.70),
                                  "test_bacc": pytest.approx(0.60)}
    assert not composite_disagrees(
        result["composite"], recompute_composite(result["metrics"]),
    )


def test_val_side_uneven_coverage_cannot_trip_the_firewall_alarm(tmp_path: Path) -> None:
    """The CR-1b invariant, on the shape that produced a false test-leak alarm."""
    from automil.scoring import composite_disagrees, recompute_composite

    specs = [(0.75, 0.80, 0.70), (0.65, 0.70, 0.60), (0.70, 0.80, None)]
    for i, (comp, auc, bacc) in enumerate(specs):
        (tmp_path / f"fold_{i}_result.json").write_text(json.dumps({
            "fold_index": i, "fold_count": 3, "status": "completed",
            "metrics": {"val_auc": auc, "val_bacc": bacc},
            "composite": comp, "elapsed_seconds": 10, "peak_vram_mb": 100,
        }))

    result = aggregate_folds(tmp_path, expected_fold_count=3)

    assert result["partial_folds"] == 2     # the null-bacc fold drops whole
    recomputed = recompute_composite(result["metrics"])
    assert recomputed is None or not composite_disagrees(result["composite"], recomputed)


def test_validation_folds_carry_composites_and_prediction_hashes(tmp_path: Path) -> None:
    """The recovery path must emit the same per-fold evidence contract as a
    normal completion: fold composites feed the paired keep-margin and
    ``val_predictions_sha256`` feeds no-op detection; without them a recovered
    node silently reverts to the marginal-SE basis."""
    _write_fold(tmp_path, 0, composite=0.80)
    _write_fold(tmp_path, 1, composite=0.60)
    # Stamp a hash onto fold 0 the way _write_fold_result_json does (top level).
    f0 = tmp_path / "fold_0_result.json"
    data = json.loads(f0.read_text())
    data["val_predictions_sha256"] = "a" * 64
    f0.write_text(json.dumps(data))

    result = aggregate_folds(tmp_path, expected_fold_count=2)

    entries = result["validation_folds"]
    assert [e["fold_index"] for e in entries] == [0, 1]
    assert [e["composite"] for e in entries] == [0.80, 0.60]
    assert entries[0]["val_predictions_sha256"] == "a" * 64
    assert entries[1]["val_predictions_sha256"] is None   # pre-hash fold file
    # Val-only projection: the sealed test block never enters an entry.
    assert all("held_out" not in e for e in entries)
    assert all(not k.startswith("test_") for e in entries for k in e["metrics"])


def test_unresolvable_fold_index_drops_the_whole_fold(tmp_path: Path) -> None:
    """One denominator: a fold that cannot be placed in the fold vector must
    not count toward composite/composite_se/partial_folds either — a split
    denominator would let the ingest-side SE recompute (which prefers
    validation_folds) replace the N-fold SE and fire the tamper WARNING on a
    benign naming gap."""
    _write_fold(tmp_path, 0, composite=0.80)
    _write_fold(tmp_path, 1, composite=0.60)
    # A fold whose filename second token is non-numeric AND whose payload
    # carries a junk fold_index: unplaceable.
    rogue = json.loads((tmp_path / "fold_0_result.json").read_text())
    rogue["fold_index"] = "not-an-int"
    rogue["composite"] = 0.99
    (tmp_path / "fold_val_extra_result.json").write_text(json.dumps(rogue))

    result = aggregate_folds(tmp_path, expected_fold_count=2)

    assert result["partial_folds"] == 2
    assert result["composite"] == pytest.approx(0.70)
    assert [e["fold_index"] for e in result["validation_folds"]] == [0, 1]
