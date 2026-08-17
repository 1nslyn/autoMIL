"""Unit tests for _write_fold_result_json in autobench CLAM runner.

Covers:
  - env-gating (AUTOMIL_RESULTS_DIR absent → no-op)
  - flat-float metric mapping to D-118 keys (Pitfall 5, flat shape)
  - CI-dict metric mapping to D-118 keys (Pitfall 5, dict shape)
  - one file per fold, fold_index in JSON
  - fold_count sourced from AUTOMIL_FOLD_COUNT env
  - missing metrics → zero fallback (no exception)
"""

from __future__ import annotations

import json

import pytest

from autobench.pipeline.clam.runner import _write_fold_result_json


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _minimal_result(
    test_auc: float = 0.85,
    test_bacc: float = 0.82,
    val_auc: float = 0.90,
    val_bacc: float = 0.84,
    elapsed: int = 100,
    vram: int = 4500,
) -> dict:
    return {
        "test_metrics": {"auc_roc": test_auc, "balanced_accuracy": test_bacc},
        "val_metrics": {"auc_roc": val_auc, "balanced_accuracy": val_bacc},
        "elapsed_seconds": elapsed,
        "peak_vram_mb": vram,
        "fold": 0,
    }


# ---------------------------------------------------------------------------
# Test 1: writes fold file when AUTOMIL_RESULTS_DIR is set
# ---------------------------------------------------------------------------

def test_writes_fold_file_when_results_dir_set(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOMIL_RESULTS_DIR", str(tmp_path))

    result = _minimal_result()
    _write_fold_result_json(2, result)

    fold_file = tmp_path / "fold_2_result.json"
    assert fold_file.exists(), "fold_2_result.json should be written to AUTOMIL_RESULTS_DIR"

    payload = json.loads(fold_file.read_text())
    assert payload["fold_index"] == 2
    assert payload["status"] == "completed"
    # val-firewall: metrics is val-only (agent-facing); test lives in sealed held_out
    assert payload["metrics"]["val_auc"] == pytest.approx(0.90)
    assert payload["metrics"]["val_bacc"] == pytest.approx(0.84)
    assert "test_auc" not in payload["metrics"]
    assert "test_bacc" not in payload["metrics"]
    assert payload["held_out"]["test_auc"] == pytest.approx(0.85)
    assert payload["held_out"]["test_bacc"] == pytest.approx(0.82)
    # composite is the VALIDATION selection signal — the primary metric alone
    # (scoring.formula: val_auc); val 0.90 != test 0.85 proves the val side won
    assert payload["composite"] == pytest.approx(0.90)
    assert payload["elapsed_seconds"] == 100
    assert payload["peak_vram_mb"] == 4500


# ---------------------------------------------------------------------------
# Test 2: no-op when AUTOMIL_RESULTS_DIR is unset
# ---------------------------------------------------------------------------

def test_noop_when_results_dir_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("AUTOMIL_RESULTS_DIR", raising=False)

    # Should not raise; no file should be written anywhere
    _write_fold_result_json(0, _minimal_result())

    assert list(tmp_path.iterdir()) == [], "No files should be written when AUTOMIL_RESULTS_DIR is unset"


# ---------------------------------------------------------------------------
# Test 3: CI-dict metric shape is unwrapped correctly (Pitfall 5)
# ---------------------------------------------------------------------------

def test_metric_keys_mapped_correctly_from_dict_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOMIL_RESULTS_DIR", str(tmp_path))

    result = {
        "test_metrics": {
            "auc_roc": {"mean": 0.91, "std": 0.02, "ci_low": 0.87, "ci_high": 0.95},
            "balanced_accuracy": {"mean": 0.88, "std": 0.01, "ci_low": 0.86, "ci_high": 0.90},
        },
        "val_metrics": {
            "auc_roc": {"mean": 0.89, "std": 0.03},
            "balanced_accuracy": {"mean": 0.85, "std": 0.02},
        },
        "elapsed_seconds": 200,
        "peak_vram_mb": 3000,
    }
    _write_fold_result_json(1, result)

    payload = json.loads((tmp_path / "fold_1_result.json").read_text())
    assert payload["held_out"]["test_auc"] == pytest.approx(0.91)
    assert payload["held_out"]["test_bacc"] == pytest.approx(0.88)
    assert payload["metrics"]["val_auc"] == pytest.approx(0.89)
    assert payload["metrics"]["val_bacc"] == pytest.approx(0.85)
    assert "test_auc" not in payload["metrics"]


# ---------------------------------------------------------------------------
# Test 4: one file per fold, each with correct fold_index
# ---------------------------------------------------------------------------

def test_writes_one_file_per_fold(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOMIL_RESULTS_DIR", str(tmp_path))

    for fold_idx in range(5):
        result = _minimal_result(test_auc=0.8 + fold_idx * 0.01)
        _write_fold_result_json(fold_idx, result)

    for fold_idx in range(5):
        fold_file = tmp_path / f"fold_{fold_idx}_result.json"
        assert fold_file.exists(), f"fold_{fold_idx}_result.json should exist"
        payload = json.loads(fold_file.read_text())
        assert payload["fold_index"] == fold_idx


# ---------------------------------------------------------------------------
# Test 5: fold_count read from AUTOMIL_FOLD_COUNT env
# ---------------------------------------------------------------------------

def test_uses_automil_fold_count_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOMIL_RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("AUTOMIL_FOLD_COUNT", "7")

    _write_fold_result_json(2, _minimal_result())

    payload = json.loads((tmp_path / "fold_2_result.json").read_text())
    assert payload["fold_count"] == 7


# ---------------------------------------------------------------------------
# Test 6: missing metrics keys → null, not zero, and no exception
# ---------------------------------------------------------------------------

def test_handles_missing_metrics_gracefully(tmp_path, monkeypatch):
    """A metric that does not exist is ``null``, NOT 0.0 (changed 2026-08-10).

    The old zero fallback is the exact anti-pattern the consuming aggregator
    documents against — ``aggregate_folds`` "must distinguish missing data from
    zero-valued data" (Pitfall 4). Writing 0.0 recorded a fold that produced no
    metrics at all as a genuine zero-AUC result, which then averaged into the
    partial composite and dragged it down as if the model had scored nothing.
    ``null`` is skipped by the aggregator instead, which is what "we have no
    value here" is supposed to mean.
    """
    monkeypatch.setenv("AUTOMIL_RESULTS_DIR", str(tmp_path))

    result = {"test_metrics": {}, "val_metrics": {}}
    _write_fold_result_json(0, result)  # should not raise

    payload = json.loads((tmp_path / "fold_0_result.json").read_text())
    assert payload["held_out"]["test_auc"] is None
    assert payload["held_out"]["test_bacc"] is None
    assert payload["metrics"]["val_auc"] is None
    assert payload["metrics"]["val_bacc"] is None
    assert payload["composite"] is None


# ---------------------------------------------------------------------------
# Test 7: ordinal tasks record qwk on both sides, clamped at 0
# ---------------------------------------------------------------------------

def test_ordinal_fold_records_clamped_qwk_on_both_sides(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOMIL_RESULTS_DIR", str(tmp_path))

    result = _minimal_result()
    result["val_metrics"]["qwk"] = -0.15   # below-chance kappa
    result["test_metrics"]["qwk"] = 0.42
    _write_fold_result_json(0, result, ordinal=True)

    payload = json.loads((tmp_path / "fold_0_result.json").read_text())
    # Recording clamp: kappa is [-1, 1], sealed consumers require [0, 1].
    assert payload["metrics"]["val_qwk"] == 0.0
    assert payload["held_out"]["test_qwk"] == pytest.approx(0.42)
    # qwk never votes: composite is still the selection metric alone.
    assert payload["composite"] == pytest.approx(0.90)


def test_ordinal_fold_missing_qwk_is_invalid(tmp_path, monkeypatch):
    """Declared-but-missing qwk nulls the recorded slot AND the composite —
    fold validity spans the full recorded evidence set."""
    monkeypatch.setenv("AUTOMIL_RESULTS_DIR", str(tmp_path))

    _write_fold_result_json(0, _minimal_result(), ordinal=True)

    payload = json.loads((tmp_path / "fold_0_result.json").read_text())
    assert payload["metrics"]["val_qwk"] is None
    assert payload["held_out"]["test_qwk"] is None
    assert payload["composite"] is None


def test_non_ordinal_fold_never_carries_qwk(tmp_path, monkeypatch):
    """qwk present in the raw metrics must NOT be sniffed into the evidence
    of a task that never declared ordinality."""
    monkeypatch.setenv("AUTOMIL_RESULTS_DIR", str(tmp_path))

    result = _minimal_result()
    result["val_metrics"]["qwk"] = 0.5
    result["test_metrics"]["qwk"] = 0.5
    _write_fold_result_json(0, result)

    payload = json.loads((tmp_path / "fold_0_result.json").read_text())
    assert "val_qwk" not in payload["metrics"]
    assert "test_qwk" not in payload["held_out"]
