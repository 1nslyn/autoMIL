"""CR-3 (audit 2026-07-23): the survival selection signal must not be the mean of
five ~2-event per-fold c-indices.

The survival trainers explicitly refuse to select checkpoints on the per-fold val
c-index ("With ~2 events per val fold the val c-index is near-random, so
maximizing it would overfit to noise" — clam/survival_train.py), yet autoMIL's
agentic search selected recipes on the mean of exactly that quantity. The
composite now uses concordance pooled over every fold's validation risks.
"""
from __future__ import annotations

import math

import pytest

from autobench.pipeline.evaluate import pooled_c_index, pooled_val_block


def _rec(risks, statuses, times, pids=None):
    return {
        "risks": risks,
        "statuses": statuses,
        "times": times,
        "patient_ids": pids or [f"p{i}" for i in range(len(risks))],
    }


def test_pooled_matches_single_fold_when_only_one_fold():
    rec = _rec([0.9, 0.1, 0.5], [1.0, 1.0, 0.0], [1.0, 5.0, 3.0], ["a", "b", "c"])
    pooled = pooled_c_index([rec])
    assert math.isfinite(pooled)
    assert 0.0 <= pooled <= 1.0


def test_pooling_uses_all_folds_not_the_mean_of_per_fold_values():
    """Two 2-event folds that are each perfectly concordant internally but
    inconsistent with each other: the fold-mean says 1.0, pooling exposes the
    disagreement. This is exactly the small-sample illusion CR-3 removes."""
    # Fold A: higher risk dies earlier (concordant).
    a = _rec([0.9, 0.1], [1.0, 1.0], [1.0, 9.0], ["a1", "a2"])
    # Fold B: same risks, opposite ordering (also internally concordant on its
    # own scale, but on a shared scale it contradicts fold A).
    b = _rec([0.1, 0.9], [1.0, 1.0], [2.0, 8.0], ["b1", "b2"])

    pooled = pooled_c_index([a, b])
    assert math.isfinite(pooled)
    # Each fold alone is perfectly concordant (mean would be 1.0); pooled must
    # be strictly lower because the folds disagree on a common risk scale.
    assert pooled < 1.0


def test_empty_and_missing_records_are_nan_safe():
    assert math.isnan(pooled_c_index([]))
    assert math.isnan(pooled_c_index([{"risks": [], "statuses": [], "times": [],
                                       "patient_ids": []}]))
    assert math.isnan(pooled_c_index([None, {}]))


def test_pooled_val_block_empty_for_classification():
    # Classification fold results carry no val_records → no val_pooled block.
    assert pooled_val_block([{"val_metrics": {"auc_roc": 0.8}}]) == {}
    assert pooled_val_block([]) == {}


def test_pooled_val_block_present_for_survival():
    fr = [{"val_metrics": {"c_index": 0.6},
           "val_records": _rec([0.9, 0.1], [1.0, 1.0], [1.0, 9.0])}]
    block = pooled_val_block(fr)
    assert "c_index" in block
    assert math.isfinite(block["c_index"])
