"""Shared val-loss helper for protocol-v3 checkpoint selection.

Every classification arm selects its checkpoint on the continuous validation
cross-entropy of its own predictions (protocol v3); the shared-schema metrics
are still reported at that checkpoint. This module is measurement code and
belongs on ``registry.protected`` alongside ``evaluate.py``.
"""
from __future__ import annotations

import numpy as np


def ce_loss(y_true, y_probs) -> float:
    """Mean cross-entropy from predicted probabilities (non-finite -> +inf)."""
    y_true = np.asarray(y_true, dtype=int)
    raw = np.asarray(y_probs, dtype=float)[np.arange(len(y_true)), y_true]
    # Finiteness is checked BEFORE clipping: +inf would clip to 1.0 and win
    # selection with loss -0.0; -inf would clip to a finite large loss.
    if not np.all(np.isfinite(raw)):
        return float("inf")
    value = float(-np.mean(np.log(np.clip(raw, 1e-12, 1.0))))
    return value if np.isfinite(value) else float("inf")
