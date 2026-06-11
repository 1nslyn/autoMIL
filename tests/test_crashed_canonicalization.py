"""REC-03 / D-06: crashed → crash canonicalization in _crashed_payload.

D-06: The _crashed_payload helper in cells/reconcile.py emits status='crashed'
today, which is NOT in the tight enum. After D-06 fix it must emit status='crash'.

Named test_crashed_canonicalization.py (not test_aggregate_folds.py) to avoid
collision with the existing tests/cells/test_aggregate_folds.py which tests the
existing aggregate_folds API.
"""
from __future__ import annotations

import pytest

from automil.cells.reconcile import _crashed_payload


def test_crashed_payload_returns_crash_not_crashed() -> None:
    """D-06: _crashed_payload must emit status='crash', not status='crashed'.

    RED until D-06 fix ships in Plan 03.
    """
    result = _crashed_payload(expected_fold_count=5)

    assert result["status"] == "crash", (
        f"D-06 not implemented: _crashed_payload returned status={result['status']!r}. "
        "The canonical status enum is 'crash'; 'crashed' is the drift value to fix."
    )


def test_crashed_payload_has_no_crashed_key() -> None:
    """D-06: result['status'] must not contain the string 'crashed'.

    The value 'crashed' is the legacy drift value; it must be canonicalized to 'crash'.
    RED until D-06 fix ships in Plan 03.
    """
    result = _crashed_payload(expected_fold_count=5)

    assert result["status"] != "crashed", (
        "D-06 not implemented: _crashed_payload still returns 'crashed'. "
        "Must be canonicalized to 'crash' to match the result.schema.json enum."
    )
    # Extra guard: 'crashed' must not appear as a substring in the status value
    assert "crashed" not in str(result["status"]), (
        f"D-06: status value {result['status']!r} contains 'crashed' substring. "
        "Expected exactly 'crash'."
    )
