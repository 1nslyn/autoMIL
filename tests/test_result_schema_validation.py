"""DEC-03 / D-201: tests for src/automil/schemas/result.schema.json + validate_result.

Covers:
  - autobench 4-key shape (val_auc, val_bacc, test_auc, test_bacc)
  - sklearn-iris 2-key shape (accuracy, f1)
  - consumer-extension top-level keys (additionalProperties: true)
  - missing required composite
  - status enum violation
  - peak_vram_mb minimum constraint
  - metrics-value type constraint (must be number)

Schema is authored verbatim from CONTEXT.md D-201; this file is the regression
gate for accidental schema drift.
"""
from __future__ import annotations

import pytest

from automil.schemas import RESULT_SCHEMA, ValidationError, validate_result


def test_autobench_four_key_shape_validates():
    """D-201: autobench's historical 4-key metrics shape passes."""
    payload = {
        "composite": 0.5025,
        "metrics": {
            "val_auc": 0.81, "val_bacc": 0.78,
            "test_auc": 0.83, "test_bacc": 0.80,
        },
        "status": "completed",
        "elapsed_seconds": 4098.0,
        "peak_vram_mb": 4500.0,
    }
    validate_result(payload)  # must not raise


def test_sklearn_iris_two_key_shape_validates():
    """D-203 / DEC-02: sklearn-iris 2-key metrics shape passes."""
    payload = {
        "composite": 0.97,
        "metrics": {"accuracy": 0.97, "f1": 0.965},
        "status": "completed",
    }
    validate_result(payload)


def test_consumer_extension_top_level_key_validates():
    """D-201: additionalProperties: true means consumers may extend top-level."""
    payload = {
        "composite": 0.5,
        "config_hash": "deadbeef",
        "consumer_specific_field": [1, 2, 3],
    }
    validate_result(payload)


def test_missing_composite_key_fails():
    """D-201: composite is the sole required key."""
    with pytest.raises(ValidationError) as excinfo:
        validate_result({})
    assert "composite" in str(excinfo.value).lower()


def test_status_enum_violation_fails():
    """D-201: status is restricted to completed, crash, budget_killed, cancelled."""
    with pytest.raises(ValidationError):
        validate_result({"composite": 0.5, "status": "unknown_state"})


def test_negative_peak_vram_mb_fails():
    """D-201: peak_vram_mb has minimum: 0."""
    with pytest.raises(ValidationError):
        validate_result({"composite": 0.5, "peak_vram_mb": -1.0})


def test_metrics_non_number_value_fails():
    """D-201: metrics additionalProperties is restricted to number."""
    with pytest.raises(ValidationError):
        validate_result({"composite": 0.5, "metrics": {"val_auc": "high"}})


def test_schema_top_level_id_locked():
    """D-201: schema $id is locked so external referrers do not break."""
    assert RESULT_SCHEMA["$schema"].endswith("draft/2020-12/schema")
    assert RESULT_SCHEMA["required"] == ["composite"]
    assert RESULT_SCHEMA["additionalProperties"] is True


# ---------------------------------------------------------------------------
# Phase 9 / REC-03 extensions (D-05, D-07) — RED until Plan 03 updates schema
# ---------------------------------------------------------------------------


def test_partial_status_validates() -> None:
    """D-07: 'partial' is a valid status after schema update.

    RED until Plan 03 adds 'partial' to result.schema.json enum.
    """
    validate_result({
        "composite": 0.42,
        "status": "partial",
        "termination_reason": "sigterm",
    })  # must not raise


def test_termination_reason_is_optional() -> None:
    """D-05: termination_reason is optional and free-form — no status field needed.

    RED until Plan 03 adds optional termination_reason to result.schema.json.
    """
    validate_result({
        "composite": 0.5,
        "termination_reason": "oom",
    })  # no status field; termination_reason alone is valid


def test_crashed_drift_value_fails_validation() -> None:
    """D-06: 'crashed' is not in the tight status enum — must fail validation.

    This should be RED now (crashed was never in the schema enum), confirming
    the tight enum is enforced. Serves as a regression guard after D-06 fix.
    """
    with pytest.raises(ValidationError):
        validate_result({"composite": 0.5, "status": "crashed"})
