"""DEC-03 / D-201: daemon-side result.json schema validation tests.

Asserts the validate_result hook in _orchestrator_daemon.py:
  - valid payloads pass through and produce normal completion notifications.
  - malformed payloads override result with a crash dict pointing at the
    schema location.
  - the None case (training script never wrote result.json) is exempt from
    validation; the orchestrator synthesises a minimal payload from the log
    and returncode.

Uses direct validate_result calls and static-text analysis of the daemon
source to verify the hook is in place. Integration-level test (running the
full daemon ingest path with a mocked collect_result) is covered by the
plan 08-09 sklearn-iris sub-gate B end-to-end test.
"""
from __future__ import annotations

from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_validate_result_imports_cleanly():
    """The validate_result symbol must be importable from automil.schemas."""
    from automil.schemas import validate_result, ValidationError
    assert callable(validate_result)
    assert issubclass(ValidationError, Exception)


def test_valid_autobench_result_json_passes_validation():
    """Well-formed autobench result.json passes validate_result without raising."""
    from automil.schemas import validate_result
    payload = {
        "primary_value": 0.502,
        "metrics": {
            "val_auc": 0.81, "val_bacc": 0.78,
            "test_auc": 0.83, "test_bacc": 0.80,
        },
        "status": "completed",
        "elapsed_seconds": 4098.0,
        "peak_vram_mb": 4500.0,
    }
    validate_result(payload)  # must not raise


def test_valid_minimal_result_json_passes_validation():
    """A minimal result.json with only primary_value required field passes."""
    from automil.schemas import validate_result
    validate_result({"primary_value": 0.95})


def test_malformed_result_json_raises_with_nonempty_message():
    """Malformed result.json raises ValidationError with a non-empty message."""
    from automil.schemas import validate_result, ValidationError
    with pytest.raises(ValidationError) as excinfo:
        validate_result({"primary_value": "not a number"})
    assert excinfo.value.message  # non-empty error string from jsonschema


def test_daemon_error_template_contains_schema_pointer():
    """The daemon's override-on-failure error string must reference the schema file.

    Read the daemon source and assert the template substrings are present.
    This is a static-text regression gate for the operator-facing message.
    """
    daemon_src = (
        _REPO_ROOT / "src" / "automil" / "backends" / "_orchestrator_daemon.py"
    ).read_text()
    assert "see automil/schemas/result.schema.json" in daemon_src
    assert "result.json failed schema validation:" in daemon_src
    assert "json_path=" in daemon_src


def test_daemon_imports_validate_result_inline():
    """The daemon imports validate_result inline in the validation path.

    Static-text regression: ensures the import remains close to the use site
    so any future refactor that drops the import is detected immediately.
    """
    daemon_src = (
        _REPO_ROOT / "src" / "automil" / "backends" / "_orchestrator_daemon.py"
    ).read_text()
    assert "from automil.schemas import validate_result" in daemon_src
    assert "ValidationError" in daemon_src


def test_status_enum_violation_raises_validation_error():
    """status not in the locked enum raises ValidationError."""
    from automil.schemas import validate_result, ValidationError
    with pytest.raises(ValidationError):
        validate_result({"primary_value": 0.5, "status": "weird_state"})


def test_negative_peak_vram_mb_raises_validation_error():
    """peak_vram_mb minimum: 0 enforced by schema."""
    from automil.schemas import validate_result, ValidationError
    with pytest.raises(ValidationError):
        validate_result({"primary_value": 0.5, "peak_vram_mb": -42.0})


def test_none_result_skips_validation_by_construction():
    """The daemon's None-result branch skips validation by construction.

    The code path 'if result is not None: validate_result(result)' means
    that when collect_result returns None, validation never runs and the
    orchestrator synthesises a minimal compliant payload. This test
    confirms the guard condition is present in the daemon source.
    """
    daemon_src = (
        _REPO_ROOT / "src" / "automil" / "backends" / "_orchestrator_daemon.py"
    ).read_text()
    # The guard 'if result is not None:' must wrap the validate_result call.
    assert "if result is not None:" in daemon_src
    # And it must appear before the 'if result is None:' fall-through block.
    guard_pos = daemon_src.index("if result is not None:")
    fallthrough_pos = daemon_src.index("if result is None:")
    assert guard_pos < fallthrough_pos, (
        "validate_result guard (result is not None) must appear before "
        "the fall-through synthesis block (result is None)"
    )
