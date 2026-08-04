"""result.json schema validation (D-201 / DEC-03).

Wraps jsonschema.Draft202012Validator with a module-level pre-compiled validator
instance. Caller surfaces `automil/schemas/result.schema.json` in error messages
so the consumer can self-correct.

The schema file is the single source of truth; this module only loads + binds.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as _ValidationError

# Re-export for callers; do NOT import jsonschema directly elsewhere in
# src/automil/. Centralising the import keeps the dependency surface auditable.
ValidationError = _ValidationError

_SCHEMA_PATH: Path = Path(__file__).parent / "result.schema.json"
RESULT_SCHEMA: dict = json.loads(_SCHEMA_PATH.read_text())
_VALIDATOR: Draft202012Validator = Draft202012Validator(RESULT_SCHEMA)

# CR-1a (audit 2026-07-23): JSON Schema's ``{"type": "number"}`` accepts the
# non-finite floats ``inf`` / ``-inf`` / ``nan`` (Python parses the ``Infinity`` /
# ``NaN`` JSON tokens into them). Because the orchestrator trusts ``composite``
# verbatim as the val-firewall selection signal, an ``Infinity`` composite would
# permanently capture ``best_node`` and force ``keep`` (an agent-writable exploit),
# and a ``NaN`` poisons every downstream ``>`` comparison and persists as an
# invalid-JSON token that breaks non-Python readers (viz SSE, jq, serde). The
# schema cannot express "finite", so we enforce it here, at the same ingestion
# boundary the schema is validated on.
_FINITE_NUMBER_FIELDS = ("composite", "elapsed_seconds", "peak_vram_mb")
_FINITE_METRIC_BLOCKS = ("metrics", "held_out")


def _require_finite(payload: dict) -> None:
    """Reject non-finite numbers in the selection signal and metric blocks."""
    for field in _FINITE_NUMBER_FIELDS:
        value = payload.get(field)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValidationError(
                f"'{field}' must be a finite number, got {value!r} "
                f"(non-finite composite/metric values are rejected by the "
                f"val-firewall; see automil/schemas/result.schema.json)"
            )
    for block in _FINITE_METRIC_BLOCKS:
        metrics = payload.get(block)
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValidationError(
                        f"'{block}.{key}' must be a finite number, got {value!r}"
                    )


def validate_result(payload: dict) -> None:
    """Validate result.json payload against the D-201 schema.

    Args:
        payload: the dict parsed from a worktree's result.json (or assembled
            in-memory by the orchestrator's status-synthesis fallback path).

    Raises:
        ValidationError: payload violates the schema, or carries a non-finite
            ``composite`` / metric value (CR-1a). ``exc.message`` carries a
            single human-readable cause; ``exc.json_path`` carries the JSON
            Pointer to the offending node. Caller surfaces both plus the literal
            pointer ``automil/schemas/result.schema.json``.
    """
    _VALIDATOR.validate(payload)
    _require_finite(payload)
