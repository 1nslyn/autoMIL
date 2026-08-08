"""Shared campaign launch-binding contract tests."""
from __future__ import annotations

import hashlib
import json

import pytest

from automil.launch_binding import LaunchBindingError, validate_launch_binding


CAMPAIGN_ID = "campaign-v1"
CELL_ID = "cell-v1"
PROTOCOL_SHA256 = "a" * 64


def _sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _open_binding() -> dict:
    binding = {
        "campaign_id": CAMPAIGN_ID,
        "cell_id": CELL_ID,
        "agent_protocol_sha256": PROTOCOL_SHA256,
        "session_id": "session-v1",
        "started_at": "2026-08-04T00:00:00+00:00",
        "bound_at": "2026-08-04T00:00:01+00:00",
    }
    return {
        "schema_version": 3,
        "campaign_id": CAMPAIGN_ID,
        "cell_id": CELL_ID,
        "agent_protocol_sha256": PROTOCOL_SHA256,
        "status": "open",
        "session": {
            "session_id": "session-v1",
            "started_at": "2026-08-04T00:00:00+00:00",
            "bound_at": "2026-08-04T00:00:01+00:00",
            "ended_at": None,
            "termination_reason": None,
            "usage": None,
            "activity": None,
        },
        "binding_sha256": _sha256(binding),
        "attestation_sha256": None,
    }


def _validate(raw: object) -> dict:
    return validate_launch_binding(
        raw,
        campaign_id=CAMPAIGN_ID,
        cell_id=CELL_ID,
        agent_protocol_sha256=PROTOCOL_SHA256,
        require_open=True,
    )


def test_open_launch_binding_returns_only_immutable_launch_identity():
    validated = _validate(_open_binding())

    assert validated == {
        "session_id": "session-v1",
        "agent_protocol_sha256": PROTOCOL_SHA256,
        "binding_sha256": _open_binding()["binding_sha256"],
        "started_at": "2026-08-04T00:00:00+00:00",
        "bound_at": "2026-08-04T00:00:01+00:00",
    }


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda raw: raw.update({"unexpected": True}), "field set"),
        (
            lambda raw: raw["session"].update(
                {"bound_at": "2026-08-04T00:00:02+00:00"}
            ),
            "binding mismatch",
        ),
        (
            lambda raw: raw["session"].update(
                {"started_at": "2026-08-04T00:00:02+00:00"}
            ),
            "cannot start after",
        ),
        (lambda raw: raw.update({"status": "finalized"}), "not an open"),
    ],
)
def test_launch_binding_rejects_schema_time_hash_and_state_drift(mutate, message):
    raw = _open_binding()
    mutate(raw)

    with pytest.raises(LaunchBindingError, match=message):
        _validate(raw)
