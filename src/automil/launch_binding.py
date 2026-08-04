"""Versioned launch-session binding shared by framework and consumers."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Mapping


class LaunchBindingError(ValueError):
    """A launch binding is malformed, drifted, or not open for submission."""


def _content_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_launch_binding(
    raw: object,
    *,
    campaign_id: str,
    cell_id: str,
    agent_protocol_sha256: str,
    require_open: bool,
) -> dict[str, Any]:
    """Validate the immutable portion of one versioned agent-session record."""
    top_fields = {
        "schema_version", "campaign_id", "cell_id", "agent_protocol_sha256",
        "status", "session", "binding_sha256", "attestation_sha256",
    }
    session_fields = {
        "session_id", "started_at", "bound_at", "ended_at",
        "termination_reason", "usage",
    }
    if not isinstance(raw, Mapping) or set(raw) != top_fields:
        raise LaunchBindingError("agent session field set is not exact")
    session = raw.get("session")
    if not isinstance(session, Mapping) or set(session) != session_fields:
        raise LaunchBindingError("agent session record field set is not exact")
    session_id = session.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise LaunchBindingError("agent session_id is invalid")
    if (
        not isinstance(agent_protocol_sha256, str)
        or len(agent_protocol_sha256) != 64
        or any(char not in "0123456789abcdef" for char in agent_protocol_sha256)
    ):
        raise LaunchBindingError("agent protocol hash is invalid")
    try:
        started_at = datetime.fromisoformat(str(session.get("started_at")))
        bound_at = datetime.fromisoformat(str(session.get("bound_at")))
    except ValueError as exc:
        raise LaunchBindingError("agent session launch timestamps are invalid") from exc
    if started_at.tzinfo is None or bound_at.tzinfo is None:
        raise LaunchBindingError("agent session launch timestamps require timezones")
    if started_at > bound_at:
        raise LaunchBindingError("agent session cannot start after controller binding")
    binding_payload = {
        "campaign_id": campaign_id,
        "cell_id": cell_id,
        "agent_protocol_sha256": agent_protocol_sha256,
        "session_id": session_id,
        "started_at": session["started_at"],
        "bound_at": session["bound_at"],
    }
    if (
        raw.get("schema_version") != 2
        or raw.get("campaign_id") != campaign_id
        or raw.get("cell_id") != cell_id
        or raw.get("agent_protocol_sha256") != agent_protocol_sha256
        or raw.get("binding_sha256") != _content_sha256(binding_payload)
    ):
        raise LaunchBindingError("agent session binding mismatch")
    if require_open and (
        raw.get("status") != "open"
        or session.get("ended_at") is not None
        or session.get("termination_reason") is not None
        or session.get("usage") is not None
        or raw.get("attestation_sha256") is not None
    ):
        raise LaunchBindingError("agent session is not an open launch binding")
    return {
        "session_id": session_id,
        "agent_protocol_sha256": agent_protocol_sha256,
        "binding_sha256": str(raw["binding_sha256"]),
        "started_at": str(session["started_at"]),
        "bound_at": str(session["bound_at"]),
    }
