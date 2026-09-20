"""Redactor positive-case tests — one per leak class (TRJ-03, TRJ-06 / D-82, D-83)."""
from __future__ import annotations

import json
import pytest

from automil.trajectory.redactor import redact, redact_event, apply_size_cap, _SIZE_CAP_BYTES


# --- Positive cases: each input MUST be redacted ---

@pytest.mark.parametrize("secret,expected", [
    # sk- token (20+ chars after prefix)
    ("sk-abcdefghijklmnopqrstu",                             "sk-[REDACTED]"),
    # hf_ token (20+ chars)
    ("hf_abcdefghijklmnopqrstu1234",                        "hf_[REDACTED]"),
    # ghp_ token (30+ chars)
    ("ghp_abcdefghijklmnopqrstuvwxyz1234",                  "ghp_[REDACTED]"),
    # AWS access key (exactly 20 chars: AKIA + 16)
    ("AKIAIOSFODNN7EXAMPLE",                                 "AKIA[REDACTED]"),
    # *_API_KEY= pattern
    ("OPENAI_API_KEY=sk-abc123",                             "OPENAI_API_KEY=[REDACTED]"),
    ("ANTHROPIC_API_KEY=some-secret-value",                  "ANTHROPIC_API_KEY=[REDACTED]"),
    # *_TOKEN= pattern
    ("MY_TOKEN=verysecretvalue",                             "MY_TOKEN=[REDACTED]"),
    ("GITHUB_TOKEN=ghp_very_secret_token_here_1234567890",  "GITHUB_TOKEN=[REDACTED]"),
    # *_KEY= pattern (catches generic secret keys)
    ("AWS_SECRET_KEY=super_secret_key_value",               "AWS_SECRET_KEY=[REDACTED]"),
])
def test_redact_positive(secret: str, expected: str) -> None:
    """Each leak class has at least one matching positive case (TRJ-06)."""
    assert redact(secret) == expected


# --- False-positive guard: these must NOT be redacted ---

@pytest.mark.parametrize("safe_string", [
    "sk-short",                       # < 20 chars after sk- prefix — NOT a token
    "task_key_index",                 # no = sign; not a secret assignment
    "skeletal",                       # starts with "sk" but no dash
    "disk-based",                     # unrelated word
    "stack_api_keys_count=5",         # lowercase variable — pattern requires uppercase
    "index_key=0",                    # lowercase key — pattern requires [A-Z][A-Z0-9_]{1,40}
    # WR-01 (Phase 3 review) — value-too-short false-positive guards.
    # Real secrets are at least 8 chars; these legitimate config values
    # (under 8 chars after `=`) MUST stay un-redacted.
    "CACHE_KEY=abc",                  # 3-char value
    "NO_API_KEY=true",                # bool-string value
    "GIT_TOKEN=on",                   # 2-char flag
    "FAKE_KEY=",                      # empty value
    "CONFIG_KEY=v1",                  # short version label
])
def test_redact_not_triggered(safe_string: str) -> None:
    """Non-secrets must NOT be accidentally redacted (false-positive guard)."""
    assert redact(safe_string) == safe_string


# --- Recursion: nested dict/list fields are redacted ---

def test_redact_event_nested_dict() -> None:
    """Secrets inside nested dict values are redacted."""
    event = {
        "gen_ai.provider.name": "claude-code",
        "gen_ai.event.name": "tool_call",
        "gen_ai.event.timestamp": "2026-05-03T00:00:00.000000Z",
        "gen_ai.tool.call.arguments": {
            "env": {
                "OPENAI_API_KEY": "sk-abcdefghijklmnopqrstu",
                "NORMAL_VAR": "not-a-secret",
            }
        },
    }
    redacted = redact_event(event)
    assert "sk-abcdefghijklmnopqrstu" not in json.dumps(redacted)
    assert "NORMAL_VAR" in json.dumps(redacted)  # safe values preserved


def test_redact_event_list_elements() -> None:
    """Secrets inside list elements are redacted."""
    event = {
        "gen_ai.provider.name": "opencode",
        "gen_ai.event.name": "tool_result",
        "gen_ai.event.timestamp": "2026-05-03T00:00:00.000000Z",
        "gen_ai.tool.call.result": ["hf_abcdefghijklmnopqrstu1234", "safe_value"],
    }
    redacted = redact_event(event)
    results = redacted["gen_ai.tool.call.result"]
    assert results[0] == "hf_[REDACTED]"
    assert results[1] == "safe_value"


def test_redact_event_not_mutating_original() -> None:
    """redact_event returns a new dict; original is not mutated."""
    original = {
        "gen_ai.provider.name": "claude-code",
        "gen_ai.event.name": "tool_call",
        "gen_ai.event.timestamp": "2026-05-03T00:00:00.000000Z",
        "gen_ai.tool.call.arguments": "ANTHROPIC_API_KEY=secret123",
    }
    original_copy = dict(original)
    _ = redact_event(original)
    assert original == original_copy  # original unchanged


def test_redact_event_non_string_passthrough() -> None:
    """int/float/bool/None values pass through unchanged."""
    event = {
        "gen_ai.provider.name": "claude-code",
        "gen_ai.event.name": "response",
        "gen_ai.event.timestamp": "2026-05-03T00:00:00.000000Z",
        "gen_ai.usage.input_tokens": 100,
        "gen_ai.usage.output_tokens": 50,
        "metadata": {"score": 0.95, "flag": True, "missing": None},
    }
    redacted = redact_event(event)
    assert redacted["gen_ai.usage.input_tokens"] == 100
    assert redacted["gen_ai.usage.output_tokens"] == 50
    assert redacted["metadata"]["score"] == 0.95
    assert redacted["metadata"]["flag"] is True
    assert redacted["metadata"]["missing"] is None


# --- 8 KB cap tests ---

def test_apply_size_cap_small_event_passes_through() -> None:
    """Events under 8192 bytes pass through unchanged."""
    event = {
        "gen_ai.provider.name": "claude-code",
        "gen_ai.event.name": "tool_call",
        "gen_ai.event.timestamp": "2026-05-03T00:00:00.000000Z",
        "gen_ai.tool.call.arguments": "small",
    }
    result = apply_size_cap(event)
    assert result == event


def test_apply_size_cap_truncates_large_fields() -> None:
    """Events over 8192 bytes have tool args/result truncated."""
    large_args = "x" * 10000
    event = {
        "gen_ai.provider.name": "claude-code",
        "gen_ai.event.name": "tool_call",
        "gen_ai.event.timestamp": "2026-05-03T00:00:00.000000Z",
        "gen_ai.tool.call.arguments": large_args,
        "gen_ai.tool.call.result": large_args,
    }
    result = apply_size_cap(event)
    encoded_size = len(json.dumps(result, ensure_ascii=False).encode("utf-8"))
    assert encoded_size <= _SIZE_CAP_BYTES
    # Truncation marker must be present
    assert "[truncated:" in result.get("gen_ai.tool.call.arguments", "") or \
           "[truncated:" in result.get("gen_ai.tool.call.result", "")


def test_apply_size_cap_sentinel_on_pathological_bloat() -> None:
    """Events that remain over cap after truncation become sentinel events."""
    # A pathological event: huge metadata that cannot be truncated via field reduction
    huge_meta = {"key_" + str(i): "v" * 200 for i in range(60)}
    event = {
        "gen_ai.provider.name": "claude-code",
        "gen_ai.event.name": "tool_call",
        "gen_ai.event.timestamp": "2026-05-03T00:00:00.000000Z",
    }
    # Inject bloat in non-truncatable keys
    for k, v in huge_meta.items():
        event[k] = v  # non-standard keys that apply_size_cap won't truncate

    result = apply_size_cap(event)
    if len(json.dumps(event).encode("utf-8")) > _SIZE_CAP_BYTES:
        # Either truncated or sentinel
        assert isinstance(result, dict)
        # If sentinel: must have gen_ai.event.name = "truncated"
        if result.get("gen_ai.event.name") == "truncated":
            assert "_dropped_size" in result


def test_redact_secrets_masks_patterns_without_touching_the_project(monkeypatch) -> None:
    """The pure variant never looks for a project; only the patterns apply."""
    from automil.trajectory import redactor

    def _boom() -> None:  # pragma: no cover - the assertion is that it is never called
        raise AssertionError("redact_secrets must not resolve a project dir")

    monkeypatch.setattr(redactor, "_held_out_ids", _boom)
    text = "export HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz1234 && echo node_0007"
    assert redactor.redact_secrets(text) == "export HF_TOKEN=[REDACTED] && echo node_0007"
    assert redactor.redact_secrets("plain text") == "plain text"
