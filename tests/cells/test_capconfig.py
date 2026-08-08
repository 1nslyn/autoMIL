"""Unit tests for cap config parsing + resolution (P2.3)."""
from __future__ import annotations

import pytest

from automil.cells.capconfig import (
    DEFAULT_BUDGET_SECONDS,
    DEFAULT_MODE,
    DEFAULT_SAFETY_BUFFER_SECONDS,
    format_duration,
    parse_duration,
    resolve_cap_config,
)


class TestParseDuration:
    @pytest.mark.parametrize("value,expected", [
        ("6h", 21600), ("30m", 1800), ("90s", 90), ("2d", 172800),
        ("3600", 3600), ("1.5h", 5400), (3600, 3600), (300.0, 300),
        ("  6h ", 21600), ("45M", 2700),
    ])
    def test_valid(self, value, expected):
        assert parse_duration(value) == expected

    @pytest.mark.parametrize("value", ["6 hours", "abc", "", "h", "6x", None, True])
    def test_invalid(self, value):
        with pytest.raises(ValueError):
            parse_duration(value)


class TestFormatDuration:
    @pytest.mark.parametrize("seconds,expected", [
        (21600, "6h"), (1800, "30m"), (90, "90s"), (172800, "2d"),
        (300, "5m"), (0, "0s"), (3661, "3661s"),
    ])
    def test_format(self, seconds, expected):
        assert format_duration(seconds) == expected


class TestResolveCapConfig:
    def test_defaults_when_no_cap_block(self):
        cap = resolve_cap_config({})
        assert cap.budget_seconds == DEFAULT_BUDGET_SECONDS
        assert cap.safety_buffer_seconds == DEFAULT_SAFETY_BUFFER_SECONDS
        assert cap.mode == DEFAULT_MODE == "agent_active"

    def test_duration_keys(self):
        cfg = {"cap": {"budget": "2h", "safety_buffer": "10m",
                       "mode": "wall_clock"}}
        cap = resolve_cap_config(cfg)
        assert cap.budget_seconds == 7200
        assert cap.safety_buffer_seconds == 600
        assert cap.mode == "wall_clock"

    def test_obsolete_seconds_keys_are_rejected(self):
        cfg = {"cap": {"budget_seconds": 100, "safety_buffer_seconds": 10}}
        with pytest.raises(ValueError, match="obsolete cap key"):
            resolve_cap_config(cfg)

    @pytest.mark.parametrize("key", ["idle_grace", "idle_grace_seconds"])
    def test_obsolete_idle_grace_keys_are_rejected_not_ignored(self, key):
        """A removed billing knob must fail loudly, not become a silent no-op."""
        with pytest.raises(ValueError, match="obsolete cap key"):
            resolve_cap_config({"cap": {key: 300}})

    def test_cli_override_wins_over_config(self):
        cfg = {"cap": {"budget": "6h"}}
        cap = resolve_cap_config(cfg, budget_override=42, buffer_override=7)
        assert cap.budget_seconds == 42
        assert cap.safety_buffer_seconds == 7

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            resolve_cap_config({"cap": {"mode": "bogus"}})

    def test_invalid_duration_raises(self):
        with pytest.raises(ValueError):
            resolve_cap_config({"cap": {"budget": "later"}})

    def test_non_dict_cap_rejected(self):
        with pytest.raises(ValueError, match="cap must be a mapping"):
            resolve_cap_config({"cap": "nonsense"})
