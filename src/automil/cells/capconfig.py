"""Cap configuration parsing + resolution (P2.3).

Human-readable durations (``cap.budget: 6h``) and the ``cap.mode`` /
``cap.idle_grace`` knobs are resolved here, with back-compat for the legacy
integer-seconds keys (``cap.budget_seconds``) and the framework fallbacks.

Precedence (highest first): explicit CLI override → ``cap.<key>`` duration →
legacy ``cap.<key>_seconds`` int → framework default.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Framework fallbacks — Leo's autoMIL-paper campaign defaults; every consumer
# may override in config.yaml. 6h budget / 30m buffer / 5m idle-grace.
DEFAULT_BUDGET_SECONDS = 21600
DEFAULT_SAFETY_BUFFER_SECONDS = 1800
DEFAULT_IDLE_GRACE_SECONDS = 300
DEFAULT_MODE = "agent_active"
VALID_MODES = ("agent_active", "wall_clock")

_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*$", re.IGNORECASE)
_UNIT_SECONDS = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(value: object) -> int:
    """Parse ``'6h'`` / ``'30m'`` / ``'90s'`` / ``'2d'`` / ``3600`` into seconds.

    A bare number (int/float, or a unit-less numeric string) is interpreted as
    seconds. Raises ``ValueError`` on anything unparseable.
    """
    if isinstance(value, bool):  # bool is an int subclass — reject explicitly
        raise ValueError(f"duration must be a number or string, got bool {value!r}")
    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str):
        raise ValueError(f"duration must be a number or string, got {type(value).__name__}")
    m = _DURATION_RE.match(value)
    if not m:
        raise ValueError(
            f"invalid duration {value!r}; use e.g. 6h, 30m, 90s, 2d, or a number of seconds"
        )
    return int(float(m.group(1)) * _UNIT_SECONDS[m.group(2).lower()])


def format_duration(seconds: float) -> str:
    """Render seconds as the largest exact unit (``21600`` → ``'6h'``)."""
    s = int(seconds)
    if s > 0 and s % 86400 == 0:
        return f"{s // 86400}d"
    if s > 0 and s % 3600 == 0:
        return f"{s // 3600}h"
    if s > 0 and s % 60 == 0:
        return f"{s // 60}m"
    return f"{s}s"


@dataclass(frozen=True)
class CapResolved:
    """Resolved cap parameters for opening a cell."""

    budget_seconds: int
    safety_buffer_seconds: int
    idle_grace_seconds: int
    mode: str


def _resolve_seconds(cap: dict, dur_key: str, legacy_key: str, default: int,
                     override: int | None) -> int:
    if override is not None:
        return int(override)
    if dur_key in cap:
        return parse_duration(cap[dur_key])
    if legacy_key in cap:
        return parse_duration(cap[legacy_key])
    return default


def resolve_cap_config(
    automil_cfg: object,
    *,
    budget_override: int | None = None,
    buffer_override: int | None = None,
) -> CapResolved:
    """Resolve the effective cap parameters from a parsed config.yaml dict.

    Raises ``ValueError`` on an invalid duration or an unknown ``cap.mode``.
    """
    cap = automil_cfg.get("cap", {}) if isinstance(automil_cfg, dict) else {}
    if not isinstance(cap, dict):
        cap = {}

    budget = _resolve_seconds(cap, "budget", "budget_seconds", DEFAULT_BUDGET_SECONDS, budget_override)
    buffer = _resolve_seconds(cap, "safety_buffer", "safety_buffer_seconds",
                              DEFAULT_SAFETY_BUFFER_SECONDS, buffer_override)
    idle_grace = _resolve_seconds(cap, "idle_grace", "idle_grace_seconds",
                                  DEFAULT_IDLE_GRACE_SECONDS, None)

    mode = str(cap.get("mode", DEFAULT_MODE))
    if mode not in VALID_MODES:
        raise ValueError(
            f"cap.mode must be one of {VALID_MODES}, got {mode!r}"
        )
    return CapResolved(
        budget_seconds=budget,
        safety_buffer_seconds=buffer,
        idle_grace_seconds=idle_grace,
        mode=mode,
    )
