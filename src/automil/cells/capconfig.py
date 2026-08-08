"""Cap configuration parsing + resolution (P2.3).

Human-readable durations (``cap.budget: 6h``) and ``cap.mode`` are resolved
here against the current config schema and framework fallbacks.

Precedence (highest first): explicit CLI override → ``cap.<key>`` duration →
framework default.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Generic framework fallbacks; every consumer may override them. The frozen
# preprint campaign separately pins 12h plus exactly 30 launches.
DEFAULT_BUDGET_SECONDS = 21600
DEFAULT_SAFETY_BUFFER_SECONDS = 1800
DEFAULT_MODE = "agent_active"
VALID_MODES = ("agent_active", "wall_clock")

# H-2: the eval-count cap is an ORTHOGONAL SECOND AXIS, not a third mode —
# VALID_MODES stays time-only. ``None`` means "no eval cap", which reproduces
# the pre-H-2 time-only behaviour for every consumer that does not opt in.
DEFAULT_EVAL_BUDGET: int | None = None

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
    mode: str
    eval_budget: int | None = DEFAULT_EVAL_BUDGET


def _resolve_seconds(
    cap: dict, key: str, default: int, override: int | None,
) -> int:
    if override is not None:
        return int(override)
    if key in cap:
        return parse_duration(cap[key])
    return default


def parse_eval_budget(value: object) -> int | None:
    """Parse ``cap.eval_budget`` into a positive int, or ``None`` for no eval cap.

    A count, NOT a duration: ``"6h"`` is rejected rather than silently read as
    21600 evaluations. ``None`` (an omitted key or an explicit YAML ``null``)
    means the cell is time-only — today's behaviour.
    """
    if value is None:
        return None
    if isinstance(value, bool):  # bool is an int subclass — reject explicitly
        raise ValueError(f"cap.eval_budget must be a positive integer, got bool {value!r}")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.strip().lstrip("-").isdigit():
        parsed = int(value.strip())
    else:
        raise ValueError(
            f"cap.eval_budget must be a positive integer count of evaluations "
            f"(or null for no eval cap), got {value!r}"
        )
    if parsed <= 0:
        raise ValueError(
            f"cap.eval_budget must be > 0 (got {parsed}); use null to disable the eval cap"
        )
    return parsed


def resolve_cap_config(
    automil_cfg: object,
    *,
    budget_override: int | None = None,
    buffer_override: int | None = None,
    eval_budget_override: int | None = None,
) -> CapResolved:
    """Resolve the effective cap parameters from a parsed config.yaml dict.

    Raises ``ValueError`` on an invalid duration, an unknown ``cap.mode``, or a
    non-positive / non-integer ``cap.eval_budget``.
    """
    if not isinstance(automil_cfg, dict):
        raise ValueError("config must be a mapping")
    cap = automil_cfg.get("cap", {})
    if not isinstance(cap, dict):
        raise ValueError("cap must be a mapping")
    obsolete = sorted(
        key
        for key in (
            "budget_seconds",
            "safety_buffer_seconds",
            "idle_grace",
            "idle_grace_seconds",
        )
        if key in cap
    )
    if obsolete:
        raise ValueError(
            f"obsolete cap key(s) {obsolete}; use cap.budget and "
            "cap.safety_buffer durations (idle-grace billing no longer "
            "exists under native active-time metering)"
        )

    budget = _resolve_seconds(cap, "budget", DEFAULT_BUDGET_SECONDS, budget_override)
    buffer = _resolve_seconds(
        cap, "safety_buffer", DEFAULT_SAFETY_BUFFER_SECONDS, buffer_override,
    )
    mode = str(cap.get("mode", DEFAULT_MODE))
    if mode not in VALID_MODES:
        raise ValueError(
            f"cap.mode must be one of {VALID_MODES}, got {mode!r}"
        )

    # H-2: second axis, resolved independently of ``mode`` (which only meters
    # seconds). An explicit override wins; otherwise the config key; otherwise
    # no eval cap.
    if eval_budget_override is not None:
        eval_budget = parse_eval_budget(eval_budget_override)
    else:
        eval_budget = parse_eval_budget(cap.get("eval_budget", DEFAULT_EVAL_BUDGET))

    return CapResolved(
        budget_seconds=budget,
        safety_buffer_seconds=buffer,
        mode=mode,
        eval_budget=eval_budget,
    )
