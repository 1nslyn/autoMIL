"""Agent-activity marker for the activity-gated budget (P2.1).

A single project-level file ``automil/.last_action_at`` records the wall-clock
instant of the agent's most recent action. It is stamped by:

  - the Claude Code ``PostToolUse`` hook (``on_tool.sh``) on EVERY tool call —
    reads, edits, web research, submits — the complete span of agent work;
  - every ``automil`` CLI invocation (the group dispatcher), as a cross-runtime
    fallback for agents without Claude Code hooks.

The orchestrator daemon reads it each tick and only advances an ``agent_active``
cell's budget while the agent acted within ``idle_grace_seconds``. The agent is
provably quiescent (no tool calls) while waiting on experiments, so the clock
pauses then — billing agent working time, not GPU wall-clock.

The marker is a single float (Unix epoch seconds). Writers are best-effort and
never raise; the reader tolerates a missing/torn/malformed file by returning
``None`` (the daemon treats that as "no recent activity" → does not accrue).
"""
from __future__ import annotations

import time
from pathlib import Path

ACTIVITY_FILENAME = ".last_action_at"


def touch_last_action(automil_dir: Path | str, *, now: float | None = None) -> None:
    """Stamp the agent-activity marker. Best-effort — never raises.

    Args:
        automil_dir: the ``automil/`` overlay directory.
        now: epoch seconds to write (defaults to ``time.time()``); injectable
            for tests.
    """
    try:
        ts = now if now is not None else time.time()
        (Path(automil_dir) / ACTIVITY_FILENAME).write_text(f"{ts:.3f}\n")
    except OSError:
        pass


def read_last_action_at(automil_dir: Path | str) -> float | None:
    """Return the last agent-activity epoch, or ``None`` if absent/unreadable.

    ``None`` is the conservative signal: the daemon does not accrue budget for a
    tick when activity is unknown, and self-corrects on the next readable tick.
    """
    try:
        return float((Path(automil_dir) / ACTIVITY_FILENAME).read_text().strip())
    except (OSError, ValueError):
        return None
