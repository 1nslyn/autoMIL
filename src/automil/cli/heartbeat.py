"""heartbeat: stamp the agent-activity marker for the activity-gated budget (P2.1).

Fired automatically by the Claude Code ``on_tool.sh`` hook and by the CLI group
dispatcher on every command. Exposed as an explicit command for other runtimes'
hooks/plugins, manual pings, and tests.
"""
from __future__ import annotations

import click

from automil.cli import main
from automil.cli._helpers import _find_automil_dir


@main.command("heartbeat")
def heartbeat() -> None:
    """Stamp automil/.last_action_at (agent-active budget signal)."""
    from automil.cells.activity import touch_last_action  # lazy

    touch_last_action(_find_automil_dir())
