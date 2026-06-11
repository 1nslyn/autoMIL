"""automil CLI - Click main group and command registration.

The package re-exports ``main`` so ``from automil.cli import main`` keeps
working for tests and the pyproject ``[project.scripts]`` entry.
"""
from __future__ import annotations

import click


@click.group()
def main():
    """autoMIL: Autonomous agent-driven MIL model improvement."""
    # Activity signal (P2.1): every automil invocation is an agent action.
    # Stamp the project-level marker so the activity-gated budget counts CLI
    # work too — and so runtimes without Claude Code's PostToolUse hook (codex,
    # opencode) still register activity. Best-effort: never break a command if
    # we're not inside a project or the marker can't be written.
    try:
        from automil.cli._helpers import _find_automil_dir
        from automil.cells.activity import touch_last_action

        touch_last_action(_find_automil_dir())
    except Exception:
        pass


# Command modules register themselves on `main` at import time.
# Order is alphabetic for readability — Click registration is idempotent on
# repeated import so cycles are not a concern.
from automil.cli import budget  # noqa: E402,F401  (P2.3 — cap.budget show/set)
from automil.cli import cancel  # noqa: E402,F401
from automil.cli import cell    # noqa: E402,F401  (CAP-06 / D-125)
from automil.cli import cells   # noqa: E402,F401  (REC-04 / D-15)
from automil.cli import check  # noqa: E402,F401
from automil.cli import control  # noqa: E402,F401
from automil.cli import gate    # noqa: E402,F401  (GTE-01..06 / D-145)
from automil.cli import heartbeat  # noqa: E402,F401  (P2.1 — activity marker)
from automil.cli import init  # noqa: E402,F401
from automil.cli import lifecycle  # noqa: E402,F401
from automil.cli import nominate  # noqa: E402,F401  (GTE-05 / D-142)
from automil.cli import orchestrator  # noqa: E402,F401
from automil.cli import promote  # noqa: E402,F401  (GTE / D-145; bare 'promote', distinct from 'promote-variant')
from automil.cli import propose  # noqa: E402,F401
from automil.cli import reconcile  # noqa: E402,F401
from automil.cli import resubmit  # noqa: E402,F401
from automil.cli import show_skill  # noqa: E402,F401  (MRT-04 / D-93)
from automil.cli import status  # noqa: E402,F401
from automil.cli import submit  # noqa: E402,F401
from automil.cli import trajectory  # noqa: E402,F401  (TRJ-04, TRJ-05 / D-94)
from automil.cli import viz  # noqa: E402,F401

__all__ = ["main"]
