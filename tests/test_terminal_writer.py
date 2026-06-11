"""REC-02 / D-09, D-10: terminal_writer writes all four artifacts in fixed order.

D-09: A standalone terminal_writer module writes all four artifacts:
      graph node (via locked_update) → completed/<node>.json
      → archive result.json → results.tsv

D-10: terminal_writer is the sole results.tsv writer; updates graph via
      locked API, never direct dict mutation.

Both _handle_completion and _handle_cap_killed_completion delegate here.

RED until Plan 06 ships src/automil/terminal_writer.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_normal_completion_writes_all_four(tmp_path: Path) -> None:
    """D-09: all four artifacts exist after write_terminal_state for a completed result.

    RED until Plan 06 ships terminal_writer.py.
    """
    try:
        from automil.terminal_writer import write_terminal_state  # noqa: F401
    except ImportError:
        pytest.fail(
            "RED: automil.terminal_writer does not exist yet. "
            "Plan 06 must create terminal_writer.py with write_terminal_state()."
        )

    # If import succeeded (post-Plan 06), verify all four artifacts are written
    # (full test body to be exercised once Plan 06 ships)
    pytest.fail("RED: write_terminal_state imported but full test body not yet implemented.")


def test_cap_kill_writes_all_four(tmp_path: Path) -> None:
    """D-09: all four artifacts exist after write_terminal_state for a budget-kill result.

    RED until Plan 06 ships terminal_writer.py.
    """
    try:
        from automil.terminal_writer import write_terminal_state  # noqa: F401
    except ImportError:
        pytest.fail(
            "RED: automil.terminal_writer does not exist yet. "
            "Plan 06 must create terminal_writer.py with write_terminal_state()."
        )

    pytest.fail("RED: write_terminal_state imported but cap-kill test body not yet implemented.")


def test_graph_updated_before_tsv(tmp_path: Path) -> None:
    """D-09: fixed write order — graph.json mtime < results.tsv mtime after write_terminal_state.

    RED until Plan 06 ships terminal_writer.py.
    """
    try:
        from automil.terminal_writer import write_terminal_state  # noqa: F401
    except ImportError:
        pytest.fail(
            "RED: automil.terminal_writer does not exist yet. "
            "Plan 06 must create terminal_writer.py with write_terminal_state()."
        )

    pytest.fail("RED: write_terminal_state imported but write-order test body not yet implemented.")
