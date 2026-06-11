"""REC-02: automil rank and results.tsv agree after terminal_writer completes.

After write_terminal_state:
  - `automil rank` must report the same best composite as results.tsv last row.
  - TSV and graph composite values must be identical (no split-write drift).

RED until Plan 06 ships terminal_writer.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_rank_and_tsv_agree_after_terminal_write(tmp_path: Path) -> None:
    """D-09/D-10: automil rank composite == results.tsv last-row composite after write_terminal_state.

    RED until Plan 06 ships terminal_writer.py.
    """
    try:
        from automil.terminal_writer import write_terminal_state  # noqa: F401
    except ImportError:
        pytest.fail(
            "RED: automil.terminal_writer does not exist yet. "
            "Plan 06 must create terminal_writer.py with write_terminal_state()."
        )

    # If import succeeded, test body exercises rank/TSV agreement
    pytest.fail("RED: write_terminal_state imported but rank/TSV agreement test not yet implemented.")
