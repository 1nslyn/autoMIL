"""REC-04 / D-14: mil_model normalization strips, lowercases, collapses internal whitespace.

D-14: mil_model is free-form, normalized — strip + lowercase + collapse internal
      whitespace before hashing into the cell key. CLAM_SB and clam_sb collapse
      to one cell.

Import: from automil.cells.state import normalize_mil_model

RED until Plan 02 ships normalize_mil_model in cells/state.py.
"""
from __future__ import annotations

import pytest


def test_clam_sb_variants_collapse() -> None:
    """D-14: CLAM_SB, clam_sb, and ' clam sb ' all normalize to the same value.

    RED until Plan 02 ships normalize_mil_model.
    """
    from automil.cells.state import normalize_mil_model  # noqa: F401 — RED until Plan 02

    result_upper = normalize_mil_model("CLAM_SB")
    result_lower = normalize_mil_model("clam_sb")
    result_spaced = normalize_mil_model(" clam sb ")

    assert result_upper == result_lower, (
        f"D-14: 'CLAM_SB' normalized to {result_upper!r} but 'clam_sb' to {result_lower!r}. "
        "Must collapse to same value."
    )
    assert result_lower == result_spaced, (
        f"D-14: 'clam_sb' → {result_lower!r} but ' clam sb ' → {result_spaced!r}. "
        "Must collapse to same value."
    )


def test_normalization_strips_leading_trailing() -> None:
    """D-14: normalize_mil_model strips leading and trailing whitespace.

    RED until Plan 02 ships normalize_mil_model.
    """
    from automil.cells.state import normalize_mil_model

    result = normalize_mil_model("  abmil  ")
    assert result == "abmil", (
        f"D-14: '  abmil  ' normalized to {result!r}, expected 'abmil'. "
        "Leading/trailing whitespace must be stripped."
    )


def test_normalization_collapses_internal_whitespace() -> None:
    """D-14: normalize_mil_model collapses internal whitespace runs to single space.

    RED until Plan 02 ships normalize_mil_model.
    """
    from automil.cells.state import normalize_mil_model

    result = normalize_mil_model("clam  sb")
    assert result == "clam sb", (
        f"D-14: 'clam  sb' normalized to {result!r}, expected 'clam sb'. "
        "Internal whitespace must be collapsed to a single space."
    )
