"""A4/A10 (claims-alignment): campaign identity locks and time containment.

The campaign policy's single source of truth is the hash-locked ``registry:``
block in each cohort template; ``search_space.py`` stays the mode-independent
mechanical declaration (free mode legitimately tunes widths). These tests pin
the invariants that make the flat per-dataset lock list safe, and that the
materialization audit asserts the lock VALUES rather than template fidelity.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from autobench.campaign import (
    CELL_TIME_CONTAINMENT,
    EXPECTED_ALLOWED_OVERRIDE_OPTIONS,
    EXPECTED_IDENTITY_LOCKED_HPARAMS,
    PROTOCOL,
)
from autobench.pipeline.hparams import FIELD_ALIASES
from autobench.pipeline.search_space import SEARCH_SPACE

REPO_ROOT = Path(__file__).resolve().parents[2]
ROSTER = ("tcga_luad", "tcga_lgg", "tcga_hnsc", "cptac_gbm", "cptac_pdac")


def _registry(dataset: str) -> dict:
    path = REPO_ROOT / "benchmarks" / "experiments" / dataset / "automil" / "config.yaml"
    return (yaml.safe_load(path.read_text()) or {})["registry"]


def test_each_locked_name_belongs_to_exactly_one_arm():
    """The flat union cannot false-lock another arm's knob: every locked name
    appears in exactly one arm's declared space (tunable or locked)."""
    for name in EXPECTED_IDENTITY_LOCKED_HPARAMS:
        owners = [
            arm for arm, space in SEARCH_SPACE.items()
            if name in space.tunable or name in space.locked
        ]
        assert len(owners) == 1, (
            f"{name!r} appears on {owners or 'no arm'} — the flat per-dataset "
            f"lock list is only safe while each name is unique to one arm"
        )


def test_no_field_alias_resolves_onto_a_locked_name():
    """An alias that maps a different canonical name onto a locked field would
    let a submit bypass the lock (or false-lock an innocent knob)."""
    locked = set(EXPECTED_IDENTITY_LOCKED_HPARAMS)
    for canonical, candidates in FIELD_ALIASES.items():
        if canonical in locked:
            continue
        overlap = set(candidates) & locked
        assert not overlap, (
            f"alias {canonical!r} -> {candidates} resolves onto locked {overlap}"
        )


def test_roster_templates_carry_the_expected_locks():
    for dataset in ROSTER:
        registry = _registry(dataset)
        assert sorted(registry["identity_locked_hparams"]) == sorted(
            EXPECTED_IDENTITY_LOCKED_HPARAMS
        ), f"{dataset}: identity_locked_hparams drifted from the audited values"
        assert sorted(registry["allowed_override_options"]) == sorted(
            EXPECTED_ALLOWED_OVERRIDE_OPTIONS
        ), f"{dataset}: allowed_override_options drifted from the audited values"


def test_capacity_knobs_stay_mechanically_tunable_in_free_mode():
    """Two-layer design: the campaign locks these; the mechanical channel does
    not — free mode legitimately tunes widths. Guard the layer separation."""
    for name in ("model_size", "M", "L", "mDim", "numLayer_Res", "hidden_dim"):
        assert any(name in space.tunable for space in SEARCH_SPACE.values()), (
            f"{name!r} left the mechanical search space — the campaign lock "
            f"belongs in the registry block, not in search_space.py"
        )


def test_protocol_declares_the_time_containment():
    assert PROTOCOL["cell_time_budget"]["budget"] == CELL_TIME_CONTAINMENT
    assert PROTOCOL["cell_time_budget"]["role"] == (
        "failure-containment-not-search-budget"
    )
    assert list(PROTOCOL["identity_locked_hparams"]) == list(
        EXPECTED_IDENTITY_LOCKED_HPARAMS
    )
