"""Module-level singleton storage for registered variants (REG-02 / D-27).

The orchestrator forks worker processes per experiment; each worker re-imports
variant modules at `train.py` startup, repopulating its own copy of these
dicts. There is NO cross-fork shared state; no fork-safety dance is needed
(CONTEXT specifics §"Registry singleton + fork safety").
"""
from __future__ import annotations

from automil.registry.spec import VariantSpec
from automil.registry.variants import LossVariant, ModelVariant, PolicyVariant

# Keyed dicts (D-27).
# MODEL_VARIANTS is keyed on (parent, name) since model variants are namespaced
# per-parent (clam_mb_v0176 != ab_mil_v0176 even if names ever collide).
# LOSS / POLICY are parent-agnostic (D-21) so a flat name key suffices.
MODEL_VARIANTS: dict[tuple[str, str], type[ModelVariant]] = {}
LOSS_VARIANTS: dict[str, type[LossVariant]] = {}
POLICY_VARIANTS: dict[str, type[PolicyVariant]] = {}

# SPEC_STORE keeps the VariantSpec for each registered class so Plan 01-08's
# `automil check` can cross-check the manifest JSON against the docstring
# header (D-44). Key shape mirrors the natural primary_value identity:
#   (kind, parent_or_None, name)
SPEC_STORE: dict[tuple[str, str | None, str], VariantSpec] = {}


def _clear_registry() -> None:
    """Test-only: clear all four dicts.

    PATTERNS.md §"Open codebase questions" #3 explicit requirement: tests
    clear singleton state between functions to prevent cross-test pollution.
    Production code never calls this — variant registration is import-time;
    un-registration during a process lifetime is not a v1 feature.
    """
    MODEL_VARIANTS.clear()
    LOSS_VARIANTS.clear()
    POLICY_VARIANTS.clear()
    SPEC_STORE.clear()
