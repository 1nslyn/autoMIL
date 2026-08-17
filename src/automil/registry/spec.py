"""Frozen provenance record for registered variants (REG-01 / D-22).

`VariantSpec` is the registry key + provenance bundle for every registered
variant. Frozen because accidental mutation would silently corrupt the
registry; tuple (not list) for `mutations` for the same reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

Kind = Literal["model", "loss", "policy"]
"""Phase 1 exhaustive kind taxonomy (D-23).

`recipe` (composition of loss + policy + hyperparameters) and `inference`
(gate-time inference variants) are deferred — Phase 5 / Phase 8 widen this
when those concepts become real surfaces.
"""


@dataclass(frozen=True)
class VariantSpec:
    """Immutable provenance record for a registered variant.

    Fields land in this order (D-22 verbatim) so docstring tools and the
    manifest writer (Plan 01-11 port-variant) preserve a stable layout.
    """
    name: str
    kind: Kind
    parent: Optional[str]
    base_commit: str
    primary_value: float
    node_id: str
    created_at: str
    mutations: tuple[str, ...] = field(default_factory=tuple)
