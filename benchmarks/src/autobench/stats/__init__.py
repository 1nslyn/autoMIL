"""Statistical utilities for turning autobench results into paper claims.

Consumer-side by design: choosing an error rate and a comparison family is a
decision about one paper's claims, not framework behaviour, so none of this
belongs in ``src/automil/`` (see ``tests/test_framework_purity.py``).

Modules:
    multiple_comparisons: Holm-Bonferroni / Benjamini-Hochberg p-value
        adjustment (H-5b), plus the argued recommendation for what the
        comparison family should be for the per-cell lift headline.
"""
from __future__ import annotations

from autobench.stats.multiple_comparisons import (
    CORRECTION_METHODS,
    adjust,
    benjamini_hochberg,
    holm_bonferroni,
)

__all__ = [
    "CORRECTION_METHODS",
    "adjust",
    "benjamini_hochberg",
    "holm_bonferroni",
]
