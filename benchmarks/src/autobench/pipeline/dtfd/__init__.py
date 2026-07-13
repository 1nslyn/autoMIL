"""DTFD-MIL integration layer for the benchmark pipeline.

Drives the vendored ``lib/DTFD-MIL`` reference directly via ``Framework.DTFD``
(see docs/design/2026-07-05-mil-model-integration-design.md §6): the
real two-tier arm — random pseudo-bag split, tier-1 (DimReduction + gated
Attention + Classifier_1fc) per pseudo-bag, AFS instance distillation, tier-2
Attention_with_Classifier — with two optimizers and the shared metric contract.
"""

from __future__ import annotations

from autobench.pipeline.dtfd.runner import run_dtfd_experiment

__all__ = ["run_dtfd_experiment"]
