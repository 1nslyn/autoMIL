"""ABMIL integration layer for the benchmark pipeline.

Drives the vendored ``lib/AttentionDeepMIL`` reference (Ilse et al., 2018)
adapted for precomputed patch-encoder features, via ``Framework.ABMIL``: a
standard one-tier MIL trainer with two model variants -- non-gated
(``"abmil"``) and gated (``"abmil_gated"``) attention pooling -- and the shared
metric contract. Reuses nnMIL's H5-bag prep, exactly like DTFD.
"""

from __future__ import annotations

from autobench.pipeline.abmil.runner import run_abmil_experiment

__all__ = ["run_abmil_experiment"]
