"""DTFD-MIL integration layer for the benchmark pipeline.

Drives ``lib/DTFD-MIL`` directly via ``Framework.DTFD`` (see
docs/superpowers/specs/2026-07-05-mil-model-integration-design.md §6).
This module is currently a stub: the two-tier trainer has not landed yet.
"""

from __future__ import annotations

from autobench.pipeline.config import ExperimentConfig


def run_dtfd_experiment(
    exp_cfg: ExperimentConfig,
    benchmark_dir: str,
    device: str = "cuda:0",
) -> dict:
    """Run all folds for a single DTFD-MIL experiment (stub).

    Signature mirrors ``nnmil.runner.run_nnmil_experiment`` so dispatch in
    ``orchestrator.py`` / ``_gpu_worker.py`` is uniform across frameworks.
    """
    raise NotImplementedError("DTFD arm not yet implemented (stub)")
