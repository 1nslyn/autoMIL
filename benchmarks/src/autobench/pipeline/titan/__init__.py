"""TITAN integration layer for the benchmark pipeline.

Drives a frozen-slide-embedding linear probe via ``Framework.TITAN`` (see
docs/superpowers/specs/2026-07-05-mil-model-integration-design.md §7).
This module is currently a stub: the linear-probe trainer has not landed yet.
"""

from __future__ import annotations

from autobench.pipeline.config import ExperimentConfig


def run_titan_experiment(
    exp_cfg: ExperimentConfig,
    benchmark_dir: str,
    device: str = "cuda:0",
) -> dict:
    """Run all folds for a single TITAN experiment (stub).

    Signature mirrors ``nnmil.runner.run_nnmil_experiment`` so dispatch in
    ``orchestrator.py`` / ``_gpu_worker.py`` is uniform across frameworks.
    """
    raise NotImplementedError("TITAN arm not yet implemented (stub)")
