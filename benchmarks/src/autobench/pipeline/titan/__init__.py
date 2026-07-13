"""TITAN integration layer for the benchmark pipeline.

Drives a frozen-slide-embedding linear probe via ``Framework.TITAN`` (see
docs/design/2026-07-05-mil-model-integration-design.md §7).
"""

from __future__ import annotations

from autobench.pipeline.titan.runner import run_titan_experiment

__all__ = ["run_titan_experiment"]
