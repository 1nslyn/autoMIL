"""A ghost ledger entry must make the worker RUN the cell, never return None.

``_completed.json`` is a cache of a fact that lives on disk, and it keeps
claiming cells whose directories were archived or purged. PR #56 taught both
schedulers that disk wins, but the GPU worker read the ledger too and was
missed. Its early-return guard did:

    if exp_id in load_completed(benchmark_dir):
        return _load_or_collect_summary(benchmark_dir, experiment)   # -> None

so a ghost entry returned a bare ``None``, and the caller turned that into
``RuntimeError: Experiment ... returned None`` -- which tears down the ENTIRE
block, stranding every healthy cell still queued behind it. On the 71-cell
re-run that was 6 ghosts per block across 6 blocks: 36 cells that could not
help but fail on contact, each taking ~10 siblings down with it. A link landed
2 cells instead of ~30.

It was also maximally confusing to diagnose. The guard sits BEFORE the
per-experiment log is opened, so the cell died in ~3 seconds writing nothing,
and the newest log on disk was days old -- from the last time it genuinely ran.
"""

from __future__ import annotations

import json
import os

import pytest

from autobench.pipeline._gpu_worker import resume_summary
from autobench.pipeline.config import (
    ExperimentConfig,
    Framework,
    ModelConfig,
    TaskConfig,
    TrainConfig,
)
from autobench.pipeline.orchestrator import mark_completed

from _helpers import make_ledger_exp, write_ledger_summary


def _exp(encoder: str = "virchow2"):
    return make_ledger_exp(
        task_name="kras", label_dict={"0": 0, "1": 1}, encoder=encoder,
        embed_dim=2560, dataset="tcga_luad",
    )


def _write_summary(benchmark_dir: str, exp) -> str:
    return write_ledger_summary(benchmark_dir, exp, task="kras")


def test_ghost_ledger_entry_makes_the_cell_run(tmp_path, capsys):
    """Ledgered, but the summary was archived -> must run, not return None."""
    bench = str(tmp_path)
    exp = _exp()
    summary = _write_summary(bench, exp)
    mark_completed(bench, exp.experiment_id)

    assert resume_summary(bench, exp) is not None, "a real completion must resume"

    os.remove(summary)  # the archive/move that orphaned the ledger entry

    assert resume_summary(bench, exp) is None, (
        "a ledger entry with no summary on disk must return None so the worker "
        "falls through and runs the cell; returning a stored summary here would "
        "report a phantom success"
    )
    assert exp.experiment_id in json.loads(
        (tmp_path / "results" / "_completed.json").read_text()
    ), "reconciliation is read-side only; the on-disk ledger must be untouched"
    assert "STALE-LEDGER" in capsys.readouterr().out, (
        "a ghost entry must be announced, not silently swallowed"
    )


def test_never_completed_experiment_returns_none(tmp_path):
    """Not in the ledger -> run it, even though a summary happens to exist.

    resume_summary must never promote an un-ledgered cell, or it would skip
    work the ledger never claimed was done.
    """
    bench = str(tmp_path)
    exp = _exp()
    _write_summary(bench, exp)

    assert resume_summary(bench, exp) is None


def test_worker_guard_cannot_return_a_bare_none(tmp_path):
    """Structural backstop: the guard must go through resume_summary.

    Pins the shape that made this fatal -- an early `return` of a possibly-None
    lookup, before the log file exists. If someone reinstates that, this fails.
    """
    import ast
    import inspect

    from autobench.pipeline import _gpu_worker

    tree = ast.parse(inspect.cleandoc(
        inspect.getsource(_gpu_worker.run_single_experiment)
    ))
    returns_before_logging: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Return) or node.value is None:
            continue
        # A direct `return <call>(...)` of the summary lookup is the bug.
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id in ("_load_or_collect_summary", "resume_summary")
        ):
            returns_before_logging.append(node.lineno)

    assert not returns_before_logging, (
        f"run_single_experiment returns a summary lookup directly at line(s) "
        f"{returns_before_logging}. That value is None for a ghost ledger entry, "
        "and the caller turns None into a RuntimeError that kills the whole "
        "block. Bind it and check for None first."
    )


@pytest.mark.parametrize("encoder", ["virchow2", "uni_v2", "hoptimus1"])
def test_each_encoder_variant_is_independent(encoder, tmp_path):
    """One cell's ghost entry must not affect its siblings.

    The real failure hit clam+abmil across all three encoders per block; this
    keeps the resolution per-experiment rather than per-task.
    """
    bench = str(tmp_path)
    ghost, healthy = _exp(encoder), _exp("other_encoder")
    _write_summary(bench, healthy)
    for exp in (ghost, healthy):
        mark_completed(bench, exp.experiment_id)

    assert resume_summary(bench, ghost) is None
    assert resume_summary(bench, healthy) is not None
