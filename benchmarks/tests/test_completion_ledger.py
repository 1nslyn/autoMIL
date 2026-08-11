"""``_completed.json`` is a cache; ``results/`` is the truth.

The ledger and the results tree drift apart the moment a cell directory moves
— archiving a stale cell, clearing one arm to re-run it, a purge that misses
the ledger. Nothing rewrites ``_completed.json`` when that happens, so it goes
on claiming completion for a cell with nothing behind it.

That cost a real 5-arm re-run: 6 of 13 cells were skipped as "already done"
while their directories sat in ``results.stale-.../``, the other 7 retrained
for 3h49m, and the job then died in
``_collect_all_summaries_or_raise`` looking for summaries that had been moved
hours earlier. Both failure modes matter — the silent skip is the worse one,
because a re-run whose whole purpose was to make every arm homogeneous quietly
left 6 arms on old results.

These tests pin the reconciliation that makes the ledger self-heal.
"""

from __future__ import annotations

import json
import os

import pytest

from autobench.pipeline.config import (
    ExperimentConfig,
    Framework,
    ModelConfig,
    TaskConfig,
    TrainConfig,
)
from autobench.pipeline.orchestrator import (
    load_completed,
    mark_completed,
    reconcile_completed,
)


def _exp(
    *,
    task_name: str = "grade",
    encoder: str = "virchow2",
    model_type: str = "clam_mb",
    framework: Framework = Framework.CLAM,
    dataset: str = "tcga_hnsc",
    seed: int = 42,
) -> ExperimentConfig:
    """One experiment at the REAL path shape (``results_subdir``)."""
    return ExperimentConfig(
        task=TaskConfig(
            name=task_name, label_col="label",
            label_dict={"g1": 0, "g2": 1, "g3": 2}, task_type="classification",
        ),
        encoder_key=encoder,
        embed_dim=768,
        model=ModelConfig(model_type=model_type),
        train=TrainConfig(seed=seed),
        n_folds=5,
        framework=framework,
        strategy="standard",
        dataset=dataset,
    )


def _write_summary(benchmark_dir: str, exp: ExperimentConfig) -> str:
    """Materialize the cell directory the orchestrator will look for."""
    cell = os.path.join(benchmark_dir, "results", exp.results_subdir)
    os.makedirs(cell, exist_ok=True)
    path = os.path.join(cell, "summary.json")
    with open(path, "w") as f:
        json.dump({"experiment_id": exp.experiment_id}, f)
    return path


class TestReconcileCompleted:
    def test_moved_cell_is_no_longer_completed(self, tmp_path):
        """The exact HNSC-54008144 failure: ledger kept, directory moved.

        Without reconciliation this cell is skipped by the run AND fatal at
        summary collection.
        """
        bench = str(tmp_path)
        exp = _exp()
        summary = _write_summary(bench, exp)
        mark_completed(bench, exp.experiment_id)

        assert reconcile_completed(bench, [exp], load_completed(bench)) == {
            exp.experiment_id,
        }

        os.remove(summary)  # the archive/move that orphaned the ledger entry

        assert reconcile_completed(bench, [exp], load_completed(bench)) == set()
        assert exp.experiment_id in load_completed(bench), (
            "the on-disk ledger is untouched; reconciliation is a read-side "
            "correction, so a concurrent writer's entries are never dropped"
        )

    def test_orphaned_entry_lands_back_in_pending(self, tmp_path):
        """Reconciliation must re-queue the cell, not merely un-count it.

        This is the half that kept 6 arms on stale results: the run reported
        them as completed and never scheduled them.
        """
        bench = str(tmp_path)
        kept, orphan = _exp(encoder="virchow2"), _exp(encoder="uni_v2")
        _write_summary(bench, kept)
        for exp in (kept, orphan):
            mark_completed(bench, exp.experiment_id)

        experiments = [kept, orphan]
        completed = reconcile_completed(bench, experiments, load_completed(bench))
        pending = [e for e in experiments if e.experiment_id not in completed]

        assert [e.experiment_id for e in pending] == [orphan.experiment_id]

    def test_completion_is_never_invented(self, tmp_path):
        """A summary on disk does NOT make a cell completed on its own.

        Reconciliation only ever removes. Promoting an un-ledgered cell would
        skip work the ledger never claimed was done.
        """
        bench = str(tmp_path)
        exp = _exp()
        _write_summary(bench, exp)

        assert load_completed(bench) == set()
        assert reconcile_completed(bench, [exp], load_completed(bench)) == set()

    def test_scoped_to_the_current_grid(self, tmp_path):
        """Only this run's experiments are considered.

        The ledger spans every task ever run in the benchmark dir, which is why
        the old banner read "Total: 13  Completed: 12" for a 13-cell grid — the
        12 counted `os` cells that were not part of it.
        """
        bench = str(tmp_path)
        in_grid = _exp(task_name="grade")
        other_task = _exp(task_name="os")
        for exp in (in_grid, other_task):
            _write_summary(bench, exp)
            mark_completed(bench, exp.experiment_id)

        assert reconcile_completed(bench, [in_grid], load_completed(bench)) == {
            in_grid.experiment_id,
        }

    def test_empty_ledger_stays_empty(self, tmp_path):
        assert reconcile_completed(str(tmp_path), [_exp()], set()) == set()


class TestBothSchedulersReconcile:
    """Neither entry point may read the ledger raw."""

    @pytest.mark.parametrize(
        "function_name", ["run_benchmark", "run_benchmark_multigpu"],
    )
    def test_scheduler_reconciles_before_computing_pending(self, function_name):
        import ast
        import inspect

        from autobench.pipeline import orchestrator

        source = inspect.getsource(getattr(orchestrator, function_name))
        tree = ast.parse(inspect.cleandoc(source))

        calls = [
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        assigned_from_raw = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "load_completed"
            and any(
                isinstance(t, ast.Name) and t.id == "completed"
                for t in node.targets
            )
        ]

        assert "reconcile_completed" in calls, (
            f"{function_name} does not reconcile the ledger against disk; a "
            "moved cell directory will be skipped as completed and then kill "
            "the run at summary collection"
        )
        assert not assigned_from_raw, (
            f"{function_name} binds `completed` straight from load_completed() "
            f"at line(s) {assigned_from_raw}; it must go through "
            "reconcile_completed() so disk stays the source of truth"
        )
