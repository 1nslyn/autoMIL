"""Every classification arm persists BOTH splits' per-slide predictions.

``write_predictions_csv`` exists so that a metric nobody thought to compute at
training time is a recomputation rather than a retrain. That only holds if every
arm actually writes both files.

CLAM wrote only ``predictions.csv``. It computed the val predictions -- it needs
them for extended val metrics -- and then threw them away, so it was the single
arm whose validation split could not be re-scored without retraining. The gap
survived the whole 71-cell re-run and was caught by the job's own contract
check, not by anything in the test suite:

    FAIL clam/standard/kras/virchow2/clam_mb/s42: 0/5 predictions_val.csv

It also hand-rolled its CSV with pandas instead of using the shared writer,
which is how it drifted out of the contract unnoticed in the first place.
"""

from __future__ import annotations

import inspect

import pytest

#: Modules that together implement one arm's CLASSIFICATION prediction writing.
#: Some arms write from the trainer, some from their evaluator, so the contract
#: is checked per ARM rather than per module. Survival paths are excluded on
#: purpose: those produce risk scores, not class predictions.
_CLASSIFICATION_ARMS = [
    pytest.param(["autobench.pipeline.clam.train"], id="clam"),
    pytest.param(["autobench.pipeline.abmil.train"], id="abmil"),
    pytest.param(
        ["autobench.pipeline.dtfd.train", "autobench.pipeline.dtfd.eval"], id="dtfd",
    ),
    pytest.param(["autobench.pipeline.titan.train"], id="titan"),
    pytest.param(["autobench.pipeline.nnmil.metrics_addon"], id="nnmil"),
]


def _module_source(dotted: str) -> str:
    module = __import__(dotted, fromlist=["_"])
    return inspect.getsource(module)


def _arm_source(modules: list[str]) -> str:
    return "\n".join(_module_source(m) for m in modules)


@pytest.mark.parametrize("modules", _CLASSIFICATION_ARMS)
def test_arm_writes_both_prediction_files(modules):
    """Both splits, or the arm cannot be re-scored without a retrain."""
    source = _arm_source(modules)
    missing = []
    if "predictions.csv" not in source:
        missing.append("predictions.csv")
    # nnMIL builds the name per split (f"predictions_{prefix}.csv"), so a
    # literal match would report a gap that does not exist. Accept either form.
    if "predictions_val.csv" not in source and "predictions_{" not in source:
        missing.append("predictions_val.csv")
    assert not missing, (
        f"{modules} never writes {missing}. Every classification arm must "
        "persist per-slide predictions for BOTH splits, or a metric added later "
        "requires retraining that arm instead of recomputing from disk."
    )


@pytest.mark.parametrize("modules", _CLASSIFICATION_ARMS)
def test_arm_uses_the_shared_writer(modules):
    """One writer, one schema.

    A hand-rolled frame is how CLAM's columns and CLAM's contract drifted apart
    without anything noticing.
    """
    source = _arm_source(modules)
    assert "write_predictions_csv" in source, (
        f"{modules} does not use write_predictions_csv; a private CSV "
        "writer drifts from the shared schema silently"
    )
    assert ".to_csv(" not in source, (
        f"{modules} hand-rolls a predictions CSV with pandas. Use "
        "write_predictions_csv so all five arms share one column set."
    )


def test_clam_resume_requires_every_artifact():
    """A fold is 'done' only when the whole artifact set exists.

    CLAM's resume guard keyed on predictions.csv + metrics.json, so a fold
    missing predictions_val.csv looked complete and was skipped by every
    subsequent run -- which is why re-running never closed the gap.
    """
    source = _module_source("autobench.pipeline.clam.train")
    guard = source.split("Resume:", 1)[1].split("seed_everything", 1)[0]
    for artifact in ("predictions_path", "val_predictions_path", "metrics_path"):
        assert artifact in guard, (
            f"CLAM's resume guard ignores {artifact}; a fold missing it would "
            "resume as complete and the gap would survive every re-run"
        )
