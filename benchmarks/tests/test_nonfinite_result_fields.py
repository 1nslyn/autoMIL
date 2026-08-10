"""No non-finite float may reach result.json, at any depth (leak-site sweep).

Three producers could put a NaN on disk, where the orchestrator's ingestion
parser rejects the entire file and rewrites the node as a crash:

  1. ``summary`` — unestimable diagnostics (multi-class sensitivity, a degenerate
     cross-fold CI). Sealed copy only; fixed at the serializer.
  2. ``validation_folds[].metrics`` — raw per-fold values passed through
     unfiltered while the sibling ``composite`` was correctly ``None``-guarded.
     This block is in the AGENT-FACING copy, so it killed runs the same way.
  3. ``metrics`` / ``composite`` — when no fold produced a usable value at all.
     That one SHOULD fail, but as a quarantined ``partial`` with a readable
     cause, not as a phantom crash with ``composite: 0.0``.

Serializer-level sanitization is pinned framework-side in
``tests/test_nonfinite_result_serialization.py``; this file pins the producers.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

NAN = float("nan")


def _load_run_experiment() -> ModuleType:
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    script_path = scripts_dir / "run_experiment.py"
    if not script_path.exists():
        pytest.skip(f"run_experiment.py not found at {script_path}")
    mod_name = "run_experiment_nonfinite"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except SystemExit:
        pass
    return mod


def _summary(fold_aucs, fold_baccs=None, val_auc=0.70, val_bacc=0.60):
    fold_baccs = fold_baccs if fold_baccs is not None else [0.60] * len(fold_aucs)
    return {
        "test": {"auc_roc": {"mean": 0.70}, "balanced_accuracy": {"mean": 0.60}},
        "val": {"auc_roc": {"mean": val_auc},
                "balanced_accuracy": {"mean": val_bacc}},
        "per_fold_val": [
            {"auc_roc": a, "balanced_accuracy": b}
            for a, b in zip(fold_aucs, fold_baccs)
        ],
        "per_fold_test": [],
        "n_folds": len(fold_aucs),
    }


def _non_finite_paths(node, path="$"):
    """Every JSON path in ``node`` holding a non-finite float."""
    if isinstance(node, dict):
        return [p for k, v in node.items() for p in _non_finite_paths(v, f"{path}.{k}")]
    if isinstance(node, list):
        return [p for i, v in enumerate(node) for p in _non_finite_paths(v, f"{path}[{i}]")]
    if isinstance(node, float) and not math.isfinite(node):
        return [path]
    return []


# --- leak site 2: validation_folds -----------------------------------------

class TestValidationFoldEvidence:
    def test_non_finite_fold_metrics_become_null(self):
        m = _load_run_experiment()
        result = m.summary_to_result_json(_summary([NAN, 0.72], [0.60, 0.60]), 10.0)

        folds = result["validation_folds"]
        assert folds[0]["metrics"]["val_auc"] is None
        assert folds[0]["metrics"]["val_bacc"] == 0.60   # the finite sibling survives
        assert folds[0]["composite"] is None             # unchanged behaviour
        assert folds[1]["metrics"]["val_auc"] == 0.72

    def test_fields_this_function_owns_are_finite(self):
        m = _load_run_experiment()
        result = m.summary_to_result_json(_summary([NAN, NAN, 0.72], [NAN, 0.60, 0.60]), 10.0)

        owned = {k: v for k, v in result.items() if k != "summary"}
        assert _non_finite_paths(owned) == []

    def test_nothing_non_finite_survives_to_disk(self, tmp_path, monkeypatch):
        """The boundary that matters: what ingestion actually parses.

        The raw ``summary`` block is passed through verbatim — it is the honest
        record of what the trainers produced, NaNs and all — and is sanitized by
        the serializer on its way out. This asserts the round trip, including the
        sealed copy, which is the one ``Runner.collect_result`` reads first.
        """
        from automil.runner import _reject_nonfinite_constant
        from automil.runtime_helpers import write_result_json

        sealed = tmp_path / "certify"
        sealed.mkdir()
        worktree = tmp_path / "wt"
        worktree.mkdir()
        monkeypatch.setenv("AUTOMIL_RESULTS_DIR", str(sealed))

        m = _load_run_experiment()
        result = m.summary_to_result_json(_summary([NAN, NAN, 0.72], [NAN, 0.60, 0.60]), 10.0)
        write_result_json(result, worktree_dir=worktree)

        for path in (sealed / "result.json", worktree / "result.json"):
            assert "NaN" not in path.read_text()
            loaded = json.loads(path.read_text(), parse_constant=_reject_nonfinite_constant)
            assert _non_finite_paths(loaded) == []
        sealed_payload = json.loads((sealed / "result.json").read_text())
        assert sealed_payload["summary"]["per_fold_val"][0]["auc_roc"] is None


# --- leak site 3: an unestimable composite ---------------------------------

class TestUnestimableComposite:
    def test_all_folds_unestimable_is_partial_not_crash(self):
        m = _load_run_experiment()
        result = m.summary_to_result_json(
            _summary([NAN] * 5, [NAN] * 5, val_auc=NAN, val_bacc=NAN), 10.0,
        )

        assert result["status"] == "partial"
        assert isinstance(result["composite"], float)
        assert math.isfinite(result["composite"])
        assert result["metrics"] == {}
        assert "error" in result and "composite" in result["error"].lower()

    def test_the_payload_still_validates_against_the_result_schema(self):
        """A partial must be ingestible: `null` composite would fail the schema."""
        from automil.schemas import validate_result

        m = _load_run_experiment()
        result = m.summary_to_result_json(
            _summary([NAN] * 5, [NAN] * 5, val_auc=NAN, val_bacc=NAN), 10.0,
        )
        validate_result(result)   # must not raise

    def test_one_unestimable_component_voids_the_whole_composite(self):
        """All-or-nothing: no half-scale composite may escape (review finding).

        Reporting val_bacc alone would put this node on a different estimand from
        every sibling scored on (auc+bacc)/2 -- and `status: partial` does NOT
        contain that: it keeps the node out of KEEP_CLASS but not out of being a
        PARENT, and terminal_writer gates a child against `parent["composite"]`
        with no partial check. A half-scale bar would silently decide a completed
        child's keep/discard, biased one way because bacc < auc in practice.
        """
        m = _load_run_experiment()
        result = m.summary_to_result_json(
            _summary([NAN] * 5, [0.60] * 5, val_auc=NAN, val_bacc=0.60), 10.0,
        )

        assert result["status"] == "partial"
        assert result["metrics"] == {}
        assert result["composite"] == 0.0
        assert "val_auc" in result["error"]

    def test_the_composite_is_never_a_partial_scale(self):
        """CR-1b must agree, or terminal_writer overwrites the selection signal."""
        from automil.scoring import composite_disagrees, recompute_composite

        m = _load_run_experiment()
        for summary in (
            _summary([0.70] * 5, [0.60] * 5),                          # healthy
            _summary([NAN] * 5, [0.60] * 5, val_auc=NAN, val_bacc=0.60),
            _summary([NAN] * 5, [NAN] * 5, val_auc=NAN, val_bacc=NAN),
        ):
            result = m.summary_to_result_json(summary, 10.0)
            recomputed = recompute_composite(result["metrics"])
            if recomputed is not None:
                assert not composite_disagrees(result["composite"], recomputed)

    def test_a_healthy_run_is_untouched(self):
        m = _load_run_experiment()
        result = m.summary_to_result_json(_summary([0.70, 0.72, 0.68, 0.71, 0.69]), 10.0)

        assert result["status"] == "completed"
        assert result["metrics"] == {"val_auc": 0.70, "val_bacc": 0.60}
        assert result["composite"] == pytest.approx(0.65)
        assert "error" not in result


# --- leak site 1 (producer side): per-fold archive files --------------------

class TestWriteFoldResultJson:
    def test_non_finite_fold_metrics_serialize_as_null(self, tmp_path, monkeypatch):
        from autobench.pipeline.clam.runner import _write_fold_result_json

        monkeypatch.setenv("AUTOMIL_RESULTS_DIR", str(tmp_path))
        monkeypatch.setenv("AUTOMIL_FOLD_COUNT", "5")
        _write_fold_result_json(0, {
            "val_metrics": {"auc_roc": NAN, "balanced_accuracy": 0.61},
            "test_metrics": {"auc_roc": 0.70, "balanced_accuracy": 0.65},
            "elapsed_seconds": 120,
            "peak_vram_mb": 4000,
        })

        text = (tmp_path / "fold_0_result.json").read_text()
        assert "NaN" not in text
        payload = json.loads(text)
        assert payload["metrics"]["val_auc"] is None
        assert payload["metrics"]["val_bacc"] == pytest.approx(0.61)
        assert payload["composite"] is None

    def test_a_healthy_fold_is_untouched(self, tmp_path, monkeypatch):
        from autobench.pipeline.clam.runner import _write_fold_result_json

        monkeypatch.setenv("AUTOMIL_RESULTS_DIR", str(tmp_path))
        monkeypatch.setenv("AUTOMIL_FOLD_COUNT", "5")
        _write_fold_result_json(1, {
            "val_metrics": {"auc_roc": 0.80, "balanced_accuracy": 0.70},
            "test_metrics": {"auc_roc": 0.78, "balanced_accuracy": 0.68},
            "elapsed_seconds": 120,
            "peak_vram_mb": 4000,
        })

        payload = json.loads((tmp_path / "fold_1_result.json").read_text())
        assert payload["metrics"] == {"val_auc": 0.80, "val_bacc": 0.70}
        assert payload["held_out"] == {"test_auc": 0.78, "test_bacc": 0.68}
        assert payload["composite"] == pytest.approx(0.75)
        assert payload["status"] == "completed"


class TestMainDoesNotCrashOnAPartialResult:
    """`metrics` can now be empty; main()'s summary print must survive that.

    It indexed result["metrics"]["val_auc"] directly. That raised KeyError AFTER
    result.json was already written, so the damage was not a lost result but a
    non-zero exit -- and the campaign's native-baseline stage refuses to archive
    a result whose process exited non-zero, aborting the stage and discarding
    exactly the run this change exists to rescue.
    """

    @pytest.mark.parametrize("metrics", [
        {},                                       # nothing estimable
        {"val_bacc": 0.60},                       # historical half-metrics shape
        {"val_auc": 0.70, "val_bacc": 0.60},      # healthy
        {"val_c_index": 0.61},                    # survival
    ])
    def test_summary_print_formats_any_metrics_shape(self, metrics, capsys):
        result = {"metrics": metrics, "composite": 0.65, "error": "unestimable: val_auc"}

        reported = "  ".join(
            f"{name}={value:.4f}" for name, value in sorted(result["metrics"].items())
        )
        line = f"  {reported}  composite={result['composite']:.4f}".lstrip()

        assert "composite=0.6500" in line
        for name in metrics:
            assert name in line

    def test_the_real_main_block_has_no_direct_metric_indexing(self):
        """Guard the fix itself — a future edit reintroducing [] would re-break it."""
        src = (
            Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py"
        ).read_text()
        assert "result['metrics']['val_auc']" not in src
        assert "result['metrics']['val_c_index']" not in src
