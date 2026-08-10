"""result.json must be standards-valid JSON: non-finite floats serialize as null.

The bug this pins
-----------------
``sensitivity``/``specificity`` are unestimable on a multi-class task, so the
per-fold value is NaN and the cross-fold CI degenerates to
``{"mean": nan, "ci_low": nan, ...}``. Those live in the ``summary`` diagnostic
block -- never in ``metrics``, never in ``composite`` -- yet they killed the run:

  1. ``_atomic_write_json`` used ``json.dumps`` at its default ``allow_nan=True``,
     emitting a bare ``NaN`` token (invalid JSON per RFC 8259).
  2. ``write_result_json`` keeps ``summary`` in the SEALED copy.
  3. ``Runner.collect_result`` reads the sealed copy FIRST and parses it with the
     CR-1a ``parse_constant`` hook, which rejects the whole file.
  4. The node was rewritten as ``{"status": "crash", "composite": 0.0}``.

So an unestimable *diagnostic* destroyed a scientifically valid run. The fix is
at the serializer, not the guard: a non-finite value is written as JSON ``null``
("no value available"), which is exactly what it means. CR-1a stays in force --
a hand-written ``NaN`` token is still rejected, and a ``null`` composite still
fails the schema's ``{"type": "number"}``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from automil.runner import Runner, _reject_nonfinite_constant
from automil.runtime_helpers import _atomic_write_json, write_result_json


def _strict_load(path: Path) -> dict:
    """Parse exactly the way Runner.collect_result does."""
    return json.loads(path.read_text(), parse_constant=_reject_nonfinite_constant)


class TestAtomicWriteJson:
    def test_nan_serializes_as_null(self, tmp_path):
        path = tmp_path / "result.json"
        _atomic_write_json(path, {"composite": 0.8, "diagnostic": float("nan")})

        assert "NaN" not in path.read_text()
        assert _strict_load(path)["diagnostic"] is None

    def test_infinities_serialize_as_null(self, tmp_path):
        path = tmp_path / "result.json"
        _atomic_write_json(path, {"hi": float("inf"), "lo": float("-inf")})

        text = path.read_text()
        assert "Infinity" not in text
        loaded = _strict_load(path)
        assert loaded == {"hi": None, "lo": None}

    def test_sanitization_reaches_nested_containers(self, tmp_path):
        """The NaN that caused this bug sits three levels deep, inside a list."""
        path = tmp_path / "result.json"
        _atomic_write_json(path, {
            "summary": {
                "val": {"sensitivity": {"mean": float("nan"), "method": "degenerate"}},
                "per_fold_val": [{"auc_roc": float("nan")}, {"auc_roc": 0.7}],
            },
        })

        loaded = _strict_load(path)
        assert loaded["summary"]["val"]["sensitivity"]["mean"] is None
        assert loaded["summary"]["val"]["sensitivity"]["method"] == "degenerate"
        assert loaded["summary"]["per_fold_val"] == [{"auc_roc": None}, {"auc_roc": 0.7}]

    def test_finite_payloads_are_untouched(self, tmp_path):
        payload = {
            "status": "completed",
            "metrics": {"val_auc": 0.87, "val_bacc": 0.81},
            "composite": 0.84,
            "composite_se": None,
            "n_folds": 5,
            "flag": True,
            "note": "unchanged",
        }
        path = tmp_path / "result.json"
        _atomic_write_json(path, payload)

        assert _strict_load(path) == payload

    def test_caller_payload_is_not_mutated(self, tmp_path):
        """Immutability: sanitization builds a new structure (CLAUDE.md)."""
        inner = {"mean": float("nan")}
        payload = {"summary": inner}
        _atomic_write_json(tmp_path / "result.json", payload)

        assert math.isnan(inner["mean"])
        assert payload["summary"] is inner


class TestCollectResultSurvivesUnestimableDiagnostics:
    """End-to-end: the exact multi-class shape that was being rejected."""

    def _sealed_and_worktree(self, tmp_path, monkeypatch):
        archive = tmp_path / "archive" / "node_a"
        sealed = archive / "certify"
        sealed.mkdir(parents=True)
        worktree = tmp_path / "wt"
        worktree.mkdir()
        monkeypatch.setenv("AUTOMIL_RESULTS_DIR", str(sealed))
        return archive, worktree

    def test_nan_summary_no_longer_crashes_the_node(self, tmp_path, monkeypatch):
        archive, worktree = self._sealed_and_worktree(tmp_path, monkeypatch)
        write_result_json(
            {
                "status": "completed",
                "metrics": {"val_auc": 0.87, "val_bacc": 0.81},
                "held_out": {"test_auc": 0.85, "test_bacc": 0.80},
                "composite": 0.84,
                # Unestimable on a 3-class task -- always NaN before the fix.
                "summary": {
                    "val": {
                        "sensitivity": {"mean": float("nan"), "n_valid_folds": 0},
                        "specificity": {"mean": float("nan"), "n_valid_folds": 0},
                        "auc_roc": {"mean": 0.87, "n_valid_folds": 5},
                    },
                },
            },
            worktree_dir=worktree,
        )

        result = Runner(tmp_path).collect_result(worktree, archive)

        assert result["status"] == "completed"
        assert result["composite"] == 0.84
        assert result["summary"]["val"]["sensitivity"]["mean"] is None
        assert result["summary"]["val"]["auc_roc"]["mean"] == 0.87
        assert "error" not in result

    def test_hand_written_nan_token_is_still_rejected(self, tmp_path):
        """CR-1a is untouched: an agent cannot smuggle in a non-finite composite."""
        archive = tmp_path / "archive" / "node_b"
        sealed = archive / "certify"
        sealed.mkdir(parents=True)
        worktree = tmp_path / "wt"
        worktree.mkdir()
        (sealed / "result.json").write_text(
            '{"status": "completed", "composite": Infinity, "metrics": {}}'
        )

        result = Runner(tmp_path).collect_result(worktree, archive)

        assert result["status"] == "crash"
        assert result["composite"] == 0.0
        assert "rejected at ingestion" in result["error"]


class TestNullCompositeStillFailsClosed:
    def test_schema_rejects_a_null_composite(self):
        """Sanitizing does not let a broken selection signal through."""
        from automil.schemas import ValidationError, validate_result

        with pytest.raises(ValidationError):
            validate_result({"status": "completed", "composite": None, "metrics": {}})

    def test_schema_rejects_a_null_metric(self):
        from automil.schemas import ValidationError, validate_result

        with pytest.raises(ValidationError):
            validate_result(
                {"status": "completed", "composite": 0.8, "metrics": {"val_auc": None}}
            )


class TestNumpyScalars:
    """``isinstance(x, float)`` is False for np.float32 (np.float64 subclasses it).

    A np.float32 NaN therefore skipped sanitization entirely and reached
    ``json.dumps``, which raises TypeError on numpy scalars — inside the SIGTERM
    handler, that costs the whole partial flush. The finiteness test is on
    ``numbers.Real`` (stdlib, so no numpy dependency on the framework side).
    """

    def test_non_finite_numpy_floats_become_null(self, tmp_path):
        np = pytest.importorskip("numpy")
        path = tmp_path / "result.json"
        _atomic_write_json(path, {
            "f32": np.float32("nan"),
            "f64": np.float64("nan"),
            "inf32": np.float32("inf"),
        })

        assert _strict_load(path) == {"f32": None, "f64": None, "inf32": None}

    def test_finite_numpy_scalars_are_serializable(self, tmp_path):
        """The contract is that whatever json_safe returns can actually be dumped."""
        np = pytest.importorskip("numpy")
        path = tmp_path / "result.json"
        _atomic_write_json(path, {
            "f32": np.float32(0.5), "f64": np.float64(0.25), "i64": np.int64(7),
        })

        loaded = _strict_load(path)
        assert loaded == {"f32": 0.5, "f64": 0.25, "i64": 7}
        assert isinstance(loaded["i64"], int)

    def test_bools_stay_json_booleans(self, tmp_path):
        """bool is an Integral subclass and must not be narrowed to 1/0."""
        path = tmp_path / "result.json"
        _atomic_write_json(path, {"yes": True, "no": False, "one": 1})

        assert '"yes": true' in path.read_text()
        assert _strict_load(path) == {"yes": True, "no": False, "one": 1}
