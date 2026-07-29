"""L-5 (n_classes half): a collapsed label_map must count DISTINCT class names.

``_parse_tasks`` (``autobench/config.py``) falls back to
``n_classes=len(label_map)`` when a dataset YAML doesn't declare ``n_classes``
explicitly. ``label_map`` maps raw values -> class name (e.g.
``{0: "low", 1: "low", 2: "high"}``, collapsing two raw grades into one
class), so ``len(label_map)`` counts raw KEYS, not distinct class names --
a task whose label_map is many-to-one silently gets an ``n_classes`` larger
than the number of classes that actually appear in the data. That ``n_classes``
flows into ``ExperimentConfig``/model construction (output layer width,
one-hot handling, AUC's ``n_classes`` argument), so a miscount is a real
correctness bug, not a cosmetic one.
"""
from __future__ import annotations

from autobench.config import _parse_tasks


class TestNClassesCountsDistinctNames:
    def test_one_to_one_label_map_unaffected(self):
        """Baseline: no collapsing, so the old and new formulas agree."""
        raw = {
            "brca": {
                "label_col": "brca_label",
                "label_map": {0: "neg", 1: "pos"},
            },
        }
        tasks = _parse_tasks(raw)
        assert tasks["brca"].n_classes == 2

    def test_collapsed_label_map_counts_distinct_names_not_keys(self):
        """Two raw grades collapse into one class name -- n_classes must be 2,
        not 3 (len(label_map))."""
        raw = {
            "grade": {
                "label_col": "grade",
                "label_map": {0: "low", 1: "low", 2: "high"},
            },
        }
        tasks = _parse_tasks(raw)
        assert tasks["grade"].n_classes == 2, (
            f"got {tasks['grade'].n_classes}; a collapsed label_map "
            "(2 raw values -> 'low') must not inflate n_classes to the raw "
            "key count"
        )

    def test_heavily_collapsed_three_class_map(self):
        """4 raw values -> 3 distinct class names."""
        raw = {
            "stage": {
                "label_col": "stage",
                "label_map": {0: "early", 1: "early", 2: "mid", 3: "late"},
            },
        }
        tasks = _parse_tasks(raw)
        assert tasks["stage"].n_classes == 3

    def test_explicit_n_classes_still_wins(self):
        """An explicit YAML n_classes is never overridden by the computed
        fallback -- only the DEFAULT changes."""
        raw = {
            "brca": {
                "label_col": "brca_label",
                "label_map": {0: "neg", 1: "pos"},
                "n_classes": 5,
            },
        }
        tasks = _parse_tasks(raw)
        assert tasks["brca"].n_classes == 5
