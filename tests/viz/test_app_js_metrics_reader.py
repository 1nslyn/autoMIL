"""DEC-04 / D-200: viz/static/app.js metric-reader migration regression test.

The dashboard JS reads node-displayed metrics. After D-200, those metrics live
under node.metrics; this file asserts the post-migration access pattern is
present and the pre-migration pattern is absent.

The JS file is not executed in this test; we assert via static text patterns.
End-to-end browser tests (Playwright/etc.) are out of scope for v1.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_APP_JS = _REPO_ROOT / "src" / "automil" / "viz" / "static" / "app.js"


@pytest.fixture
def app_js_text() -> str:
    """Return app.js source text (skip if file is absent)."""
    if not _APP_JS.exists():
        pytest.skip(f"app.js not found at {_APP_JS}")
    return _APP_JS.read_text()


def test_app_js_reads_node_metrics_post_d200(app_js_text: str):
    """D-200: viz reads node.metrics, not top-level node[key]."""
    assert "(node.metrics || {})[pair[0]]" in app_js_text, (
        "Expected the post-D-200 access pattern '(node.metrics || {})[pair[0]]' "
        "in app.js. The viz dashboard must read consumer metrics from "
        "node.metrics after Phase 8 graph.py dict-spread refactor."
    )


def test_app_js_pre_d200_pattern_absent(app_js_text: str):
    """Regression: the pre-D-200 access pattern is removed."""
    # The exact pre-migration line.
    forbidden = "var val = node[pair[0]];"
    assert forbidden not in app_js_text, (
        f"Found pre-D-200 access pattern {forbidden!r} in app.js. "
        "This pattern reads top-level node[key] which no longer carries "
        "consumer metrics after D-200; it must be replaced by "
        "'var val = (node.metrics || {})[pair[0]];'."
    )


def test_app_js_metric_fields_val_only(app_js_text: str):
    """Val-firewall: the dashboard surfaces validation metrics only; test is
    sealed (certify.json) and must never appear in the viz during search."""
    # Validation metrics ARE shown (the selection signal the tree optimizes).
    for label_pair in (
        "['val_auc', 'Val AUC']",
        "['val_bacc', 'Val BACC']",
    ):
        assert label_pair in app_js_text, (
            f"metricFields entry {label_pair!r} missing from app.js. The viz "
            "must surface the validation metrics that drive keep/discard."
        )
    # Test metrics must NOT be surfaced (val-firewall quarantine).
    for forbidden in ("'Test AUC'", "'Test BACC'", "['test_auc'", "['test_bacc'"):
        assert forbidden not in app_js_text, (
            f"Found test metric {forbidden!r} in app.js metricFields. Under the "
            "val-firewall, test is sealed in certify.json and must not appear "
            "in the dashboard during search."
        )
