"""DEC-06 / D-204: docs/training-script-contract.md regression test.

Asserts the document exists and covers all 6 contract items + cross-links
+ pitfalls. Anchor strings are coordinated with plan 08-07 Task 1; do NOT
edit these without updating the document.
"""
from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DOC = _REPO_ROOT / "docs" / "training-script-contract.md"


@pytest.fixture
def doc_text() -> str:
    """Return the doc text or fail (doc must exist)."""
    assert _DOC.exists(), f"docs/training-script-contract.md missing at {_DOC}"
    return _DOC.read_text()


def test_doc_exists():
    """DEC-06: docs/training-script-contract.md must exist at repo root."""
    assert _DOC.exists()


def test_doc_covers_six_contract_items(doc_text: str):
    """D-204: all 6 contract items present with the anchor phrasing."""
    anchors = [
        "Read `automil/config.yaml`",
            "Honor the framework-owned device masks",
            "Branch on `AUTOMIL_ACCELERATOR`",
        "Exit cleanly on `SIGTERM`",
        "Write `result.json`",
        "Declared env vars are present at startup",
    ]
    for anchor in anchors:
        assert anchor in doc_text, f"missing contract anchor: {anchor!r}"


def test_doc_cross_links_examples_and_schema(doc_text: str):
    """DEC-06: doc cross-links sklearn-iris reference and the schema file."""
    assert "examples/sklearn-iris/train.py" in doc_text
    assert "automil/schemas/result.schema.json" in doc_text


def test_doc_documents_both_sigterm_patterns(doc_text: str):
    """D-204: doc covers BOTH multi-fold and single-shot SIGTERM patterns."""
    assert "register_sigterm_flush" in doc_text  # Pattern A
    assert "signal.signal" in doc_text           # Pattern B


def test_doc_documents_two_named_pitfalls(doc_text: str):
    """D-204: 2 named pitfalls (cleanup-after-write, sys.exit-without-write)."""
    assert "Common pitfalls" in doc_text
    assert "Writing result.json AFTER cleanup" in doc_text
    assert "sys.exit(0)" in doc_text


def test_doc_references_env_required(doc_text: str):
    """DEC-06 cross-link: doc references env.required + automil check."""
    assert "env.required" in doc_text
    assert "automil check" in doc_text


def test_doc_no_em_or_en_dashes(doc_text: str):
    """Leo memory feedback_no_em_dashes: prose uses periods/commas/and."""
    # Em dash U+2014, en dash U+2013
    assert "—" not in doc_text, "em dash (U+2014) found in docs/training-script-contract.md"
    assert "–" not in doc_text, "en dash (U+2013) found in docs/training-script-contract.md"


def test_doc_minimum_length(doc_text: str):
    """Documentation completeness floor: >=120 lines."""
    line_count = len(doc_text.splitlines())
    assert line_count >= 120, f"doc has only {line_count} lines; expected >=120"
