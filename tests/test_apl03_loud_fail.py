"""APL-03: loud failure for loop-opening variants at apply time.

Tests in this file are RED until Plan 10-02 adds _classify_variant_route to
src/automil/cli/lifecycle/apply.py.

Registry pollution guard: every test calls _clear_registry() via the
autouse fixture, preventing cross-test state pollution (Pitfall 2 in RESEARCH.md).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
import yaml
from click.testing import CliRunner

from automil.registry._state import (
    LOSS_VARIANTS,
    MODEL_VARIANTS,
    POLICY_VARIANTS,
    _clear_registry,
)
from automil.registry.spec import VariantSpec

# Import target — will raise ImportError until Plan 10-02 adds this function.
try:
    from automil.cli.lifecycle.apply import _classify_variant_route
    _CLASSIFY_AVAILABLE = True
except ImportError:
    _classify_variant_route = None  # type: ignore[assignment]
    _CLASSIFY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_registry_after_each():
    """Ensure the registry is clean before AND after each test (Pitfall 2)."""
    _clear_registry()
    yield
    _clear_registry()


def _make_spec(kind: str, name: str, parent: str | None = None) -> VariantSpec:
    return VariantSpec(
        name=name,
        kind=kind,
        parent=parent,
        base_commit="abc123",
        composite=0.85,
        node_id="node_0001",
        created_at="2026-06-11T00:00:00+00:00",
    )


def _register_dummy_loss_variant(name: str = "focal_dummy") -> str:
    """Register a concrete LossVariant subclass into LOSS_VARIANTS and return name."""
    from automil.registry.variants.loss import LossVariant
    from automil.registry.spec import VariantSpec

    class _DummyLoss(LossVariant):
        def __call__(
            self,
            logits: Any,
            targets: Any,
            *,
            instance_logits: Optional[Any] = None,
            instance_labels: Optional[Any] = None,
        ) -> Any:
            raise NotImplementedError("dummy loss — not for real use")

    LOSS_VARIANTS[name] = _DummyLoss
    return name


def _register_dummy_model_variant(
    name: str = "clam_test_v0",
    parent: str = "clam_mb",
) -> tuple[str, str]:
    """Register a concrete ModelVariant subclass into MODEL_VARIANTS and return (parent, name)."""
    from automil.registry.variants.model import ModelVariant

    class _DummyModel(ModelVariant):
        CLAM_ARGS: dict = {"model_size": "big"}

        def forward(self, features: Any, coords: Optional[Any] = None) -> Any:
            raise NotImplementedError("dummy model — not for real use")

    MODEL_VARIANTS[(parent, name)] = _DummyModel
    return parent, name


def _register_dummy_policy_variant(name: str = "sam_dummy") -> str:
    """Register a concrete PolicyVariant subclass into POLICY_VARIANTS and return name."""
    from automil.registry.variants import PolicyVariant

    class _DummyPolicy(PolicyVariant):
        pass  # PolicyVariant has no abstract methods in Phase 10

    POLICY_VARIANTS[name] = _DummyPolicy
    return name


# ---------------------------------------------------------------------------
# Test 1: registered LossVariant raises ClickException at apply time
# RED until Plan 10-02 adds _classify_variant_route.
# ---------------------------------------------------------------------------


def test_loss_variant_raises_click_exception_at_apply_time(tmp_path):
    """A registered LossVariant (custom callable) must raise ClickException at apply time.

    Classification rule (RESEARCH.md §APL-03): a loss variant is a custom
    callable (loop-opening) iff it is in LOSS_VARIANTS as a LossVariant subclass.
    The error message must contain 'requires loop opening' and 'ISSUE-007 / RTA'.
    """
    if not _CLASSIFY_AVAILABLE:
        pytest.fail(
            "APL-03: _classify_variant_route not found in automil.cli.lifecycle.apply. "
            "Implement in Plan 10-02."
        )

    import click

    loss_name = _register_dummy_loss_variant("focal_dummy")
    selection = {
        "model": {"variant": None, "parent": None},
        "loss": {"variant": loss_name},
        "policy": {"variant": None},
    }

    # variants_root can be a tmp dir — the function should check LOSS_VARIANTS
    # (already populated by _register_dummy_loss_variant) without needing to scan.
    variants_root = tmp_path / "variants"
    variants_root.mkdir()

    with pytest.raises(click.ClickException) as exc_info:
        _classify_variant_route(selection, variants_root)

    error_msg = str(exc_info.value.format_message())
    assert "requires loop opening" in error_msg.lower() or "loop" in error_msg.lower(), (
        f"Error message should mention 'requires loop opening'. Got: {error_msg}"
    )
    assert "ISSUE-007" in error_msg or "RTA" in error_msg, (
        f"Error message should reference 'ISSUE-007 / RTA'. Got: {error_msg}"
    )


# ---------------------------------------------------------------------------
# Test 2: model variant does NOT raise
# RED until Plan 10-02 adds _classify_variant_route.
# ---------------------------------------------------------------------------


def test_model_variant_does_not_raise(tmp_path):
    """A registered ModelVariant must NOT raise — it is seam-expressible."""
    if not _CLASSIFY_AVAILABLE:
        pytest.fail(
            "APL-03: _classify_variant_route not found in automil.cli.lifecycle.apply. "
            "Implement in Plan 10-02."
        )

    parent, name = _register_dummy_model_variant()
    selection = {
        "model": {"variant": name, "parent": parent},
        "loss": {"variant": None},
        "policy": {"variant": None},
    }

    variants_root = tmp_path / "variants"
    variants_root.mkdir()

    # Should not raise — model variants are always seam-expressible.
    _classify_variant_route(selection, variants_root)


# ---------------------------------------------------------------------------
# Test 3: string selector (bag_loss override) does NOT raise
# RED until Plan 10-02 adds _classify_variant_route.
# ---------------------------------------------------------------------------


def test_string_selector_loss_does_not_raise(tmp_path):
    """A string-selector bag_loss value ("svm") is NOT a LossVariant subclass → no raise.

    Pitfall 5 (RESEARCH.md): a ModelVariant with CLAM_ARGS={"bag_loss": "svm"}
    uses a string selector expressible through the open seam — must NOT be
    classified as loop-opening.  The key distinction: the variant name is not
    in LOSS_VARIANTS.
    """
    if not _CLASSIFY_AVAILABLE:
        pytest.fail(
            "APL-03: _classify_variant_route not found in automil.cli.lifecycle.apply. "
            "Implement in Plan 10-02."
        )

    # "svm" is a string selector — NOT registered as a LossVariant subclass.
    # LOSS_VARIANTS is empty (cleared by autouse fixture).
    selection = {
        "model": {"variant": None, "parent": None},
        "loss": {"variant": "svm"},
        "policy": {"variant": None},
    }

    variants_root = tmp_path / "variants"
    variants_root.mkdir()

    # Should NOT raise — "svm" is not in LOSS_VARIANTS (no registered subclass).
    _classify_variant_route(selection, variants_root)


# ---------------------------------------------------------------------------
# Test 4: error raised BEFORE config.yaml is mutated (ordering integrity)
# RED until Plan 10-02 adds _classify_variant_route AND inserts it in apply().
# ---------------------------------------------------------------------------


def test_error_raised_before_config_mutation(tmp_path, monkeypatch):
    """APL-03 ordering: ClickException fires BEFORE config.yaml is written.

    This is Pitfall 3 from RESEARCH.md: the route classifier MUST fire before
    any config mutation.  Order: derive selection → classify route (raise if
    loop-opening) → backup → write.

    Strategy: monkeypatch _atomic_write_text to record if it was ever called.
    If ClickException is raised AND _atomic_write_text was NOT called, ordering
    is correct.
    """
    if not _CLASSIFY_AVAILABLE:
        pytest.fail(
            "APL-03: _classify_variant_route not found in automil.cli.lifecycle.apply. "
            "Implement in Plan 10-02."
        )

    import subprocess as _sp

    # Set up a full automil project so we can invoke 'automil apply' via CLI runner.
    _sp.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    _sp.run(["git", "config", "user.name", "test"], cwd=tmp_path, capture_output=True)
    _sp.run(["git", "config", "user.email", "t@t.com"], cwd=tmp_path, capture_output=True)
    (tmp_path / "README.md").write_text("# test\n")
    _sp.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
    _sp.run(["git", "commit", "-m", "init"], cwd=tmp_path, capture_output=True, check=True)

    monkeypatch.chdir(tmp_path)
    from automil.cli import main as automil_main
    runner = CliRunner()
    runner.invoke(automil_main, ["init"])

    adir = tmp_path / "automil"
    # Write graph.json with a loss-only node.
    loss_name = _register_dummy_loss_variant("focal_ordering_test")
    graph = {
        "schema_version": 1,
        "meta": {
            "best_node_id": None, "best_composite": 0.0,
            "total_executed": 0, "total_proposed": 0,
            "next_id": 1, "baseline_composite": 0.0,
            "scoring": {"exploration_weight": 0.005, "novelty_weight": 0.003},
        },
        "nodes": {
            "node_0001": {
                "id": "node_0001",
                "type": "executed",
                "status": "keep",
                "composite": 0.5,
                "variant_spec": {"kind": "loss", "name": loss_name, "parent": None},
            }
        },
        "technique_stats": {},
    }
    (adir / "graph.json").write_text(json.dumps(graph, indent=2))

    # Track whether _atomic_write_text was called.
    write_called: list[bool] = []

    from automil.cli.lifecycle import _shared
    original_write = _shared._atomic_write_text

    def _spy_write(path: Path, text: str) -> None:
        write_called.append(True)
        original_write(path, text)

    monkeypatch.setattr(_shared, "_atomic_write_text", _spy_write)

    result = runner.invoke(automil_main, ["apply", "node_0001"])

    # The apply command should have exited non-zero (ClickException).
    assert result.exit_code != 0, (
        "Expected apply to fail with ClickException for loop-opening loss variant. "
        f"Got exit_code={result.exit_code}, output={result.output}"
    )
    # _atomic_write_text must NOT have been called (error before mutation).
    assert not write_called, (
        "APL-03 ordering violated: config.yaml was written before the loop-opening "
        "ClickException was raised. Move _classify_variant_route call before the "
        "backup+write step in apply.py."
    )
