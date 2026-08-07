"""DEC-05 / D-202: tests for _validate_env_required helper in src/automil/cli/check.py.

Pure function under test; uses monkeypatch to manipulate os.environ.

Iter-2 / F-05 adds test_env_required_non_list_warns_and_skips_validation
which exercises the call-site warning emission path (CliRunner-based;
asserts the operator sees the type-mismatch warning rather than a silent
no-op).

Iter-2 / F-07 adds test_template_has_scoring_block which regression-prevents
the DEC-04 `automil/config.yaml: scoring.formula` field from disappearing
from the framework template.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from automil.cli.check import _validate_env_required


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_empty_required_list_returns_no_missing(monkeypatch):
    """D-202: empty list = nothing to validate."""
    assert _validate_env_required({"env": {"required": []}}) == []


def test_missing_var_in_required_returns_name(monkeypatch):
    """D-202: missing var name surfaces in the return list."""
    monkeypatch.delenv("_AUTOMIL_TEST_MISSING_VAR_QQ", raising=False)
    result = _validate_env_required(
        {"env": {"required": ["_AUTOMIL_TEST_MISSING_VAR_QQ"]}}
    )
    assert result == ["_AUTOMIL_TEST_MISSING_VAR_QQ"]


def test_present_var_in_required_returns_empty(monkeypatch):
    """D-202: vars set in os.environ count as present."""
    monkeypatch.setenv("_AUTOMIL_TEST_PRESENT_VAR", "value")
    result = _validate_env_required(
        {"env": {"required": ["_AUTOMIL_TEST_PRESENT_VAR"]}}
    )
    assert result == []


def test_multiple_missing_vars_returns_all_in_order(monkeypatch):
    """D-202: per-name iteration; preserves declaration order."""
    monkeypatch.setenv("_AUTOMIL_TEST_PRESENT_A", "1")
    monkeypatch.delenv("_AUTOMIL_TEST_MISSING_B", raising=False)
    monkeypatch.delenv("_AUTOMIL_TEST_MISSING_C", raising=False)
    result = _validate_env_required(
        {"env": {"required": [
            "_AUTOMIL_TEST_PRESENT_A",
            "_AUTOMIL_TEST_MISSING_B",
            "_AUTOMIL_TEST_MISSING_C",
        ]}}
    )
    assert result == ["_AUTOMIL_TEST_MISSING_B", "_AUTOMIL_TEST_MISSING_C"]


def test_required_not_a_list_returns_empty():
    """D-202: type-mismatch is surfaced as a warning at the call site, not here."""
    assert _validate_env_required({"env": {"required": "not a list"}}) == []
    assert _validate_env_required({"env": {"required": 42}}) == []
    assert _validate_env_required({"env": {"required": None}}) == []


def test_no_env_section_returns_empty():
    """D-202: missing env block = no required vars to validate."""
    assert _validate_env_required({}) == []
    assert _validate_env_required({"env": None}) == []


def test_present_var_with_empty_string_counts_as_present(monkeypatch):
    """CONTEXT anti-pattern #6: env vars are presence-only, not value-based."""
    monkeypatch.setenv("_AUTOMIL_TEST_EMPTY_VAR", "")
    result = _validate_env_required(
        {"env": {"required": ["_AUTOMIL_TEST_EMPTY_VAR"]}}
    )
    assert result == []


def test_present_var_with_todo_sentinel_counts_as_present(monkeypatch):
    """CONTEXT anti-pattern #6: TODO_FILL_IN is YAML-config-only, not env vars."""
    monkeypatch.setenv("_AUTOMIL_TEST_SENTINEL_VAR", "TODO_FILL_IN")
    result = _validate_env_required(
        {"env": {"required": ["_AUTOMIL_TEST_SENTINEL_VAR"]}}
    )
    assert result == []


def test_env_required_non_list_warns_and_skips_validation(tmp_path: Path, monkeypatch):
    """Iter-2 / F-05: type-mismatch surfaces as an operator-visible warning at the call site.

    When env.required is a STRING (e.g. "AUTOBENCH_OVARIAN_ROOT") instead of
    a list, `automil check` must:
      1. Emit a warning naming the type mismatch (operator-visible).
      2. Skip env.required validation (no spurious "Missing required env var").
      3. NOT crash (return cleanly).

    Strategy: invoke automil check via CliRunner against a tmp project whose
    automil/config.yaml has a string-shaped env.required value; capture
    stdout/stderr; assert the warning text and absence of crash.
    """
    from automil.cli import main as cli

    # Build a minimal automil project with a malformed config.
    project = tmp_path / "proj"
    automil_dir = project / "automil"
    automil_dir.mkdir(parents=True)
    for subdir in ["queue", "running", "archive", "completed"]:
        (automil_dir / "orchestrator" / subdir).mkdir(parents=True)
    bad_cfg = {
        "orchestrator": {"backend": "local"},
        "env": {
            # Type mismatch: should be a list, supplied as a string.
            "required": "AUTOBENCH_OVARIAN_ROOT",
            "passthrough": ["AUTOMIL_RUNTIME"],
        },
    }
    (automil_dir / "config.yaml").write_text(yaml.safe_dump(bad_cfg))
    (project / ".git").mkdir()
    # Minimal HEAD file so git commands do not completely fail.
    (project / ".git" / "HEAD").write_text("ref: refs/heads/main\n")

    # Invoke automil check.
    runner = CliRunner()
    monkeypatch.chdir(project)
    result = runner.invoke(cli, ["check"], catch_exceptions=True)

    combined = (result.output or "")

    # 1. Warning must be operator-visible. The locked phrase is the
    #    type-mismatch warning emitted by the check() command body.
    assert "env.required must be a list of var names" in combined, (
        f"Operator-visible type-mismatch warning missing.\n"
        f"Output:\n{combined}\n"
    )

    # 2. No spurious "Missing required env var: AUTOBENCH_OVARIAN_ROOT" because
    #    validation was skipped.
    assert "Missing required env var: AUTOBENCH_OVARIAN_ROOT" not in combined, (
        f"Validation must be skipped on type mismatch; got:\n{combined}"
    )

    # 3. Did not crash with an unhandled Python exception.
    assert "Traceback" not in combined, (
        f"automil check crashed on malformed env.required:\n{combined}"
    )


def test_template_has_scoring_block():
    """Iter-2 / F-07: framework template surfaces DEC-04 scoring.formula contract.

    REQUIREMENTS.md DEC-04 + ROADMAP Phase 8 success criterion 3 name
    `automil/config.yaml: scoring.formula` verbatim. The framework template
    config.yaml.j2 MUST surface this field so fresh `automil init` outputs
    document the composite-scoring contract for new consumers.
    """
    template = (_REPO_ROOT / "src" / "automil" / "templates"
                / "config.yaml.j2").read_text()
    assert "scoring:" in template, (
        "config.yaml.j2 missing top-level `scoring:` block; DEC-04 contract "
        "not surfaced for new consumers (F-07 fix)."
    )
    assert "formula:" in template, (
        "config.yaml.j2 `scoring:` block missing `formula:` field; "
        "ROADMAP Phase 8 success criterion 3 names this verbatim (F-07 fix)."
    )
    # B2 (claims-alignment): the D-200 "documentation-only" contract is over —
    # CR-1b treats scoring.formula as the reducer used to recompute the
    # composite from validation metrics, and an unknown value fails
    # `automil check` / graph seeding. The comment block must teach the
    # reducer vocabulary and the explicit opt-out so operators cannot follow
    # a stale example into silently disabling the leak detector.
    assert "reducer" in template and "trust_reported" in template, (
        "config.yaml.j2 scoring block must document the CR-1b reducer "
        "semantics (mean/max/min) and the trust_reported opt-out (B2)."
    )
