"""Phase 8 D-208 acceptance gate, single load-bearing test file.

Each test maps to exactly one of the 11 clauses in D-208. Failing ANY clause
fails Phase 8. Mirrors Phase 6 D-179 and Phase 7 D-198 single-file aggregators.

Run as the final gate:
    uv run pytest tests/acceptance/test_phase8_acceptance.py -v

Iter-2 / F-09 fix: clause 11 was previously a tortured assertion that always
passed (Python operator precedence + REQUIREMENTS.md presence collapsed the
expression to a trivially-true value). Replaced with deterministic content
checks against CHANGELOG.md (produced by Task 2 of this plan) and
REQUIREMENTS.md DEC-XX traceability rows (produced by Task 4).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC_AUTOMIL = _REPO_ROOT / "src" / "automil"


def _pytest(target: str, extra_args: list[str] | None = None) -> subprocess.CompletedProcess:
    """Run pytest against a target path; capture stdout/stderr."""
    cmd = [sys.executable, "-m", "pytest", target, "-q", "--tb=line"]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(
        cmd, cwd=_REPO_ROOT, capture_output=True, text=True,
    )


# ---------------------------------------------------------------------------
# Clause 1: framework purity grep returns zero non-allowlisted hits (D-201)
# ---------------------------------------------------------------------------

def test_d208_clause_01_framework_purity():
    """D-208 #1: zero autobench/AUTOBENCH_/benchmarks/ non-allowlisted in src/automil/."""
    out = _pytest("tests/test_framework_purity.py")
    assert out.returncode == 0, (
        f"D-208 clause 1 (framework purity) failed:\n"
        f"{out.stdout[-1500:]}\n{out.stderr[-500:]}"
    )


# ---------------------------------------------------------------------------
# Clause 2: result.json schema-validated at ingestion (DEC-03)
# ---------------------------------------------------------------------------

def test_d208_clause_02_result_schema_validation():
    """D-208 #2: schemas/result.schema.json exists; daemon validates at ingestion."""
    schema_path = _SRC_AUTOMIL / "schemas" / "result.schema.json"
    assert schema_path.exists(), "src/automil/schemas/result.schema.json missing"
    schema = json.loads(schema_path.read_text())
    assert schema["required"] == ["composite"]
    assert schema["additionalProperties"] is True

    # Daemon ingest path validates.
    daemon_src = (_SRC_AUTOMIL / "backends"
                  / "_orchestrator_daemon.py").read_text()
    assert "from automil.schemas import validate_result" in daemon_src
    assert "see automil/schemas/result.schema.json" in daemon_src

    # Schema-validation tests pass.
    out = _pytest("tests/test_result_schema_validation.py")
    assert out.returncode == 0, (
        f"schema validation tests failed:\n{out.stdout[-500:]}"
    )


# ---------------------------------------------------------------------------
# Clause 3: graph.py dict-spread storage (DEC-04)
# ---------------------------------------------------------------------------

def test_d208_clause_03_graph_dict_spread():
    """D-208 #3: graph.py stores opaque consumer metrics; composite-only Pareto."""
    graph_src = (_SRC_AUTOMIL / "graph.py").read_text()
    # ``composite_se`` became a framework-owned top-level scalar after D-200,
    # so the write path now filters that key instead of blindly copying the
    # entire mapping.  Keep this acceptance gate on the contract, not one stale
    # local-variable spelling.
    assert '"metrics": {k: v for k, v in metrics.items()' in graph_src
    assert 'node["metrics"] = {k: v for k, v in metrics.items()' in graph_src
    assert '"metrics": dict(comp_metrics)' in graph_src
    assert "composite-only dominance" in graph_src

    # Dict-spread tests pass.
    out = _pytest("tests/test_graph_dict_spread.py")
    assert out.returncode == 0, (
        f"graph dict-spread tests failed:\n{out.stdout[-500:]}"
    )


# ---------------------------------------------------------------------------
# Clause 4: env.required validator + config.yaml.j2 schema (DEC-05)
# ---------------------------------------------------------------------------

def test_d208_clause_04_env_required_validator():
    """D-208 #4: env.required validated by automil check; config template extended."""
    check_src = (_SRC_AUTOMIL / "cli" / "check.py").read_text()
    assert "_validate_env_required" in check_src
    assert "Missing required env var:" in check_src

    template_src = (_SRC_AUTOMIL / "templates" / "config.yaml.j2").read_text()
    assert "required: []" in template_src
    assert "passthrough:" in template_src
    # Iter-2 / F-07: scoring block surfaced in template.
    assert "scoring:" in template_src
    assert "formula:" in template_src

    out = _pytest("tests/cli/test_check_env_required.py")
    assert out.returncode == 0, (
        f"env.required validator tests failed:\n{out.stdout[-500:]}"
    )


# ---------------------------------------------------------------------------
# Clause 5: sklearn-iris consumer exists end-to-end (DEC-02)
# ---------------------------------------------------------------------------

def test_d208_clause_05_sklearn_iris_consumer_exists():
    """D-208 #5: examples/sklearn-iris/ ships with train.py + automil/ scaffolding."""
    iris = _REPO_ROOT / "examples" / "sklearn-iris"
    assert iris.exists(), "examples/sklearn-iris/ missing"
    assert (iris / "train.py").exists()
    assert (iris / "automil" / "config.yaml").exists()
    assert (iris / "automil" / "program.md").exists()
    assert (iris / "automil" / "variants" / "classifier_v0" / "logistic_v0.py").exists()
    assert (iris / "README.md").exists()

    # Consumer is decoupled (no automil.* imports).
    train_src = (iris / "train.py").read_text()
    assert "from automil" not in train_src and "import automil" not in train_src


# ---------------------------------------------------------------------------
# Clause 6: training-script contract doc covers 6 contract items (DEC-06)
# ---------------------------------------------------------------------------

def test_d208_clause_06_training_script_contract_doc():
    """D-208 #6: docs/training-script-contract.md covers all 6 D-204 items."""
    doc = _REPO_ROOT / "docs" / "training-script-contract.md"
    assert doc.exists(), "docs/training-script-contract.md missing"

    out = _pytest("tests/test_phase8_docs_exist.py")
    assert out.returncode == 0, (
        f"docs-exist tests failed:\n{out.stdout[-500:]}"
    )


# ---------------------------------------------------------------------------
# Clause 7: framework purity grep test PASSES with hardcoded allowlist (D-206)
# ---------------------------------------------------------------------------

def test_d208_clause_07_framework_purity_grep_gate():
    """D-208 #7: tests/test_framework_purity.py PASSES with hardcoded allowlist."""
    test_file = _REPO_ROOT / "tests" / "test_framework_purity.py"
    assert test_file.exists()
    test_src = test_file.read_text()
    assert "_ALLOWLIST" in test_src
    # Stable-text anchor (DBT-03 / 14-02): file is still tracked in allowlist
    # and the content anchor string is present. Decoupled from line-number drift.
    assert "_orchestrator_daemon.py:" in test_src  # file is tracked in allowlist
    assert "Consumer-specific vars (e.g. AUTOBENCH_*_ROOT)" in test_src  # content anchor
    assert "src/automil/cli/lifecycle/verify_repro.py:84" in test_src
    # Iter-2 / F-01 fix: revert_baseline.py:87 default-help allowlist entry.
    assert "src/automil/cli/lifecycle/revert_baseline.py:87" in test_src

    out = _pytest("tests/test_framework_purity.py")
    assert out.returncode == 0, (
        f"framework purity grep gate failed:\n{out.stdout[-500:]}"
    )


# ---------------------------------------------------------------------------
# Clause 8: final acceptance gate sub-gate B passes in CI (D-205)
# ---------------------------------------------------------------------------

def test_d208_clause_08_final_acceptance_gate():
    """D-208 #8: sub-gate B (sklearn-iris end-to-end via real orchestrator) PASSES; A+C skip cleanly."""
    out = _pytest(
        "tests/acceptance/test_final_phase8_acceptance.py::test_subgate_b_sklearn_iris_end_to_end"
    )
    assert out.returncode == 0, (
        f"sub-gate B (sklearn-iris end-to-end) failed:\n"
        f"{out.stdout[-1500:]}\n{out.stderr[-500:]}"
    )

    # Sub-gates A and C skip cleanly (no AUTOBENCH_CCRCC_ROOT in CI).
    out_all = _pytest(
        "tests/acceptance/test_final_phase8_acceptance.py", extra_args=["-v"]
    )
    text = out_all.stdout + out_all.stderr
    # Either PASSED (with data) or SKIPPED (without data); never FAILED.
    assert "FAILED" not in text, (
        f"sub-gates have failures:\n{text[-1500:]}"
    )


# ---------------------------------------------------------------------------
# Clause 9: Phase 7 baseline preserved + >=10 new tests (D-208 numbered)
# ---------------------------------------------------------------------------

def test_d208_clause_09_baseline_plus_10_tests():
    """D-208 #9: Phase 7 baseline (838+ tests) preserved; >=10 new added."""
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    text = out.stdout + out.stderr
    collected = 0
    for line in text.splitlines():
        if "collected" in line and "test" in line:
            for tok in line.split():
                if tok.isdigit():
                    collected = max(collected, int(tok))
    # Phase 7 close was 848+; Phase 8 adds at least 10. Floor: 858.
    assert collected >= 858, (
        f"D-208 #9: expected >=858 collected tests "
        f"(848 Phase 7 baseline + 10 Phase 8 minimum); got {collected}"
    )


# ---------------------------------------------------------------------------
# Clause 10: CHANGELOG entry at 8.0.0 (BREAKING)
# ---------------------------------------------------------------------------

def test_d208_clause_10_changelog_8_0_0():
    """D-208 #10: CHANGELOG.md has ## 8.0.0 entry marking BREAKING decoupling."""
    changelog = _REPO_ROOT / "CHANGELOG.md"
    assert changelog.exists()
    text = changelog.read_text()
    assert "## 8.0.0" in text, "CHANGELOG.md missing ## 8.0.0 heading"
    assert "BREAKING" in text, "CHANGELOG.md 8.0.0 must include BREAKING marker"
    # env.required becoming mandatory + node['metrics'] migration both BREAKING.
    assert "env.required" in text or "env: required" in text
    assert "node[\"metrics\"]" in text or "node['metrics']" in text or "metrics dict" in text


# ---------------------------------------------------------------------------
# Clause 11: ROADMAP + STATE + REQUIREMENTS reflect Phase 8 + v1.0 complete
# Iter-2 / F-09 fix: anchor on CHANGELOG.md content (Task 2 produces it) +
# REQUIREMENTS.md DEC-XX rows (Task 4 produces them). NOT ROADMAP/STATE which
# are circular self-references for this plan.
# ---------------------------------------------------------------------------

def test_d208_clause_11_state_roadmap_complete():
    """D-208 #11: cross-doc consistency at milestone (F-09 fix; non-circular).

    Anchors:
      - CHANGELOG.md head section heading is `## 8.0.0` AND contains
        F-06 migration note text (env.required + AUTOBENCH_OVARIAN_ROOT
        explicit example, since CHANGELOG is the recovery surface).
      - REQUIREMENTS.md traceability table has DEC-01..DEC-07 marked
        Complete (NOT Pending).

    These are deterministic content checks. The previous clause-11 logic
    asserted ROADMAP/STATE were updated (which 08-10 itself does), creating
    circular validation. F-09 anchors on CHANGELOG (Task 2) +
    REQUIREMENTS.md (Task 4), both of which are independently auditable
    artifacts of this plan with deterministic post-conditions.
    """
    # 1. CHANGELOG head section anchor.
    changelog_path = _REPO_ROOT / "CHANGELOG.md"
    assert changelog_path.exists(), "CHANGELOG.md missing"
    changelog_text = changelog_path.read_text()

    # Post-v1.0 semver releases and an Unreleased section now correctly precede
    # the historical 8.0.0 milestone.  Validate the milestone's own section
    # instead of requiring it to remain the newest changelog entry forever.
    milestone_match = re.search(
        r"^## 8\.0\.0\b.*?(?=^## |\Z)",
        changelog_text,
        re.MULTILINE | re.DOTALL,
    )
    assert milestone_match is not None, "CHANGELOG.md has no 8.0.0 milestone section"
    milestone_text = milestone_match.group(0)

    # F-06 migration note: explicit AUTOBENCH-shaped operator-recovery snippet.
    # The CHANGELOG (NOT the framework template) is where the 4-cell matrix
    # (env.required vs env.passthrough x example values vs sentinel) is
    # resolved with concrete autobench-shaped examples for migration.
    assert "AUTOBENCH_OVARIAN_ROOT" in milestone_text, (
        "CHANGELOG.md 8.0.0 BREAKING section must include the F-06 migration "
        "recovery snippet with explicit AUTOBENCH_OVARIAN_ROOT example. "
        "(Framework template config.yaml.j2 stays framework-pure per F-06; "
        "CHANGELOG is the migration surface.)"
    )
    assert "AUTOBENCH_CCRCC_ROOT" in milestone_text, (
        "CHANGELOG.md 8.0.0 BREAKING section must include AUTOBENCH_CCRCC_ROOT "
        "example for autobench-shaped consumers per F-06 4-cell resolution."
    )
    assert "env.required" in milestone_text or "env:\n  required:" in milestone_text
    assert "passthrough" in milestone_text

    # 2. REQUIREMENTS.md DEC-XX rows transitioned Pending -> Complete.
    # After milestone close, REQUIREMENTS.md is archived to milestones/v1.0-REQUIREMENTS.md
    # per the standard /gsd-complete-milestone workflow. The archive takes precedence
    # when it exists (post-v1.0-close state); fall back to live path when absent.
    archive_path = _REPO_ROOT / ".planning" / "milestones" / "v1.0-REQUIREMENTS.md"
    live_path = _REPO_ROOT / ".planning" / "REQUIREMENTS.md"
    # v1.0 acceptance gate must validate the v1.0 record.
    # Archive takes precedence when it exists (post-v1.0-close state).
    if archive_path.exists():
        req_path = archive_path
    elif live_path.exists():
        req_path = live_path
    else:
        req_path = live_path  # triggers assert below
    assert req_path.exists(), (
        "Neither .planning/milestones/v1.0-REQUIREMENTS.md nor .planning/REQUIREMENTS.md found"
    )
    req_text = req_path.read_text()

    for dec_id in ("DEC-01", "DEC-02", "DEC-03", "DEC-04",
                   "DEC-05", "DEC-06", "DEC-07"):
        pending_row = f"| {dec_id} | Phase 8 | Pending |"
        complete_row = f"| {dec_id} | Phase 8 | Complete |"
        assert pending_row not in req_text, (
            f"{dec_id} still marked Pending in REQUIREMENTS.md; Task 4 must "
            f"transition it to Complete at milestone close."
        )
        assert complete_row in req_text, (
            f"{dec_id} not marked Complete in REQUIREMENTS.md; expected the "
            f"row '{complete_row}' (Task 4 / F-09 / F-10 fix)."
        )
