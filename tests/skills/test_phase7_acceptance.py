"""D-198 acceptance gate (Phase 7 / STP-01..07).

Single-file aggregator that asserts every clause of D-198 is satisfied. Mirrors
the Phase 6 D-179 / test_phase6_acceptance.py pattern: each clause is a
dedicated test function; passing all 11 means Phase 7 is shippable.

Run: `uv run pytest tests/skills/test_phase7_acceptance.py -v`
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_AGENT_ASSETS = _REPO_ROOT / "src" / "automil" / "agent_assets"
_BACKENDS_DIR = _REPO_ROOT / "src" / "automil" / "backends"
_CLI_DIR = _REPO_ROOT / "src" / "automil" / "cli"


# ---------------------------------------------------------------------------
# Clause 1 -- Backend.healthcheck ABC + 6 LocalBackend unit tests pass
# ---------------------------------------------------------------------------

def test_phase7_acceptance_clause_01_backend_healthcheck_abc_and_6_unit_tests():
    """D-198 clause 1: Backend.healthcheck ABC + 6 LocalBackend tests pass."""
    from automil.backends.base import Backend, HealthReport

    assert "healthcheck" in Backend.__abstractmethods__, (
        "healthcheck is not abstract on Backend"
    )
    assert hasattr(HealthReport, "__dataclass_fields__"), (
        "HealthReport is not a dataclass"
    )
    expected_fields = {
        "gpu_count", "gpu_vram_gb", "accelerator", "python_version",
        "automil_version", "detection_status", "detection_warnings", "detected_at",
    }
    assert set(HealthReport.__dataclass_fields__) == expected_fields, (
        f"HealthReport fields: expected {expected_fields}, "
        f"got {set(HealthReport.__dataclass_fields__)}"
    )

    result = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/backends/test_local_healthcheck.py", "-q", "--tb=line"],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"6 healthcheck tests did not pass:\n{result.stdout[-500:]}\n{result.stderr[-500:]}"
    )


# ---------------------------------------------------------------------------
# Clause 2 -- automil init calls healthcheck; --no-healthcheck flag exists;
#             config.yaml stamps from HealthReport
# ---------------------------------------------------------------------------

def test_phase7_acceptance_clause_02_init_stamps_and_no_healthcheck_flag():
    """D-198 clause 2: automil init --no-healthcheck flag + healthcheck stamping."""
    help_out = subprocess.run(
        [sys.executable, "-m", "automil", "init", "--help"],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    assert help_out.returncode == 0, help_out.stderr
    assert "--no-healthcheck" in help_out.stdout, (
        f"--no-healthcheck flag absent from `automil init --help`:\n{help_out.stdout}"
    )

    result = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/cli/test_init_healthcheck.py", "-q", "--tb=line"],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"init healthcheck tests did not pass:\n{result.stdout[-500:]}"
    )


# ---------------------------------------------------------------------------
# Clause 3 -- failed detection prompts override (STP-03 / D-191)
# ---------------------------------------------------------------------------

def test_phase7_acceptance_clause_03_failed_detection_prompts_override():
    """D-198 clause 3 / STP-03: failed detection + click.confirm declined -> abort.

    The dedicated test lives in tests/cli/test_init_healthcheck.py; clause 2
    already runs that whole file. Here we verify the specific test function
    exists and is named per the D-198 contract.
    """
    test_file = _REPO_ROOT / "tests" / "cli" / "test_init_healthcheck.py"
    assert test_file.exists(), f"test file missing: {test_file}"
    text = test_file.read_text()
    assert "test_init_aborts_on_failed_detection_user_decline" in text, (
        "clause-3 test function `test_init_aborts_on_failed_detection_user_decline` "
        "not found in tests/cli/test_init_healthcheck.py"
    )


# ---------------------------------------------------------------------------
# Clause 4 -- _shared/automil-setup/SKILL.md has D-189..D-196 narrative;
#             _overlay.py rebuild propagates
# ---------------------------------------------------------------------------

def test_phase7_acceptance_clause_04_skill_content_and_overlay_propagation():
    """D-198 clause 4: _shared/automil-setup/SKILL.md narrative + overlay rebuild."""
    shared = _AGENT_ASSETS / "_shared" / "skills" / "automil-setup" / "SKILL.md"
    assert shared.exists(), f"_shared SKILL.md missing: {shared}"
    text = shared.read_text()

    required_sections = [
        "## Architecture",
        "## Steps",
        "## Inspection Heuristics",
        "## Drafting Conventions",
        "## Idempotency Protocol",
        "## Setup-Done Gate",
        "## Failure Modes",
    ]
    for s in required_sections:
        assert s in text, f"section {s!r} missing from _shared SKILL.md"

    result = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/agent_assets/test_overlay_propagation_phase7.py", "-q", "--tb=line"],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"overlay propagation tests did not pass:\n{result.stdout[-500:]}"
    )


# ---------------------------------------------------------------------------
# Clause 5 -- tests/skills/test_setup_idempotency.py: zero unprompted changes
# ---------------------------------------------------------------------------

def test_phase7_acceptance_clause_05_idempotency_zero_unprompted_changes():
    """D-198 clause 5: re-running setup produces zero unprompted file changes."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/skills/test_setup_idempotency.py", "-q", "--tb=line"],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"idempotency tests did not pass:\n{result.stdout[-500:]}"
    )


# ---------------------------------------------------------------------------
# Clause 6 -- tests/skills/test_setup_dry_run_gate.py: known-bad config aborts
# ---------------------------------------------------------------------------

def test_phase7_acceptance_clause_06_setup_done_gate_aborts_on_bad_config():
    """D-198 clause 6 has direct end-to-end coverage in the main test run.

    Do not recursively invoke pytest here: the direct tests launch real daemon
    subprocesses, and nesting them duplicates minutes of work while adding no
    coverage.  This clause pins the named acceptance cases; pytest executes
    their bodies independently in the same suite.
    """
    test_file = _REPO_ROOT / "tests" / "skills" / "test_setup_dry_run_gate.py"
    text = test_file.read_text()
    for name in (
        "test_setup_gate_aborts_on_known_bad_config",
        "test_setup_gate_passes_on_known_good_config",
        "test_setup_gate_polling_terminates_within_90s",
    ):
        assert f"def {name}(" in text, f"missing direct acceptance test: {name}"


# ---------------------------------------------------------------------------
# Clause 7 -- Phase 6 baseline preserved; >=10 new tests added across STP-01..07
# ---------------------------------------------------------------------------

def test_phase7_acceptance_clause_07_baseline_preserved_and_new_tests_added():
    """D-198 clause 7: Phase 6 848-test baseline preserved; >=10 new tests added.

    Phase 6 closed at 848 tests. Phase 7 adds approximately 28 across
    07-01..07-10 + 07-04b. Acceptance threshold: >=858 (baseline + 10 floor).
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    collected_line = [
        l for l in result.stdout.splitlines()
        if "collected" in l and ("test" in l)
    ]
    assert collected_line, (
        f"no collection summary found:\n{result.stdout[-500:]}"
    )
    count = int(collected_line[-1].split()[0])
    assert count >= 858, (
        f"test collection regressed below Phase-6 baseline + 10: "
        f"got {count}, need >=858"
    )


# ---------------------------------------------------------------------------
# Clause 8 -- CHANGELOG entry at 7.0.0 (BREAKING, Backend.healthcheck)
# ---------------------------------------------------------------------------

def test_phase7_acceptance_clause_08_changelog_7_0_0_breaking_entry():
    """D-198 clause 8: CHANGELOG.md has a 7.0.0 BREAKING entry for Backend.healthcheck.

    F-04 lock: heading shape is `## 7.0.0` (no brackets, matching the Phase 6
    entry shape `## 6.0.0`). The grep below anchors to that exact form, NOT an
    either-or.
    """
    changelog = _REPO_ROOT / "CHANGELOG.md"
    assert changelog.exists(), "CHANGELOG.md missing at repo root"
    text = changelog.read_text()

    assert "## 7.0.0" in text, (
        "no `## 7.0.0` entry found (heading shape locked per F-04; "
        "must NOT be `## [7.0.0]`)"
    )
    assert "Backend.healthcheck" in text, (
        "Backend.healthcheck not mentioned in CHANGELOG.md"
    )
    assert "BREAKING" in text or "breaking" in text, (
        "BREAKING change marker absent from CHANGELOG.md"
    )


# ---------------------------------------------------------------------------
# Clause 9 -- automil check passes against a tmp project initialised via
#             `automil init --no-healthcheck`
# ---------------------------------------------------------------------------

def test_phase7_acceptance_clause_09_automil_check_passes_on_workstation(tmp_path):
    """D-198 clause 9: automil check passes on a tmp project via `automil init --no-healthcheck`.

    F-03 fix: the previous version of this test self-skipped when the framework
    repo lacked a root-level `automil/config.yaml` (the framework repo never has
    one). This rewrite constructs a tmp project inside `tmp_path` via the real CLI
    and runs `automil check` against THAT, exercising the real workstation smoke.
    """
    if shutil.which("automil") is None:
        pytest.skip(
            "automil console-script not on PATH; "
            "install via `pip install -e .` first"
        )

    repo = tmp_path / "fake_consumer"
    repo.mkdir()

    # Minimal git scaffold + train.py so `automil init --no-healthcheck` succeeds.
    (repo / "train.py").write_text(
        "if __name__ == '__main__':\n"
        "    import json, pathlib\n"
        "    pathlib.Path('result.json').write_text("
        "json.dumps({'status': 'completed'}))\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=repo, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "test"], cwd=repo, check=True
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=repo, check=True)

    init_proc = subprocess.run(
        ["automil", "init", "--no-healthcheck"],
        cwd=repo, capture_output=True, text=True, timeout=30,
    )
    assert init_proc.returncode == 0, (
        f"automil init failed (rc={init_proc.returncode}):\n"
        f"stdout={init_proc.stdout}\nstderr={init_proc.stderr}"
    )
    assert (repo / "automil" / "config.yaml").exists(), (
        "init did not produce config.yaml"
    )

    # ``check`` now returns non-zero for every must-fix issue. Replace the
    # intentionally invalid template placeholders so this remains a true
    # workstation acceptance test rather than asserting that blocking setup
    # errors exit successfully.
    (repo / "features").mkdir()
    (repo / "splits").mkdir()
    (repo / "mapping.csv").write_text("slide_id,label\n")
    config_path = repo / "automil" / "config.yaml"
    config = yaml.safe_load(config_path.read_text()) or {}
    config["data"] = {
        "features_dir": "features",
        "splits_dir": "splits",
        "mapping_csv": "mapping.csv",
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    check_proc = subprocess.run(
        ["automil", "check"],
        cwd=repo, capture_output=True, text=True, timeout=30,
    )
    assert check_proc.returncode == 0, (
        f"automil check failed against fresh tmp project "
        f"(rc={check_proc.returncode}):\n"
        f"stdout={check_proc.stdout}\nstderr={check_proc.stderr}"
    )


# ---------------------------------------------------------------------------
# Clause 10 -- SLURM/Ray Backend.healthcheck raise NotImplementedError
# ---------------------------------------------------------------------------

def test_phase7_acceptance_clause_10_distributed_backends_raise_notimplemented():
    """D-198 clause 10: SLURM/Ray Backend.healthcheck raise locked NotImplementedError.

    Two sub-verifications:
      a) Source files contain the locked D-189 message string.
      b) Dedicated deferred-contract suite (07-04) passes.
      c) Parametrised contract test in test_contract.py includes healthcheck case.
    """
    locked_msg = "healthcheck deferred to Phase 7+ for distributed backends"
    for fname in ("slurm.py", "ray.py", "mock_slurm.py"):
        f = _BACKENDS_DIR / fname
        if not f.exists():
            continue
        text = f.read_text()
        assert locked_msg in text, (
            f"locked D-189 message missing from {fname}"
        )

    # Dedicated deferred-contract suite (07-04).
    result = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/backends/test_distributed_healthcheck_deferred.py",
         "-q", "--tb=line"],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"distributed deferred-contract tests failed:\n{result.stdout[-500:]}"
    )

    # Parametrised contract test extension (07-04b).
    contract_text = (
        _REPO_ROOT / "tests" / "backends" / "test_contract.py"
    ).read_text()
    assert "test_healthcheck_returns_health_report" in contract_text, (
        "parametrised healthcheck contract test missing from "
        "tests/backends/test_contract.py (see plan 07-04b)"
    )


# ---------------------------------------------------------------------------
# Clause 11 -- Framework purity: zero autobench refs in Phase-7 src files
# ---------------------------------------------------------------------------

def test_phase7_acceptance_clause_11_framework_purity_no_autobench_refs():
    """D-198 clause 11: zero autobench/AUTOBENCH_/benchmarks/ refs in Phase-7 src files.

    Note: config.yaml.j2 owned by Phase 8 framework-purity gate
    (tests/test_framework_purity.py) which has a content-anchored allowlist
    for the autobench-shaped migration comments at lines 105 and 122. Phase 7
    clause 11 only checks files that are exclusively Phase-7-new.
    """
    phase7_src_files = [
        _BACKENDS_DIR / "base.py",
        _BACKENDS_DIR / "local.py",
        _BACKENDS_DIR / "slurm.py",
        _BACKENDS_DIR / "ray.py",
        _BACKENDS_DIR / "mock_slurm.py",
        _CLI_DIR / "init.py",
        _CLI_DIR / "submit.py",
        _AGENT_ASSETS / "_shared" / "skills" / "automil-setup" / "SKILL.md",
        _AGENT_ASSETS / "codex" / "skills" / "automil-setup" / "SKILL.md",
    ]
    bad_terms = ("autobench", "AUTOBENCH_", "benchmarks/")
    for f in phase7_src_files:
        if not f.exists():
            continue
        text = f.read_text()
        for term in bad_terms:
            assert term not in text, (
                f"{f}: contains {term!r} (framework purity violation)"
            )

    # Em-dash gate (Leo's standing memory): no NEW em-dashes / en-dashes in
    # Phase 7 authored content. Files that existed before Phase 7 carry
    # pre-existing em-dashes in docstrings (Phase 2/3/4/5/6 era); those are
    # out of Phase 7 scope. Only scan files whose content is entirely Phase-7-new:
    # the SKILL.md files and the config.yaml template.
    em_dash = "—"  # U+2014 EM DASH
    en_dash = "–"  # U+2013 EN DASH
    _new_in_phase7 = [
        _AGENT_ASSETS / "_shared" / "skills" / "automil-setup" / "SKILL.md",
        _AGENT_ASSETS / "codex" / "skills" / "automil-setup" / "SKILL.md",
    ]
    for f in _new_in_phase7:
        if not f.exists():
            continue
        text = f.read_text()
        assert em_dash not in text and en_dash not in text, (
            f"{f}: contains em-dash or en-dash (feedback_no_em_dashes violation)"
        )
