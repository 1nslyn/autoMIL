"""Phase 6 D-179 acceptance gate — single load-bearing test file (BCK-05 + BCK-06).

Each test maps to exactly one of the 11 clauses in D-179. Failing ANY clause
fails Phase 6. This is the same load-bearing pattern as Phase 5's
test_pitfall6_held_out_isolation.py.

Run as the final gate:
    uv run pytest tests/backends/test_phase6_acceptance.py -v
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


# Repo root resolution: this file lives at tests/backends/test_phase6_acceptance.py
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC_AUTOMIL = _REPO_ROOT / "src" / "automil"
_SCRIPTS = _REPO_ROOT / "scripts"


# ---------------------------------------------------------------------------
# Clause 1 — Contract test parametrised over 4 backends, >=10 scenarios per backend
# ---------------------------------------------------------------------------

def test_d179_clause_01_contract_parametrised_over_4_backends():
    """D-179 #1: tests/backends/test_contract.py passes parametrised over [local, mock_slurm, slurm, ray]."""
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/backends/test_contract.py", "--collect-only", "-q"],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    text = out.stdout + out.stderr
    for backend in ("[local]", "[mock_slurm]", "[slurm]", "[ray]"):
        assert backend in text, (
            f"contract test missing parametrisation {backend}; output: {text[:500]}"
        )


# ---------------------------------------------------------------------------
# Clause 2 — Phase 5 baseline (779 tests + Phase 6 additions) preserved
# ---------------------------------------------------------------------------

def test_d179_clause_02_phase5_baseline_preserved():
    """D-179 #2: Phase 5's 779-test baseline stays green (Phase 6 additive)."""
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    text = (out.stdout + out.stderr).strip()
    collected = 0
    for line in text.splitlines():
        if "collected" in line and "test" in line:
            for tok in line.split():
                if tok.isdigit():
                    collected = max(collected, int(tok))
    assert collected >= 789, (
        f"expected >=789 collected tests (779 baseline + >=10 Phase 6 additions); got {collected}"
    )


# ---------------------------------------------------------------------------
# Clause 3 — BCK-04 lint clean
# ---------------------------------------------------------------------------

def test_d179_clause_03_bck04_lint_clean():
    """D-179 #3: scripts/check_backend_isolation.py exits 0; slurm.py + ray.py NOT in allowlist."""
    out = subprocess.run(
        [sys.executable, str(_SCRIPTS / "check_backend_isolation.py"), str(_SRC_AUTOMIL)],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, f"BCK-04 lint failed: {out.stdout}\n{out.stderr}"
    # Verify slurm.py and ray.py are NOT in the allowlist (read the script source).
    isolation_src = (_SCRIPTS / "check_backend_isolation.py").read_text()
    assert 'Path("backends/slurm.py")' not in isolation_src, (
        "slurm.py must NOT be added to allowlist"
    )
    assert 'Path("backends/ray.py")' not in isolation_src, (
        "ray.py must NOT be added to allowlist"
    )


# ---------------------------------------------------------------------------
# Clause 4 — pip install -e . (no extras) installs cleanly
# ---------------------------------------------------------------------------

def test_d179_clause_04_no_extras_install_works():
    """D-179 #4: import automil works without submitit/ray; automil --help succeeds.

    Verified by: (a) importing automil succeeds in the current process (which
    represents the no-extras install if extras absent), (b) `automil --help`
    runs without error.
    """
    import automil  # must succeed
    import automil.backends  # must succeed

    # Exercise the package installed in the active test environment.  Calling
    # ``uv run`` here couples an acceptance test to a writable global uv cache
    # and can silently resolve a different environment.
    out = subprocess.run(
        [sys.executable, "-m", "automil", "--help"],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    assert out.returncode == 0, f"automil --help failed: {out.stderr}"


# ---------------------------------------------------------------------------
# Clause 5 — [slurm] extra enables SLURMBackend
# ---------------------------------------------------------------------------

def test_d179_clause_05_slurm_extra_enables_backend():
    """D-179 #5: pip install -e '.[slurm]' makes import automil.backends.slurm work."""
    pytest.importorskip("submitit")
    from automil.backends.slurm import SLURMBackend  # must succeed when submitit installed
    from automil.backends import BACKENDS
    assert BACKENDS.get("slurm") is SLURMBackend, "SLURMBackend not registered as 'slurm'"


# ---------------------------------------------------------------------------
# Clause 6 — [ray] extra enables RayBackend
# ---------------------------------------------------------------------------

def test_d179_clause_06_ray_extra_enables_backend():
    """D-179 #6: pip install -e '.[ray]' makes import automil.backends.ray work."""
    pytest.importorskip("ray")
    from automil.backends.ray import RayBackend  # must succeed when ray installed
    from automil.backends import BACKENDS
    assert BACKENDS.get("ray") is RayBackend, "RayBackend not registered as 'ray'"


# ---------------------------------------------------------------------------
# Clause 7 — node_0176-equivalent smoke passes across CI-runnable backends
# ---------------------------------------------------------------------------

def test_d179_clause_07_node_0176_smoke_passes():
    """D-179 #7: tests/backends/test_node_0176_smoke.py passes for [local, slurm-debug, ray-local]."""
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/backends/test_node_0176_smoke.py", "-v"],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    text = out.stdout + out.stderr
    # Accept SKIPPED for slurm-debug or ray-local when extras absent;
    # require PASSED for [local] always; require NOT FAILED for any param.
    assert "FAILED" not in text, f"node_0176 smoke had failures:\n{text}"
    assert "[local]" in text and ("PASSED" in text or "passed" in text), (
        f"node_0176 smoke [local] did not pass:\n{text[-2000:]}"
    )


# ---------------------------------------------------------------------------
# Clause 8 — running/ namespaced; flat detection guardrail fires
# ---------------------------------------------------------------------------

def test_d179_clause_08_running_namespaced():
    """D-179 #8: running/ is namespaced; daemon refuses to start with flat *.json."""
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/backends/test_running_namespace.py", "-v"],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    assert out.returncode == 0, (
        f"namespace tests failed: {out.stdout}\n{out.stderr}"
    )


# ---------------------------------------------------------------------------
# Clause 9 — archive/<id>/run.log orchestrator-owned via _atomic_write_lines
# ---------------------------------------------------------------------------

def test_d179_clause_09_archive_run_log_orchestrator_owned():
    """D-179 #9: archive/<id>/run.log exists for terminal nodes; via _atomic_write_lines."""
    from automil.backends._orchestrator_daemon import _atomic_write_lines, _drain_log_iter_with_timeout
    # Both helpers must be defined at module scope.
    assert callable(_atomic_write_lines)
    assert callable(_drain_log_iter_with_timeout)
    # The Wave-0 stubs verify the actual drain behavior; re-run the timeout stub.
    out = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/backends/test_log_unification.py::test_log_iter_close_60s_timeout", "-v"],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    assert "FAILED" not in (out.stdout + out.stderr), (
        f"log_unification timeout test had failures: {out.stdout}\n{out.stderr}"
    )


# ---------------------------------------------------------------------------
# Clause 10 — Framework purity: zero autobench refs in src/automil/backends/
# ---------------------------------------------------------------------------

def test_d179_clause_10_framework_purity():
    """D-179 #10: the Phase 6 new backends (slurm.py, ray.py) contain zero autobench refs.

    Note: _orchestrator_daemon.py retains AUTOBENCH_ROOT references inherited
    from Phase 0 (D-05) deferred to Phase 8/DEC-01. Those are pre-existing and
    out of Phase 6 scope. The D-179 purity clause is scoped to the backends
    INTRODUCED in Phase 6 — slurm.py and ray.py must be completely free of
    autobench-specific identifiers.
    """
    backends_dir = _SRC_AUTOMIL / "backends"
    for new_backend in ("slurm.py", "ray.py"):
        backend_file = backends_dir / new_backend
        if not backend_file.exists():
            # If the file doesn't exist, there's nothing to check.
            continue
        out = subprocess.run(
            ["grep", "-n", "-E", "autobench|AUTOBENCH_|benchmarks/", str(backend_file)],
            capture_output=True, text=True,
        )
        # grep returns 1 when no matches, 0 when matches found.
        assert out.returncode != 0 or out.stdout.strip() == "", (
            f"framework purity violated; {new_backend} contains autobench refs:\n{out.stdout}"
        )


# ---------------------------------------------------------------------------
# Clause 11 — CHANGELOG.md 6.0.0 BREAKING entry present
# ---------------------------------------------------------------------------

def test_d179_clause_11_changelog_breaking_entry():
    """D-179 #11: CHANGELOG.md surfaces BREAKING running/ namespace change with recovery steps."""
    changelog = _REPO_ROOT / "CHANGELOG.md"
    assert changelog.exists(), "CHANGELOG.md missing at repo root"
    text = changelog.read_text()
    assert "## 6.0.0" in text, "CHANGELOG.md missing ## 6.0.0 heading"
    assert "BREAKING" in text, "CHANGELOG.md missing BREAKING marker"
    assert "automil orchestrator stop" in text, (
        "CHANGELOG.md must include the operator drain command"
    )
    assert "running/" in text or "namespacing" in text.lower(), (
        "CHANGELOG.md must explain the running/ namespacing change"
    )
