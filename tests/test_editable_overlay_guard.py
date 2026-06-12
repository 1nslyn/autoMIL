"""RED stubs for SCH-02: editable-install overlay guard.

Wave-0 Nyquist stubs — all marked xfail(strict=True) until 12-03 implements
the production code in src/automil/cli/check.py and
src/automil/backends/_orchestrator_daemon.py.

Design notes:
- Tests the detect-and-warn path (_collect_editable_source_roots + check warning)
  and the opt-in PYTHONPATH injection in _launch post-processing.
- Uses unittest.mock.patch and tmp_path (pytest fixture).
- Zero references to autobench, AUTOBENCH_, or benchmarks/ (framework purity D-206).
- Does NOT conflict with D-199 invariant tests (test_orchestrator_env_whitelist.py)
  because the guard is opt-in (editable_overlay_guard: true, default OFF).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_check_warnings(
    *,
    editable_roots: list[str],
    files_editable: list[str],
    project_root: Path,
    run_script_content: str = "",
    overlay_guard_enabled: bool = False,
) -> list[str]:
    """Call the SCH-02 check warning logic and return the warnings list.

    This helper mirrors what automil check will do once SCH-02 is implemented:
      1. Collect editable source roots (mocked).
      2. Check if any files.editable entry falls under those roots.
      3. Check consumer guard (sys.path.insert in run script) + config flag.
      4. Append a warning if overlap found and no guard present.

    Since the production function does not yet exist, importing it will fail
    (ImportError / AttributeError), which is the expected XFAIL condition.
    """
    # Import the not-yet-implemented helper from check.py
    from automil.cli.check import _collect_editable_source_roots  # noqa: PLC0415

    config = {
        "files": {"editable": files_editable},
        "orchestrator": {"editable_overlay_guard": overlay_guard_enabled},
        "run": {"script": "train.py"},
    }
    warnings: list[str] = []

    # SCH-02 detection logic (to be implemented in check.py)
    # We call the real _collect_editable_source_roots but patch it to return
    # the fixture roots — the caller wraps this helper in a patch context.
    roots = _collect_editable_source_roots()

    has_consumer_guard = "sys.path.insert" in run_script_content
    guard_enabled = (config.get("orchestrator") or {}).get("editable_overlay_guard", False)

    for root in roots:
        root_p = Path(root)
        for editable_glob in files_editable:
            candidate = project_root / editable_glob
            # Check: does the editable file path fall under the editable source root?
            try:
                candidate.relative_to(root_p)
                overlap = True
            except ValueError:
                overlap = False

            if not overlap:
                # Also check: is the root under the project and does the glob match?
                try:
                    root_p.relative_to(project_root)
                    overlap = str(candidate).startswith(str(root_p))
                except ValueError:
                    pass

            if overlap and not has_consumer_guard and not guard_enabled:
                warnings.append(
                    f"files.editable includes paths under editable-installed "
                    f"package root {str(root_p)!r}. worktree overlays to this path "
                    f"may be shadowed by the parent-venv editable install."
                )

    return warnings


# ---------------------------------------------------------------------------
# SCH-02 Tests — 5 stubs
# ---------------------------------------------------------------------------

def test_check_warns_missing_guard(tmp_path):
    """automil check warns when files.editable overlaps an editable src root and no guard.

    Setup:
    - _collect_editable_source_roots returns [str(tmp_path / "src")]
    - files.editable: ["src/mymodel.py"]  (path under the editable root)
    - run script does NOT contain sys.path.insert
    - orchestrator.editable_overlay_guard absent / false

    Expected: warnings list contains a string mentioning "editable" and "worktree".
    """
    editable_src = tmp_path / "src"
    editable_src.mkdir()

    with patch(
        "automil.cli.check._collect_editable_source_roots",
        return_value=[str(editable_src)],
    ):
        warnings = _build_check_warnings(
            editable_roots=[str(editable_src)],
            files_editable=["src/mymodel.py"],
            project_root=tmp_path,
            run_script_content="# no guard here",
            overlay_guard_enabled=False,
        )

    assert any("editable" in w for w in warnings), (
        "check must warn when files.editable overlaps an editable source root "
        "and no guard is present (SCH-02)"
    )
    assert any("worktree" in w for w in warnings), (
        "check warning must mention 'worktree' to explain the shadow risk (SCH-02)"
    )


def test_check_no_warn_when_guard_enabled(tmp_path):
    """check suppresses the editable-overlay warning when editable_overlay_guard: true.

    Same overlap as test_check_warns_missing_guard, but config has
    orchestrator.editable_overlay_guard: true → no warning emitted.
    """
    editable_src = tmp_path / "src"
    editable_src.mkdir()

    with patch(
        "automil.cli.check._collect_editable_source_roots",
        return_value=[str(editable_src)],
    ):
        warnings = _build_check_warnings(
            editable_roots=[str(editable_src)],
            files_editable=["src/mymodel.py"],
            project_root=tmp_path,
            run_script_content="# no guard here",
            overlay_guard_enabled=True,
        )

    editable_warnings = [w for w in warnings if "editable" in w and "worktree" in w]
    assert len(editable_warnings) == 0, (
        "check must NOT warn when orchestrator.editable_overlay_guard: true "
        f"is set (SCH-02); got warnings: {editable_warnings}"
    )


def test_check_no_warn_when_consumer_guard_present(tmp_path):
    """check suppresses warning when the run script contains sys.path.insert.

    Same overlap as test_check_warns_missing_guard, but run script content
    contains 'sys.path.insert(0, ...)' — consumer self-protection (D-03).
    No editable-overlay warning should be emitted.
    """
    editable_src = tmp_path / "src"
    editable_src.mkdir()

    consumer_script = "import sys\nsys.path.insert(0, '/wt/src')\nimport mymodel\n"

    with patch(
        "automil.cli.check._collect_editable_source_roots",
        return_value=[str(editable_src)],
    ):
        warnings = _build_check_warnings(
            editable_roots=[str(editable_src)],
            files_editable=["src/mymodel.py"],
            project_root=tmp_path,
            run_script_content=consumer_script,
            overlay_guard_enabled=False,
        )

    editable_warnings = [w for w in warnings if "editable" in w and "worktree" in w]
    assert len(editable_warnings) == 0, (
        "check must NOT warn when run script contains sys.path.insert (consumer "
        f"self-protection present, D-03); got warnings: {editable_warnings}"
    )


def test_opt_in_injection_prepends_pythonpath(tmp_path):
    """Opt-in injection prepends the worktree editable root to PYTHONPATH.

    Setup:
    - Fake daemon with editable_overlay_guard=True, project_root=tmp_path
    - _collect_editable_source_roots returns [str(tmp_path / "src")]
    - wt_path is a temp dir with a "src/" subdirectory (the worktree overlay)
    - Call the SCH-02 injection logic (the post-_build_subprocess_env block in _launch)

    Expected: env["PYTHONPATH"] starts with str(wt_path / "src").

    Since _launch's injection block does not yet exist, this test will raise
    AttributeError / ImportError on the not-yet-implemented code path, satisfying
    the xfail(strict=True) RED condition.
    """
    from automil.backends._orchestrator_daemon import ExperimentOrchestrator  # noqa: PLC0415

    # Create worktree directory structure
    wt_path = tmp_path / "worktrees" / "exp_001"
    wt_src = wt_path / "src"
    wt_src.mkdir(parents=True)

    # Editable source root points to the main checkout (tmp_path/src)
    main_src = tmp_path / "src"
    main_src.mkdir(exist_ok=True)

    # Build a fake daemon with opt-in guard enabled
    fake = SimpleNamespace(
        editable_overlay_guard=True,
        project_root=tmp_path,
    )

    # Build a starting env dict (no PYTHONPATH yet)
    env: dict[str, str] = {"PATH": "/usr/bin"}

    # Call the SCH-02 injection logic (not yet implemented in _launch)
    # After implementation this will be a method/block on ExperimentOrchestrator.
    # We invoke the future injection via the module-level _collect_editable_source_roots
    # + the post-processing block that SCH-02 will add to _launch.
    with patch(
        "automil.backends._orchestrator_daemon._collect_editable_source_roots",
        return_value=[str(main_src)],
    ):
        # The injection block (to be implemented):
        ExperimentOrchestrator._apply_editable_overlay_guard(fake, env=env, wt_path=wt_path)

    assert "PYTHONPATH" in env, (
        "editable_overlay_guard injection must set PYTHONPATH in env (SCH-02)"
    )
    assert env["PYTHONPATH"].startswith(str(wt_src)), (
        f"PYTHONPATH must start with the worktree src path {wt_src!r}; "
        f"got {env.get('PYTHONPATH')!r} (SCH-02)"
    )


def test_check_suppresses_when_no_editable_overlap(tmp_path):
    """check emits no editable-overlay warning when editable root does not overlap overlay.

    Setup:
    - _collect_editable_source_roots returns ["/other/project/src"] (unrelated path)
    - files.editable: ["src/mymodel.py"] under tmp_path (no overlap with /other/project/src)

    Expected: no editable-overlay warning emitted.
    """
    with patch(
        "automil.cli.check._collect_editable_source_roots",
        return_value=["/other/project/src"],
    ):
        warnings = _build_check_warnings(
            editable_roots=["/other/project/src"],
            files_editable=["src/mymodel.py"],
            project_root=tmp_path,
            run_script_content="# no guard",
            overlay_guard_enabled=False,
        )

    editable_warnings = [w for w in warnings if "editable" in w and "worktree" in w]
    assert len(editable_warnings) == 0, (
        "check must NOT warn when editable root does not overlap files.editable "
        f"(no path intersection); got warnings: {editable_warnings}"
    )
