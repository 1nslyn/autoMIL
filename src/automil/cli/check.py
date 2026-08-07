"""check command: validate project setup before running experiments."""
from __future__ import annotations

import json
import os
import site
import subprocess
from pathlib import Path

import click
import yaml

from automil.cli import main
from automil.cli._helpers import _find_automil_dir, _find_git_root

def _collect_editable_source_roots() -> list[str]:
    """Return editable source root paths from site-packages .pth / egg-link files.

    Scans site.getsitepackages() and site.getusersitepackages() for three
    editable-install file patterns:
      - _editable_impl_*.pth   (uv / pip PEP 660, modern)
      - __editable__*.pth      (older pip PEP 660 variant)
      - *.egg-link             (legacy setup.py develop)

    Each matching file's text content is the source root path. Returns a
    list[str] of source root directory paths that actually exist on disk.
    Catches OSError on file read and skips that file.
    """
    roots: list[str] = []
    try:
        site_dirs = list(site.getsitepackages())
    except AttributeError:
        # Old virtualenv (common on SLURM/HPC) monkey-patches away
        # getsitepackages; fall back gracefully so automil check and
        # _apply_editable_overlay_guard never crash in those environments.
        site_dirs = []
    user_site = site.getusersitepackages()
    if user_site:
        site_dirs.append(user_site)

    for site_dir in site_dirs:
        p = Path(site_dir)
        if not p.is_dir():
            continue
        for pattern in ("_editable_impl_*.pth", "__editable__*.pth", "*.egg-link"):
            for pth_file in p.glob(pattern):
                try:
                    content = pth_file.read_text().strip()
                    if content and Path(content).is_dir():
                        roots.append(content)
                except OSError:
                    continue
    return roots


# D-172 — required SLURM directives. `signal` is framework-mandated (Phase 4 D-115)
# and rejected if operator tries to override.
_REQUIRED_SLURM_DIRECTIVES: list[str] = [
    "partition", "account", "cpus_per_task", "mem_gb",
]
_FORBIDDEN_SLURM_DIRECTIVE_KEYS: list[str] = ["signal"]
_TODO_SENTINEL: str = "TODO_FILL_IN"


def _validate_slurm_directives(config: dict) -> None:
    """Raise SlurmDirectivesIncompleteError if SLURM config is incomplete (D-172).

    Checks:
      1. backend.slurm.walltime_seconds is a positive integer.
      2. All keys in _REQUIRED_SLURM_DIRECTIVES present and not equal to _TODO_SENTINEL.
      3. No keys in _FORBIDDEN_SLURM_DIRECTIVE_KEYS present (signal is framework-mandated).

    Pure function: no I/O, no Click. Wave-0 unit tests exercise it directly.
    """
    from automil.backends.errors import SlurmDirectivesIncompleteError  # noqa: PLC0415

    backend_cfg = config.get("backend", {}) or {}
    slurm_cfg = backend_cfg.get("slurm", {}) or {}
    directives = slurm_cfg.get("directives", {}) or {}

    walltime = slurm_cfg.get("walltime_seconds")
    missing: list[str] = []
    if not isinstance(walltime, int) or walltime <= 0:
        missing.append("walltime_seconds")

    for key in _REQUIRED_SLURM_DIRECTIVES:
        val = directives.get(key)
        if val is None:
            missing.append(key)
        elif isinstance(val, str) and val == _TODO_SENTINEL:
            missing.append(key)

    for forbidden in _FORBIDDEN_SLURM_DIRECTIVE_KEYS:
        if forbidden in directives:
            # D-172: framework-mandated signal cannot be overridden.
            missing.append(forbidden)

    if missing:
        raise SlurmDirectivesIncompleteError(missing)


def _validate_env_required(config: dict) -> list[str]:
    """Return env vars declared in env.required but absent from os.environ (D-202 / DEC-05).

    Pure function: no Click, no I/O, no exceptions. Wave-0 unit tests exercise
    it directly. Caller appends one issue per missing var (per-name iteration
    matches D-202's "for each missing var, emits a clear error" contract).

    Sentinel semantics: env vars are presence-only. A var that is set to
    'TODO_FILL_IN' (or any other value, including empty string) counts as
    PRESENT. The TODO sentinel pattern from _validate_slurm_directives only
    applies to YAML config values, not runtime env vars (PATTERNS.md
    anti-pattern #6).

    Returns:
        Names of vars declared under env.required that are not in os.environ.
        Empty list when all are set OR the env.required list is empty/missing/
        wrong-typed (the type-mismatch case is surfaced via a warning at the
        call site, not via the validator return value).
    """
    env_section = (config or {}).get("env") or {}
    raw_required = env_section.get("required", []) or []
    if not isinstance(raw_required, list):
        return []  # type-mismatch surfaced as a warning at the call site
    required = [str(k) for k in raw_required]
    return [k for k in required if k not in os.environ]


def _validate_ray_backend(config: dict, issues: list[str], warnings: list[str]) -> None:
    """Append issues/warnings for Ray backend selection (D-173 advisory).

    - Missing [ray] extra → issues.
    - RAY_ADDRESS set + connection fails → warnings (advisory, non-blocking).
    - RAY_ADDRESS set + connection ok → echo "Ray cluster reachable".
    """
    backend_cfg = config.get("backend", {}) or {}
    if backend_cfg.get("name") != "ray":
        return

    try:
        import ray  # noqa: PLC0415, F401
    except ImportError:
        issues.append(
            "backend.name is 'ray' but the [ray] extra is not installed. "
            "Run: uv sync --extra ray"
        )
        return

    ray_address = os.environ.get("RAY_ADDRESS")
    if not ray_address:
        return  # operator may be deferring to local fallback; non-issue.

    # Advisory connect-test (1s).
    import ray as _ray  # noqa: PLC0415
    try:
        if not _ray.is_initialized():
            _ray.init(address=ray_address, ignore_reinit_error=True, log_to_driver=False)
        click.echo(f"Ray cluster at {ray_address!r}: reachable.")
    except ConnectionError:
        warnings.append(
            f"RAY_ADDRESS={ray_address!r} set but cluster unreachable "
            f"(ConnectionError). Advisory only — operator may be intentionally pre-init."
        )


@main.command()
def check():
    """Validate project setup before running experiments."""
    git_root = _find_git_root()
    adir = _find_automil_dir()
    issues = []
    warnings = []

    # Check config.yaml
    config_path = adir / "config.yaml"
    config: dict = {}
    if not config_path.exists():
        issues.append("automil/config.yaml not found. Run 'automil init' first.")
    else:
        config = yaml.safe_load(config_path.read_text()) or {}

        # Check run script (skip if run.command is set — script may not exist)
        run_command = config.get("run", {}).get("command")
        run_script = config.get("run", {}).get("script") or "train.py"
        if not run_command:
            if not (git_root / run_script).exists():
                issues.append(f"Training script '{run_script}' not found at {git_root / run_script}")
            else:
                script_content = (git_root / run_script).read_text()
                if "result.json" not in script_content:
                    warnings.append(f"Training script '{run_script}' may not write result.json")

        # Check data paths
        for key in ["features_dir", "splits_dir", "mapping_csv"]:
            path = config.get("data", {}).get(key, "")
            if path and path.startswith("/path/to"):
                issues.append(f"data.{key} is still a placeholder: {path}")
            elif path and "${" not in path:
                resolved = Path(path)
                if not resolved.is_absolute():
                    resolved = git_root / resolved
                if not resolved.exists():
                    warnings.append(f"data.{key} path does not exist: {path}")

        # Check files.editable
        editable = config.get("files", {}).get("editable", [])
        if not editable:
            warnings.append("files.editable is empty. Auto-detect will capture ALL changed files.")

        # SCH-02 (D-02): warn when files.editable overlaps an editable-installed
        # package source root and no worktree import guard is present (ISSUE-010).
        editable_roots = _collect_editable_source_roots()
        run_script_path = git_root / (config.get("run", {}).get("script") or "train.py")
        run_command = config.get("run", {}).get("command")
        if run_command:
            # run.command set — no script file to inspect; assume no consumer guard
            has_consumer_guard = False
        elif run_script_path.exists() and run_script_path.suffix == ".py":
            # Only inspect Python scripts for sys.path.insert; shell wrappers
            # (.sh, etc.) never contain it and would always produce a false positive.
            has_consumer_guard = "sys.path.insert" in run_script_path.read_text()
        else:
            # Non-Python script or missing script: can't determine guard presence.
            has_consumer_guard = False
        overlay_guard_enabled = bool(
            (config.get("orchestrator") or {}).get("editable_overlay_guard", False)
        )
        for root in editable_roots:
            root_p = Path(root)
            for editable_glob in editable:
                candidate = git_root / editable_glob
                # Check if the candidate path falls under the editable source root.
                try:
                    candidate.relative_to(root_p)
                    overlap = True
                except ValueError:
                    overlap = False
                if not overlap:
                    # Also check via string prefix (handles non-resolved paths).
                    overlap = str(candidate).startswith(str(root_p) + os.sep) or str(candidate) == str(root_p)
                if overlap and not has_consumer_guard and not overlay_guard_enabled:
                    warnings.append(
                        f"files.editable includes paths under editable-installed package "
                        f"source root '{root}'. Worktree overlays to this path may be "
                        f"shadowed by the parent-venv editable install. "
                        f"Fix: add sys.path.insert(0, <worktree_src>) to your run script, "
                        f"or set orchestrator.editable_overlay_guard: true in automil/config.yaml."
                    )
                    break  # one warning per editable root is sufficient

        # Check baseline
        baseline_comp = config.get("baseline", {}).get("composite", 0)
        if baseline_comp == 0:
            warnings.append("baseline.composite is 0. Set this after running your first experiment.")

        # P2.3: surface the resolved cell budget so the operator sees it at setup.
        from automil.cells import format_duration  # noqa: PLC0415
        from automil.cells.capconfig import resolve_cap_config  # noqa: PLC0415
        try:
            cap = resolve_cap_config(config)
            click.echo(
                f"cap budget: {format_duration(cap.budget_seconds)} "
                f"(mode={cap.mode}, "
                f"safety_buffer={format_duration(cap.safety_buffer_seconds)})"
            )
            if not (0 < cap.safety_buffer_seconds < cap.budget_seconds):
                warnings.append(
                    f"cap.safety_buffer ({cap.safety_buffer_seconds}s) must satisfy "
                    f"0 < buffer < budget ({cap.budget_seconds}s)."
                )
            if cap.mode == "agent_active":
                from automil.activity_hooks import (  # noqa: PLC0415
                    missing_claude_activity_hooks,
                )

                # Campaign cells intentionally live inside the controller's
                # outer git repository and do not contain their own .git.
                # Their runtime overlay is nevertheless cell-local, adjacent
                # to automil/config.yaml; the git root remains authoritative
                # only for repo-owned training source.
                settings_path = adir.parent / ".claude" / "settings.json"
                try:
                    settings = json.loads(settings_path.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    issues.append(
                        "cap.mode=agent_active requires the Claude active-time "
                        f"observer, but {settings_path} is unreadable: {exc}. "
                        "Run `uv run automil init --update --runtime claude "
                        "--no-healthcheck`, or choose cap.mode: wall_clock."
                    )
                else:
                    missing_hooks = missing_claude_activity_hooks(settings)
                    if missing_hooks:
                        issues.append(
                            "cap.mode=agent_active is missing observer setting(s): "
                            f"{', '.join(missing_hooks)}. Run `uv run "
                            "automil init --update --runtime claude "
                            "--no-healthcheck`."
                        )
        except ValueError as exc:
            issues.append(f"cap config invalid: {exc}")

    # Check GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            warnings.append("nvidia-smi failed. GPU scheduling may not work correctly.")
        else:
            n_gpus = len(result.stdout.strip().splitlines())
            click.echo(f"GPUs detected: {n_gpus}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        warnings.append("nvidia-smi not found. GPU scheduling will use fallback.")

    # Check orchestrator directories
    for d in ["queue", "running", "archive", "completed"]:
        if not (adir / "orchestrator" / d).exists():
            issues.append(f"automil/orchestrator/{d}/ missing. Run 'automil init'.")

    # Cell files are accounting evidence. Report each obsolete/invalid journal
    # independently so operators can rematerialize the affected cells without
    # one bad row hiding the rest of the registry.
    from automil.cells import scan_cells  # noqa: PLC0415

    for error in scan_cells(adir / "cells").errors:
        issues.append(f"Cell journal {error.path.name}: {error.message}")

    # D-172/D-173: Phase 6 backend validation (only when a non-local backend is selected).
    backend_name = (config.get("backend", {}) or {}).get("name", "local")
    if backend_name == "slurm":
        try:
            _validate_slurm_directives(config)
        except Exception as exc:  # SlurmDirectivesIncompleteError or any other issue
            from automil.backends.errors import SlurmDirectivesIncompleteError  # noqa: PLC0415
            if isinstance(exc, SlurmDirectivesIncompleteError):
                issues.append(
                    f"backend.slurm directives incomplete — missing or "
                    f"sentinel-valued: {exc.missing_keys}. "
                    f"Edit automil/config.yaml: backend.slurm."
                )
            else:
                raise
    elif backend_name == "ray":
        _validate_ray_backend(config, issues, warnings)
    elif backend_name != "local":
        warnings.append(
            f"backend.name={backend_name!r} is unknown. Expected one of: local, slurm, ray."
        )

    # CLN-05: report the resolved nvidia-smi path so operators can see whether
    # path pinning is in effect (D-18). The constant is set at orchestrator.py
    # module import via shutil.which('nvidia-smi') — see Plan 03.
    from automil.orchestrator import NVIDIA_SMI_PATH

    if NVIDIA_SMI_PATH != "nvidia-smi":
        click.echo(f"nvidia-smi: {NVIDIA_SMI_PATH}")
    else:
        click.echo("nvidia-smi: bare PATH lookup (path detection failed)")

    # CLN-02 / D-04 / D-06: surface the subprocess env whitelist so the operator
    # knows exactly what experiment processes will receive. Hardcoded system
    # whitelist comes from the orchestrator module; per-project passthrough is
    # read fresh from the config we already loaded above.
    from automil.orchestrator import (
        _SYSTEM_ENV_WHITELIST_LITERAL,
        _SYSTEM_ENV_WHITELIST_PREFIX,
    )

    literal_list = ", ".join(sorted(_SYSTEM_ENV_WHITELIST_LITERAL))
    prefix_list = ", ".join(f"{p}*" for p in _SYSTEM_ENV_WHITELIST_PREFIX)
    click.echo(f"env whitelist (system, literal): {literal_list}")
    click.echo(f"env whitelist (system, prefix-glob): {prefix_list}")

    passthrough: list[str] = []
    env_section = (config or {}).get("env") or {}
    raw_pt = env_section.get("passthrough", []) or []
    if isinstance(raw_pt, list):
        passthrough = [str(k) for k in raw_pt]
    else:
        warnings.append(
            f"config.yaml: env.passthrough must be a list of var names; "
            f"got {type(raw_pt).__name__} — ignoring."
        )
    if passthrough:
        click.echo("env.passthrough:")
        for key in passthrough:
            present = "OK" if key in os.environ else "MISSING"
            click.echo(f"  {key}: passthrough {present}")
    else:
        click.echo("env.passthrough: (none declared)")

    # D-202 / DEC-05: env.required validation.
    env_section_chk = (config or {}).get("env") or {}
    raw_required = env_section_chk.get("required", [])
    if raw_required and not isinstance(raw_required, list):
        warnings.append(
            f"config.yaml: env.required must be a list of var names; "
            f"got {type(raw_required).__name__}, ignoring."
        )

    missing_required = _validate_env_required(config)
    for name in missing_required:
        issues.append(
            f"Missing required env var: {name}; see automil/config.yaml: "
            f"env.required. Set the variable before running 'automil submit' "
            f"or 'automil orchestrator start'."
        )

    if not missing_required and isinstance(raw_required, list) and raw_required:
        click.echo(f"env.required: {len(raw_required)} declared, all set OK.")
    elif not raw_required:
        click.echo("env.required: (none declared)")

    # --- Phase 1 registry checks (REG-04 / REG-05 / D-46) ---
    from automil.registry.config import load_registry_config
    from automil.registry.scanner import scan_variants
    from automil.registry._state import _clear_registry
    from automil.registry.manifest import Manifest

    reg_cfg = load_registry_config(adir)

    # Protected-files dirty check (REG-05 / D-34): both staged and unstaged dirty fail.
    if reg_cfg.protected:
        try:
            git_status = subprocess.run(
                ["git", "status", "--porcelain", "--"] + list(reg_cfg.protected),
                cwd=git_root, capture_output=True, text=True, timeout=10,
            )
            dirty_lines = [ln for ln in git_status.stdout.splitlines() if ln.strip()]
            if dirty_lines:
                issues.append(
                    "registry.protected paths dirty in working tree:\n      "
                    + "\n      ".join(dirty_lines[:20])
                    + (
                        f"\n      ... ({len(dirty_lines) - 20} more)"
                        if len(dirty_lines) > 20 else ""
                    )
                    + "\n      Run `automil revert-baseline` to reset, or "
                    "commit the changes to a variant module via "
                    "`automil port-variant <node_id>`."
                )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            warnings.append(
                "Could not run `git status` for protected files — "
                "git may not be on PATH."
            )

    # Registry consistency (D-46).
    variants_root = adir / "variants"
    if variants_root.exists():
        _clear_registry()  # avoid pollution from prior CLI calls in same process
        scan_result = scan_variants(variants_root)
        for path, exc_str in scan_result.failed:
            issues.append(f"Variant module {path} failed import: {exc_str}")
        for var_path in scan_result.imported:
            manifest_path = var_path.with_suffix(".json")
            if not manifest_path.exists():
                warnings.append(
                    f"Variant module {var_path} has no sibling manifest "
                    f"({manifest_path.name}). Run `automil port-variant <node_id>` "
                    f"to regenerate, or remove the variant module."
                )
                continue
            try:
                manifest = Manifest.read(manifest_path)
            except (ValueError, FileNotFoundError) as e:
                issues.append(f"Manifest {manifest_path} invalid: {e}")
                continue
            ok, reason = manifest.cross_check_with_module(var_path)
            if not ok:
                issues.append(
                    f"Manifest {manifest_path.name} mismatches docstring of "
                    f"{var_path.name}: {reason}"
                )

    # Repro manifest (D-40 / D-46): warn-not-fail if missing or stale.
    repro_path = adir / "repro_manifest.yaml"
    if not repro_path.exists():
        warnings.append(
            "automil/repro_manifest.yaml not found. Run "
            "`automil verify-repro <node_id>` after porting variants to "
            "generate the reproduction-sanity report."
        )
    else:
        if variants_root.exists():
            max_var_mtime = 0.0
            for p in variants_root.rglob("*.py"):
                try:
                    mt = p.stat().st_mtime
                    if mt > max_var_mtime:
                        max_var_mtime = mt
                except OSError:
                    continue
            try:
                repro_mtime = repro_path.stat().st_mtime
                if max_var_mtime > repro_mtime:
                    warnings.append(
                        "automil/repro_manifest.yaml is older than the newest "
                        "variant module. Run `automil verify-repro <node_id>` "
                        "to refresh."
                    )
            except OSError:
                pass

    # Report
    if issues:
        click.echo("\nISSUES (must fix):")
        for i, issue in enumerate(issues, 1):
            click.echo(f"  {i}. {issue}")

    if warnings:
        click.echo("\nWARNINGS:")
        for i, w in enumerate(warnings, 1):
            click.echo(f"  {i}. {w}")

    if not issues and not warnings:
        click.echo("All checks passed. Ready to run experiments.")
    elif not issues:
        click.echo(f"\n{len(warnings)} warning(s), no blocking issues.")
    else:
        click.echo(f"\n{len(issues)} issue(s) must be fixed before running.")
        raise click.exceptions.Exit(1)
