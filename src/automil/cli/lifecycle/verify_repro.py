"""verify-repro command: rerun a node's recipe via registry; write manifest (CLI-09 / REG-09 / D-39)."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click

from automil.cli import main
from automil.cli._helpers import _find_automil_dir, _find_git_root
from automil.cli.lifecycle._shared import (
    _atomic_write_text,
    _get_node_or_die,
    _load_registry_or_die,
)

logger = logging.getLogger(__name__)


def _run_consumer_program(
    git_root: Path,
    adir: Path,
    node_id: str,
    base_commit: str,
) -> tuple[float, float]:
    """Run the consumer's program.py in a tmp worktree; return (primary_value, runtime_s).

    Phase 1 simple path: use a tempfile worktree via subprocess git, run
    `python program.py` with CUDA_VISIBLE_DEVICES not set, read result.json.

    The consumer's `program.py` path is read from `automil/config.yaml: run.script`
    (default `program.py`).
    """
    import shutil
    import tempfile

    import yaml

    cfg = yaml.safe_load((adir / "config.yaml").read_text()) or {}
    program_rel = (cfg.get("run") or {}).get("script") or "program.py"

    # Create a tmp worktree at base_commit.
    wt_root = Path(tempfile.mkdtemp(prefix="automil-verify-"))
    try:
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(wt_root), base_commit],
            cwd=git_root,
            capture_output=True,
            text=True,
            check=True,
        )

        # Apply the active config.yaml into the worktree (the worktree's
        # checkout has the base config; we want the CURRENT config so the
        # variant selection takes effect).
        wt_config = wt_root / adir.relative_to(git_root) / "config.yaml"
        wt_config.parent.mkdir(parents=True, exist_ok=True)
        # Copy current config.yaml.
        shutil.copy2(adir / "config.yaml", wt_config)

        # Copy variants/ tree (the registered modules need to be present in
        # the worktree for the consumer's program.py to import them).
        wt_variants = wt_root / adir.relative_to(git_root) / "variants"
        if wt_variants.exists():
            shutil.rmtree(wt_variants)
        if (adir / "variants").exists():
            shutil.copytree(adir / "variants", wt_variants)

        # Run program.py.
        program_path = wt_root / program_rel
        if not program_path.exists():
            raise click.ClickException(
                f"Consumer's program.py not found at {program_path} in worktree. "
                f"Set `run.script` in automil/config.yaml or place program.py at the project root."
            )

        # Use a clean env (no AUTOBENCH_* leakage; CUDA visibility removed).
        # Use sys.executable to ensure the correct Python interpreter (with
        # automil importable) is used — bare "python" may resolve to system
        # Python which lacks the framework.
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "AUTOMIL_NODE_ID": node_id,
        }

        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, str(program_path)],
            cwd=str(wt_root),
            env=env,
            capture_output=True,
            text=True,
        )
        runtime_s = time.time() - t0

        if proc.returncode != 0:
            raise click.ClickException(
                f"Consumer program.py exited {proc.returncode}: "
                f"{proc.stderr.strip()[:500]}. Inspect logs in {wt_root}."
            )

        result_path = wt_root / "result.json"
        if not result_path.exists():
            raise click.ClickException(
                f"Consumer program.py did not write result.json. "
                f"See {result_path}."
            )

        result = json.loads(result_path.read_text())
        primary_value = float(result.get("primary_value", 0.0))
        return primary_value, runtime_s

    finally:
        # Cleanup worktree.
        try:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(wt_root)],
                cwd=git_root,
                capture_output=True,
            )
        except Exception:
            pass
        if wt_root.exists():
            import shutil as _shutil
            _shutil.rmtree(wt_root, ignore_errors=True)


@main.command("verify-repro")
@click.argument("node_id")
@click.option(
    "--tolerance",
    default=None,
    type=float,
    help="Override registry.repro_tolerance (default: 0.005 from config).",
)
def verify_repro(node_id: str, tolerance: Optional[float]):
    """Reproduce a node's experiment via the registry path; write a manifest.

    Workflow: after porting a node's variant via `automil port-variant` and
    setting it active via `automil apply`, run `automil verify-repro <node_id>`
    to re-execute the recipe in a clean worktree and confirm the new primary_value
    matches the recorded primary_value within tolerance. Writes
    automil/repro_manifest.yaml with {expected, actual, tolerance, status:
    pass | fail}.

    Exit code: 0 on pass, non-zero on fail (so CI can gate).

    Phase 1 acceptance gate: this command works correctly on the synthetic
    mini-consumer in tests/fixtures/ (D-50). Real consumers (CCRCC etc.)
    follow up with their own demonstrations.
    """
    import yaml

    adir = _find_automil_dir()
    git_root = _find_git_root()
    cfg = _load_registry_or_die(adir)

    tol = tolerance if tolerance is not None else cfg.repro_tolerance

    node = _get_node_or_die(adir, node_id)
    expected = float(node.get("primary_value", 0.0))
    base_commit = node.get("base_commit", "HEAD")

    click.echo(f"verify-repro {node_id}: expected primary_value={expected:.6f}, tolerance=\xb1{tol}")
    click.echo(f"Running consumer program in worktree at {base_commit[:8]}...")

    actual, runtime_s = _run_consumer_program(git_root, adir, node_id, base_commit)

    diff = abs(actual - expected)
    status = "pass" if diff <= tol else "fail"

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=git_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    manifest = {
        "node_id": node_id,
        "expected_primary_value": expected,
        "actual_primary_value": actual,
        "tolerance": tol,
        "status": status,
        "git_sha": git_sha,
        "runtime_seconds": round(runtime_s, 3),
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    manifest_path = adir / "repro_manifest.yaml"
    _atomic_write_text(manifest_path, yaml.safe_dump(manifest, sort_keys=False))

    click.echo(f"Actual primary_value: {actual:.6f} (diff {diff:.6f})")
    click.echo(f"Status: {status}")
    click.echo(f"Manifest: {manifest_path}")

    if status == "fail":
        raise click.ClickException(
            f"verify-repro: primary_value drifted beyond tolerance "
            f"(|{actual:.6f} - {expected:.6f}| = {diff:.6f} > {tol}). "
            f"Inspect {manifest_path} and the variant module for porting bugs."
        )
