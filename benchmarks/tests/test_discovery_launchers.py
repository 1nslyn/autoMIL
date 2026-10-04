"""Contracts for the discovery launchers' shell layer.

The per-cell job takes its allocation from SLURM or from the workstation
chain driver, and refuses to start with neither. The driver validates its
arguments before it touches anything. The library's wall end and pick order
are shared by both paths.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "benchmarks" / "scripts"
LIB = SCRIPTS / "slurm" / "discovery_lib.sh"
JOB = SCRIPTS / "slurm" / "submit_discovery_campaign.sh"
WRAPPER = SCRIPTS / "slurm" / "submit_discovery_cell.sh"
DRIVER = SCRIPTS / "run_discovery_chain.sh"
SCAN = {
    "pending": ["a", "c"], "finishable": ["b"], "claimed": [], "done": [],
    "stranded": [], "blocked": [], "notes": {}, "squeue_ok": False,
}


def _env(**extra: str) -> dict[str, str]:
    return {
        "PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp"),
        "USER": os.environ.get("USER", "tester"), **extra,
    }


def _run(argv: list[str], env: dict[str, str], cwd: Path | None = None):
    return subprocess.run(argv, capture_output=True, text=True, env=env, cwd=cwd)


def _lib(snippet: str, env: dict[str, str]):
    return _run(["bash", "-c", f'source "{LIB}"; {snippet}'], env)


def _checkout(tmp_path: Path) -> Path:
    """The driver derives the checkout from its own path: a copy of the
    launchers in an empty tree keeps every side effect inside tmp_path."""
    root = tmp_path / "checkout"
    (root / "benchmarks" / "scripts" / "slurm").mkdir(parents=True)
    for source in (DRIVER, LIB, JOB):
        shutil.copy(source, root / source.relative_to(REPO_ROOT))
    return root


@pytest.mark.parametrize("script", [LIB, JOB, WRAPPER, DRIVER], ids=lambda p: p.name)
def test_launcher_scripts_parse(script):
    result = _run(["bash", "-n", str(script)], _env())
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "missing", [None, "DISC_RUN_ID", "DISC_GPUS", "DISC_WALL_END", "DISC_CELL"],
)
def test_job_refuses_without_an_allocation(tmp_path, missing):
    """Not a SLURM job and no complete workstation allocation: the job stops
    before it reads or writes anything."""
    allocation = {
        "DISC_RUN_ID": "ws1", "DISC_GPUS": "0",
        "DISC_WALL_END": "1790000000", "DISC_CELL": "cell",
    }
    env = _env(DISC_PROJECT_DIR=str(tmp_path))
    if missing is not None:
        env.update({key: value for key, value in allocation.items() if key != missing})
    result = _run(["bash", str(JOB)], env, cwd=tmp_path)
    assert result.returncode == 1
    assert "run_discovery_chain.sh" in result.stdout
    assert list(tmp_path.iterdir()) == []


def test_job_takes_a_complete_workstation_allocation(tmp_path):
    """With all four DISC_* values the job passes its allocation check and
    stops at the next one: the project dir here is no campaign checkout."""
    env = _env(
        DISC_PROJECT_DIR=str(tmp_path), DISC_RUN_ID="ws1", DISC_GPUS="0",
        DISC_WALL_END="1790000000", DISC_CELL="cell",
    )
    result = _run(["bash", str(JOB)], env, cwd=tmp_path)
    assert result.returncode == 1
    assert "is not the campaign checkout" in result.stdout


@pytest.mark.parametrize("args, message", [
    (["--gpus", "0"], "--runtime is required"),
    (["--runtime", "runtime-aihub"], "--gpus must list"),
    (["--runtime", "runtime-aihub", "--gpus", "0,,1"], "--gpus must list"),
    (["--runtime", "runtime-aihub", "--gpus", "gpu0"], "--gpus must list"),
    (["--runtime", "runtime-aihub", "--gpus", "0,1,0"], "an index twice"),
    (["--runtime", "runtime-aihub", "--gpus", "0", "--wall-hours", "11"], "at least 12"),
    (["--runtime", "runtime-aihub", "--gpus", "0", "--wall-hours", "2.5"], "at least 12"),
    (["--runtime", "runtime-aihub", "--gpus", "0", "--resume"], "unknown option"),
], ids=["no-set", "no-gpus", "empty-index", "named-gpu", "repeated-index",
        "short-wall", "fractional-wall", "unknown-option"])
def test_driver_rejects_bad_arguments_before_any_side_effect(tmp_path, args, message):
    root = _checkout(tmp_path)
    result = _run(["bash", str(root / DRIVER.relative_to(REPO_ROOT)), *args], _env())
    assert result.returncode == 2
    assert message in result.stdout
    assert not (root / "logs").exists()


def test_driver_refuses_a_set_that_is_not_materialized(tmp_path):
    root = _checkout(tmp_path)
    result = _run(
        ["bash", str(root / DRIVER.relative_to(REPO_ROOT)),
         "--runtime", "runtime-elsewhere", "--gpus", "0,1"],
        _env(),
    )
    assert result.returncode == 1
    assert "runtime not materialized" in result.stdout
    assert not (root / "logs").exists()


@pytest.mark.parametrize("value, expected", [
    ("1790000000", "1790000000"),
    ("", "0"),
    ("1.79e9", "0"),
    ("-1", "0"),
])
def test_workstation_wall_end_is_the_drivers_epoch_second(value, expected):
    """A malformed end reads as unknown, so the job starts no session."""
    result = _lib("wall_end_epoch", _env(DISC_WALL_END=value))
    assert result.stdout.strip() == expected


def test_slurm_job_asks_squeue_even_with_a_workstation_wall_end(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    asked = tmp_path / "asked"
    squeue = bin_dir / "squeue"
    squeue.write_text(f'#!/bin/sh\necho "$@" > "{asked}"\necho N/A\n')
    squeue.chmod(0o755)
    env = _env(SLURM_JOB_ID="4242", DISC_WALL_END="1790000000")
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    result = _lib("wall_end_epoch", env)
    assert result.stdout.strip() == "0"
    assert "4242" in asked.read_text()


def _candidates(*cell: str):
    return _lib(
        f'pyrun() {{ "{sys.executable}" "$@"; }}; disc_candidates "$SCAN_JSON" {" ".join(cell)}',
        _env(SCAN_JSON=json.dumps(SCAN)),
    )


def test_candidates_put_finish_only_recoveries_first():
    result = _candidates()
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["finish:b", "full:a", "full:c"]


def test_candidates_narrow_to_one_cell_and_refuse_any_other():
    assert _candidates("c").stdout.split() == ["full:c"]
    refused = _candidates("zz")
    assert refused.returncode != 0
    assert "zz is not finishable or pending" in refused.stderr


def _fits_wall(tmp_path: Path, hours_left: int, elapsed_total: float | None):
    """disc_fits_wall for one cell with a five-fold baseline of
    ``elapsed_total`` seconds, on three GPUs, ``hours_left`` before the wall."""
    import time

    runtime = tmp_path / "runtime-ws"
    cell = runtime / "cell"
    cell.mkdir(parents=True)
    baseline = None if elapsed_total is None else {
        "resources": {"elapsed_seconds": {"total": elapsed_total}},
    }
    (cell / "campaign_state.json").write_text(json.dumps({"baseline": baseline}))
    snippet = (
        f'pyrun() {{ "{sys.executable}" "$@"; }}; RUNTIME="{runtime}"; '
        'disc_fits_wall cell 0,1,2'
    )
    env = _env(DISC_WALL_END=str(int(time.time()) + hours_left * 3600))
    return _run(["bash", "-c", f'source "{LIB}"; {snippet}'], env, cwd=REPO_ROOT)


def test_a_cell_that_would_outlast_its_wall_is_refused(tmp_path):
    """A KRAS CLAM baseline of 4.3 h on the RTX predicts 33.7 h on three
    GPUs (four batches one after another, one of them at the 10 h attempt
    timeout): a 12 h wall would end the session mid-discovery and strand it."""
    refused = _fits_wall(tmp_path, 12, 15600.0)
    assert refused.returncode != 0
    assert "exceeds 85% of the wall" in refused.stdout


def test_a_cell_that_fits_its_wall_starts(tmp_path):
    result = _fits_wall(tmp_path, 48, 15600.0)
    assert result.returncode == 0, result.stdout
    assert "predicted 33.7 h on 3 GPU" in result.stdout


def test_a_cell_without_a_baseline_time_is_refused(tmp_path):
    refused = _fits_wall(tmp_path, 48, None)
    assert refused.returncode != 0
    assert "no baseline time" in refused.stdout


@pytest.mark.skipif(
    shutil.which("setsid") is None or shutil.which("flock") is None,
    reason="the driver needs util-linux setsid and flock",
)
def test_ctrl_c_ends_the_chain_after_the_running_cell(tmp_path):
    """Forged Ctrl-C mid-cell: the cell's job runs to its end in its own
    session, and no second cell starts."""
    import signal
    import time

    root = _checkout(tmp_path)
    started = tmp_path / "started"
    stub_lib = root / "benchmarks/scripts/slurm/discovery_lib.sh"
    stub_lib.write_text(f"""
disc_paths() {{ PROJECT_DIR="{root}"; RUNTIME_NAME=set; LOG_DIR="{tmp_path}/logs"; mkdir -p "$LOG_DIR"; }}
disc_env() {{ :; }}
disc_static_preflight() {{ :; }}
disc_usage_probe() {{ :; }}
disc_scan() {{ echo '{{}}'; }}
disc_scan_report() {{ :; }}
disc_candidates() {{ printf 'full:first\\nfull:second\\n'; }}
""")
    stub_job = root / "benchmarks/scripts/slurm/submit_discovery_campaign.sh"
    stub_job.write_text(f'echo "$DISC_CELL" >> "{started}"; sleep 3; exit 0\n')
    (root / "logs").mkdir()
    driver = subprocess.Popen(
        ["bash", str(root / DRIVER.relative_to(REPO_ROOT)),
         "--runtime", "set", "--gpus", "0"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=_env(),
    )
    try:
        deadline = time.time() + 20
        while not started.exists() and time.time() < deadline:
            time.sleep(0.1)
        driver.send_signal(signal.SIGINT)
        output, _ = driver.communicate(timeout=30)
    finally:
        driver.kill()
    assert driver.returncode == 130, output
    assert started.read_text().split() == ["first"]
    assert "stop requested" in output
