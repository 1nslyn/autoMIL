"""Wave 0 stub for D-176 acceptance smoke (BCK-05/06).

Parametrised over [local, slurm-debug, ray-local]; runs a CCRCC node_0176-equivalent
synthetic 1-fold variant; asserts result.json primary_value within +-0.005 of LocalBackend baseline.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

# Synthetic baseline: matches what the synthetic train.py below produces.
_LOCAL_BASELINE_COMPOSITE = 0.502


def _setup_smoke_project(tmp_path: Path) -> tuple[Path, Path]:
    """Init a minimal git repo with a synthetic train.py producing primary_value=0.502."""
    train_py = tmp_path / "train.py"
    train_py.write_text(
        "import json, pathlib\n"
        "result = {\n"
        "    'status': 'completed',\n"
        "    'metrics': {'val_auc': 0.87, 'val_bacc': 0.81, 'test_auc': 0.87, 'test_bacc': 0.83},\n"
        "    'primary_value': 0.502,\n"
        "    'elapsed_seconds': 1,\n"
        "}\n"
        "pathlib.Path('result.json').write_text(json.dumps(result))\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)
    automil_dir = tmp_path / "automil"
    (automil_dir / "orchestrator" / "running").mkdir(parents=True)
    (automil_dir / "orchestrator" / "archive").mkdir(parents=True)
    return tmp_path, automil_dir


@pytest.mark.parametrize("backend_name", ["local", "slurm-debug", "ray-local"])
def test_node_0176_equivalent_primary_value_within_tolerance(backend_name, tmp_path):
    """D-176: every CI-runnable backend reproduces primary_value within +-0.005 of LocalBackend baseline."""
    if backend_name.startswith("slurm"):
        pytest.importorskip("submitit")
    if backend_name.startswith("ray"):
        pytest.importorskip("ray")

    project_root, automil_dir = _setup_smoke_project(tmp_path)
    # Implementation in plan 06-09 wires the backend factory and runs the synthetic spec.
    from tests.backends._smoke_helpers import run_node_0176_smoke
    primary_value = run_node_0176_smoke(backend_name, project_root, automil_dir)
    assert abs(primary_value - _LOCAL_BASELINE_COMPOSITE) <= 0.005, \
        f"backend={backend_name} primary_value={primary_value} drifted > 0.005 from baseline {_LOCAL_BASELINE_COMPOSITE}"
