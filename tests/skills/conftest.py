"""Shared fixtures for tests/skills/ (used by 07-08, 07-09, 07-10)."""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def tmp_git_repo(tmp_path: Path) -> Path:
    """Create a tmp git repo with a synthetic train.py + initial commit.

    The synthetic train.py imports torch and defines a single nn.Module subclass
    so the skill's AST-walk model-class heuristic finds exactly one candidate.
    The repo is committed once (HEAD) so `automil submit` and `automil init`
    have a base_commit to anchor against.

    Per RESEARCH.md Pattern B, this fixture is the gold reference shape for
    skill-driven idempotency / dry-run-gate tests.
    """
    repo = tmp_path / "fake_project"
    repo.mkdir()

    (repo / "train.py").write_text(
        "import json, pathlib\n"
        "import torch\n"
        "\n"
        "class MyModel(torch.nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.linear = torch.nn.Linear(8, 2)\n"
        "    def forward(self, x):\n"
        "        return self.linear(x)\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    result = {\n"
        "        'status': 'completed',\n"
        "        'metrics': {'val_auc': 0.5, 'val_bacc': 0.5, 'test_auc': 0.5, 'test_bacc': 0.5},\n"
        "        'primary_value': 0.5,\n"
        "        'elapsed_seconds': 1,\n"
        "        'peak_vram_mb': 100,\n"
        "    }\n"
        "    pathlib.Path('result.json').write_text(json.dumps(result))\n"
    )
    (repo / "README.md").write_text("test repo\n")

    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=repo, check=True)

    return repo


def fake_nvidia_smi_3gpu(argv, **kwargs):
    """Stub for subprocess.run that simulates a 3-GPU CUDA workstation.

    Used by tests in 07-08, 07-09, 07-10 that need a deterministic healthcheck
    output without spawning real nvidia-smi.
    """
    if "mig.mode.current" in str(argv):
        return MagicMock(stdout="Disabled\nDisabled\nDisabled\n", returncode=0, stderr="")
    return MagicMock(stdout="0, 49140\n1, 49140\n2, 49140\n", returncode=0, stderr="")
