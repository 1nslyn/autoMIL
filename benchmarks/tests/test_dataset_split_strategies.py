"""Every dataset YAML declares only implemented split strategies, consistently."""

from pathlib import Path

import pytest
import yaml

from autobench.pipeline.splits import DEVCV_PARTITION_SEEDS, _STRATEGIES

DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets"
DECLARES_DEVCV = [
    path for path in sorted(DATASETS_DIR.glob("**/*.yaml"))
    if "devcv" in (yaml.safe_load(path.read_text()).get("split_strategies") or {})
]


def test_every_devcv_strategy_is_registered():
    assert set(DEVCV_PARTITION_SEEDS) <= set(_STRATEGIES)
    assert len(set(DEVCV_PARTITION_SEEDS.values())) == len(DEVCV_PARTITION_SEEDS)


def test_some_dataset_declares_devcv():
    assert DECLARES_DEVCV


@pytest.mark.parametrize("path", DECLARES_DEVCV, ids=lambda p: p.stem)
def test_a_dataset_declares_all_devcv_strategies_and_their_tasks(path):
    raw = yaml.safe_load(path.read_text())
    declared = set(raw["split_strategies"])
    assert set(DEVCV_PARTITION_SEEDS) <= declared
    for task, feasible in raw["task_strategy_feasibility"].items():
        assert set(feasible) <= declared, task
        if "devcv" in feasible:
            assert set(DEVCV_PARTITION_SEEDS) <= set(feasible), task
