"""Centralized nnMIL imports with sys.path management.

nnMIL is not installed as a package — it lives at benchmarks/lib/nnMIL/.
This module adds it to sys.path once and re-exports the components we need.
"""

import sys

from autobench import LIB_ROOT

_LIB_DIR = str(LIB_ROOT)

if _LIB_DIR not in sys.path:
    sys.path.insert(0, _LIB_DIR)

from nnMIL.training.trainers.classification_trainer import ClassificationTrainer  # noqa: E402
from nnMIL.training.trainers.survival_trainer import SurvivalTrainer  # noqa: E402
from nnMIL.training.trainers.survival_porpoise_trainer import (  # noqa: E402
    SurvivalPorpoiseTrainer,
)
from nnMIL.preprocessing.experiment_planner import ExperimentPlanner  # noqa: E402
from nnMIL.utilities.plan_loader import load_plan  # noqa: E402

__all__ = [
    "ClassificationTrainer",
    "SurvivalTrainer",
    "SurvivalPorpoiseTrainer",
    "ExperimentPlanner",
    "load_plan",
]
