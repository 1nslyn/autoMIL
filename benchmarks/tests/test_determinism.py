"""L-2: every trainer must seed through the one shared, cuDNN-deterministic helper.

Before ``autobench.pipeline.determinism`` existed, seven trainer modules
each defined their own ``seed_everything``/``_seed_everything``, and the
copies had drifted: only clam/train.py and smmile/train.py set
``torch.backends.cudnn.benchmark = False`` / ``cudnn.deterministic = True``.
DTFD (classification AND survival), ABMIL, and TITAN (classification AND
survival) omitted them, so a fixed seed did not actually make those five
trainers reproducible -- cuDNN's autotuner was still free to pick a
different convolution kernel run to run.

This test pins two things: the shared helper itself sets the cuDNN flags,
and every trainer module that used to define its own copy now imports the
SAME function object (an identity check, so a future contributor
reintroducing a local copy fails this test rather than silently
reintroducing the asymmetry).
"""
from __future__ import annotations

import importlib

import pytest
import torch

from autobench.pipeline.determinism import seed_everything


class TestSeedEverything:
    def test_sets_cudnn_determinism_flags(self):
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            seed_everything(123)

            assert torch.backends.cudnn.benchmark is False
            assert torch.backends.cudnn.deterministic is True
        finally:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = False

    def test_same_seed_reproduces_random_and_numpy_and_torch_streams(self):
        import random
        import numpy as np

        seed_everything(7)
        a = (random.random(), np.random.rand(), torch.rand(1).item())

        seed_everything(7)
        b = (random.random(), np.random.rand(), torch.rand(1).item())

        assert a == b

    def test_different_seeds_diverge(self):
        import random

        seed_everything(1)
        a = random.random()
        seed_everything(2)
        b = random.random()
        assert a != b


# (module path, attribute name) for every trainer that used to carry its own
# copy of seed_everything. dtfd/abmil/titan named theirs "_seed_everything";
# clam named theirs "seed_everything" -- the identity check doesn't care
# which name a module binds it under, only that it's the SAME function.
#
_CONSUMERS = [
    ("autobench.pipeline.clam.train", "seed_everything"),
    ("autobench.pipeline.dtfd.train", "_seed_everything"),
    ("autobench.pipeline.dtfd.survival_train", "_seed_everything"),
    ("autobench.pipeline.abmil.train", "_seed_everything"),
    ("autobench.pipeline.titan.train", "_seed_everything"),
    ("autobench.pipeline.titan.survival_train", "_seed_everything"),
]


class TestEveryTrainerUsesTheSharedHelper:
    @pytest.mark.parametrize("module_path,attr", _CONSUMERS)
    def test_trainer_binds_the_shared_function_object(self, module_path, attr):
        module = importlib.import_module(module_path)
        bound = getattr(module, attr)
        assert bound is seed_everything, (
            f"{module_path}.{attr} is not autobench.pipeline.determinism."
            "seed_everything -- a local reimplementation has crept back in, "
            "which is exactly how the cuDNN-flag drift (L-2) happened."
        )

