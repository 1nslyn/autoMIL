"""Protocol v5: the training-bag transform and the pre-validation hook.

Both seams are inert unless a policy defines them: the native path hands back
the very same bag object and draws no random number, so a baseline run cannot
change. An overriding policy gets a CPU generator private to its seam and
fold, and the runtime refuses results the arm's model could not take.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from automil.registry.variants.policy import PolicyVariant  # noqa: E402
from autobench.pipeline.policy_dispatch import PolicyRuntime  # noqa: E402


class _OptimizerOnly(PolicyVariant):
    def wrap_optimizer(self, opt):
        return opt


class _DropHalf(_OptimizerOnly):
    def transform_bag(self, features, *, label, epoch, generator):
        keep = torch.rand(features.shape[0], generator=generator) < 0.5
        keep[0] = True
        return features[keep]


def _rng_states():
    return torch.get_rng_state(), np.random.get_state()[1].copy(), random.getstate()


def _assert_rng_unchanged(before):
    torch_state, numpy_state, python_state = before
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert np.array_equal(np.random.get_state()[1], numpy_state)
    assert random.getstate() == python_state


def _runtime(policy=None, *, seed=42, fold=0):
    return PolicyRuntime(name="p" if policy else None, policy_factory=policy).for_fold(
        seed=seed, fold=fold,
    )


def test_the_native_runtime_returns_the_same_bag_and_draws_nothing():
    runtime = _runtime()
    bag = torch.randn(6, 4)
    before = _rng_states()
    assert runtime.transform_bag(bag, label=1, epoch=0) is bag
    assert runtime.before_validation(epoch=0) is None
    _assert_rng_unchanged(before)


def test_a_policy_without_the_seams_leaves_bags_and_generators_alone():
    runtime = _runtime(_OptimizerOnly)
    bag = torch.randn(6, 4)
    before = _rng_states()
    assert runtime.transform_bag(bag, label=1, epoch=0) is bag
    runtime.before_validation(epoch=0)
    _assert_rng_unchanged(before)
    assert runtime._generators == {}


def test_an_overriding_policy_draws_from_a_private_reproducible_generator():
    bag = torch.arange(40.0).reshape(20, 2)
    before = _rng_states()
    first = [_runtime(_DropHalf).transform_bag(bag, label=0, epoch=e) for e in range(1)]
    again = [_runtime(_DropHalf).transform_bag(bag, label=0, epoch=e) for e in range(1)]
    other_fold = _runtime(_DropHalf, fold=1).transform_bag(bag, label=0, epoch=0)
    _assert_rng_unchanged(before)
    assert torch.equal(first[0], again[0])
    assert not torch.equal(first[0], other_fold)
    assert first[0].shape[0] < bag.shape[0]


def test_successive_epochs_draw_new_subsets_from_the_same_stream():
    runtime = _runtime(_DropHalf)
    bag = torch.arange(40.0).reshape(20, 2)
    first = runtime.transform_bag(bag, label=0, epoch=0)
    second = runtime.transform_bag(bag, label=0, epoch=1)
    assert not torch.equal(first, second)


def test_modifying_the_bag_in_place_is_refused():
    class InPlace(_OptimizerOnly):
        def transform_bag(self, features, *, label, epoch, generator):
            return features.mul_(2.0)

    with pytest.raises(RuntimeError, match="in place"):
        _runtime(InPlace).transform_bag(torch.ones(4, 3), label=0, epoch=0)


@pytest.mark.parametrize("result", [
    lambda bag: bag.double(),
    lambda bag: bag[:, :2],
    lambda bag: bag.unsqueeze(0),
    lambda bag: bag.tolist(),
])
def test_the_result_must_keep_the_bags_dtype_and_feature_shape(result):
    class Bad(_OptimizerOnly):
        def transform_bag(self, features, *, label, epoch, generator):
            return result(features.clone())

    with pytest.raises(TypeError):
        _runtime(Bad).transform_bag(torch.ones(4, 3), label=0, epoch=0)


def test_the_arm_can_require_the_bag_shape_and_a_minimum_size():
    runtime = _runtime(_DropHalf)
    bag = torch.arange(40.0).reshape(20, 2)
    with pytest.raises(TypeError, match="keep"):
        runtime.transform_bag(bag, label=0, epoch=0, keep_shape=True)
    with pytest.raises(ValueError, match="at least 20"):
        _runtime(_DropHalf).transform_bag(bag, label=0, epoch=0, min_instances=20)


def test_an_overriding_policy_needs_the_folds_seed_and_index():
    runtime = PolicyRuntime(name="p", policy_factory=_DropHalf).for_fold()
    with pytest.raises(RuntimeError, match="seed and index"):
        runtime.transform_bag(torch.ones(4, 3), label=0, epoch=0)


def test_the_pre_validation_hook_runs_and_must_return_none():
    calls = []

    class Records(_OptimizerOnly):
        def before_validation(self, *, epoch):
            calls.append(epoch)

    class Returns(_OptimizerOnly):
        def before_validation(self, *, epoch):
            return True

    runtime = _runtime(Records)
    runtime.before_validation(epoch=3)
    assert calls == [3]
    with pytest.raises(TypeError, match="must return None"):
        _runtime(Returns).before_validation(epoch=0)
