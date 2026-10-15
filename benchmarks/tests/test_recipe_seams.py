"""The composite recipe's policy seams other than the optimizer, held to closed forms.

The stopping gate, the loss target and weights, the training-bag subset and the
check on the constants each run through the call the trainers make, on tiny
inputs. ``test_recipe_units.py`` holds the optimizer wrapper; ``test_recipes.py``
drives the policy through the real arms.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from _recipe_probe import WIDTH, recipe_class  # noqa: E402
from automil.registry._state import _clear_registry  # noqa: E402
from autobench.pipeline.policy_step import TrainStep  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_registry():
    _clear_registry()
    yield
    _clear_registry()


# --- the stopping gate ---------------------------------------------------------------------


@pytest.mark.parametrize("default, epoch, expected", [
    (True, 2, False), (True, 3, True), (True, 9, True), (False, 3, False), (False, 0, False),
])
def test_early_stopping_waits_for_the_minimum_epoch(tmp_path, default, epoch, expected):
    policy = recipe_class(tmp_path, MIN_EPOCHS=3)()
    decision = policy.should_stop(default=default, epoch=epoch, metrics={})
    assert decision is expected


@pytest.mark.parametrize("default", [True, False])
def test_without_a_minimum_epoch_the_default_decision_stands(tmp_path, default):
    policy = recipe_class(tmp_path)()
    assert policy.should_stop(default=default, epoch=0, metrics={}) is default


# --- the loss ----------------------------------------------------------------------------


def _objective(policy, labels, n_classes=2):
    """The objective the policy's ``train_step`` asked the arm's pass to train on."""
    seen = []

    def backward_on(features, objective):
        seen.append(objective)
        return torch.zeros(len(labels), n_classes), torch.tensor(0.0)

    policy.train_step(TrainStep(
        torch.zeros(len(labels), WIDTH), torch.tensor(labels), n_classes=n_classes, epoch=0,
        policy="probe", backward_on=backward_on, step=lambda: None, zero_grad=lambda: None,
        generator=lambda: None,
    ))
    assert len(seen) == 1
    return seen[0]


def test_without_a_loss_switch_the_arms_own_loss_is_untouched(tmp_path):
    objective = _objective(recipe_class(tmp_path)(), [0, 1])
    assert objective.target is None and objective.weight is None


def test_label_smoothing_trains_toward_one_minus_eps_on_the_label_and_eps_over_classes_elsewhere(tmp_path):
    objective = _objective(recipe_class(tmp_path, LS_EPS=0.2)(), [0, 2, 1], n_classes=3)
    third = 0.2 / 3
    expected = torch.tensor([
        [0.8 + third, third, third], [third, third, 0.8 + third], [third, 0.8 + third, third],
    ])
    assert torch.allclose(objective.target, expected)
    assert objective.weight is None


def test_the_positive_share_weights_each_class_to_half_the_loss(tmp_path):
    objective = _objective(recipe_class(tmp_path, POSITIVE_SHARE=0.25)(), [1, 0, 0])
    assert objective.weight.tolist() == pytest.approx([0.5 / 0.25, 0.5 / 0.75, 0.5 / 0.75])
    assert objective.target is None


def test_class_weights_apply_per_class_and_multiply_with_the_positive_share(tmp_path):
    objective = _objective(recipe_class(tmp_path, CLASS_WEIGHTS=(1.0, 3.0))(), [1, 0])
    assert objective.weight.tolist() == pytest.approx([3.0, 1.0])
    both = _objective(recipe_class(tmp_path, CLASS_WEIGHTS=(1.0, 3.0), POSITIVE_SHARE=0.5)(), [1, 0])
    assert both.weight.tolist() == pytest.approx([3.0, 1.0])  # 0.5 / 0.5 = 1 for both classes


def test_the_class_share_is_for_two_classes_and_the_weights_must_cover_every_class(tmp_path):
    with pytest.raises(ValueError, match="POSITIVE_SHARE"):
        _objective(recipe_class(tmp_path, POSITIVE_SHARE=0.3)(), [0, 2], n_classes=3)
    with pytest.raises(ValueError, match="CLASS_WEIGHTS"):
        _objective(recipe_class(tmp_path, CLASS_WEIGHTS=(1.0, 2.0))(), [0, 2], n_classes=3)


# --- the bag ------------------------------------------------------------------------------


def _bag(rows):
    return torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3)


def _generator(seed=0):
    return torch.Generator().manual_seed(seed)


def test_without_a_tile_switch_the_bag_comes_back_untouched_and_no_number_is_drawn(tmp_path):
    policy, bag, generator = recipe_class(tmp_path)(), _bag(20), _generator()
    state = generator.get_state()
    assert policy.transform_bag(bag, label=0, epoch=0, generator=generator) is bag
    assert torch.equal(generator.get_state(), state)


@pytest.mark.parametrize("constants, rows, kept", [
    ({"MAX_TILES": 10}, 17, 10),
    ({"KEEP_FRAC": 0.5}, 20, 10),
    ({"MAX_TILES": 12, "KEEP_FRAC": 0.5}, 20, 10),      # the smaller cap wins
    ({"MAX_TILES": 12, "KEEP_FRAC": 0.9}, 20, 12),
    ({"KEEP_FRAC": 0.1}, 40, 8),                         # never below the floor
    ({"MAX_TILES": 2}, 40, 8),
    ({"MAX_TILES": 2, "MIN_TILES": 2}, 40, 2),
])
def test_a_tile_cap_keeps_a_random_subset_in_order_and_never_fewer_than_the_floor(tmp_path, constants, rows, kept):
    policy, bag = recipe_class(tmp_path, **constants)(), _bag(rows)
    version = bag._version
    out = policy.transform_bag(bag, label=1, epoch=3, generator=_generator())
    assert out.shape == (kept, 3) and out.dtype == bag.dtype
    picked = [int(row[0]) // 3 for row in out]
    assert picked == sorted(set(picked)), "a subset of distinct tiles, in their original order"
    assert torch.equal(out, bag[picked])
    assert bag._version == version


@pytest.mark.parametrize("constants, rows", [
    ({"MAX_TILES": 100}, 17), ({"KEEP_FRAC": 1.0}, 17), ({"MAX_TILES": 3}, 5), ({"MAX_TILES": 10}, 1),
])
def test_a_bag_that_is_already_small_enough_comes_back_untouched(tmp_path, constants, rows):
    policy, bag, generator = recipe_class(tmp_path, **constants)(), _bag(rows), _generator()
    state = generator.get_state()
    assert policy.transform_bag(bag, label=0, epoch=0, generator=generator) is bag
    assert torch.equal(generator.get_state(), state)


def test_the_subset_is_reproducible_from_the_generator_and_changes_with_it(tmp_path):
    policy, bag = recipe_class(tmp_path, MAX_TILES=10)(), _bag(40)
    first = policy.transform_bag(bag, label=0, epoch=0, generator=_generator(1))
    again = policy.transform_bag(bag, label=0, epoch=0, generator=_generator(1))
    other = policy.transform_bag(bag, label=0, epoch=0, generator=_generator(2))
    assert torch.equal(first, again) and not torch.equal(first, other)


# --- the constants ------------------------------------------------------------------------


@pytest.mark.parametrize("constants", [
    {"ARM": "resnet"}, {"EMA_DECAY": 1.0}, {"EMA_DECAY": True}, {"EMA_DECAY": "fast"},
    {"LS_EPS": 1.5}, {"POSITIVE_SHARE": -0.1}, {"CLASS_WEIGHTS": (1.0,)}, {"CLASS_WEIGHTS": (1.0, -1.0)},
    {"MAX_TILES": 1.5}, {"KEEP_FRAC": 1.5}, {"MIN_TILES": 0}, {"WARMUP_STEPS": -1},
    {"WARMUP_START": 2.0}, {"COSINE_EPOCHS": 0.5}, {"STEP_ANNEAL": (2,)}, {"STEP_ANNEAL": (-1, 0.5)},
    {"STEP_ANNEAL": (1, 0.0)}, {"MIN_EPOCHS": -1}, {"UNFREEZE": (3, 1, 0.1)}, {"UNFREEZE": (1, 3)},
    {"UNFREEZE": (1, 3, 1.5)}, {"BODY_LR_SCALE": -1.0},
    {"BODY_LR_SCALE": True}, {"KEEP_FRAC": True},  # True would read as the number 1: a switch that does nothing
])
def test_a_malformed_constant_is_refused_by_name(tmp_path, constants):
    (name,) = {k for k in constants if k != "ARM"} or {"ARM"}
    with pytest.raises(ValueError, match=name):
        recipe_class(tmp_path, **constants)()


@pytest.mark.parametrize("constants", [{"MAX_TILES": 512}, {"KEEP_FRAC": 0.5}])
def test_the_tile_caps_are_refused_on_simple_mil_because_nnmil_pools_the_padding(tmp_path, constants):
    """nnMIL pads every transformed bag back to its original length and pools the zero rows
    without a mask, so a cap there is a subset plus heavy zero padding."""
    (name,) = constants
    with pytest.raises(ValueError, match=f"{name}.*simple_mil.*zero"):
        recipe_class(tmp_path, ARM="simple_mil", **constants)()


@pytest.mark.parametrize("arm", ["abmil", "clam_mb", "dtfd_mil"])
def test_the_body_scale_is_refused_off_simple_mil_where_it_would_slow_the_projection(tmp_path, arm):
    with pytest.raises(ValueError, match="BODY_LR_SCALE.*simple_mil only"):
        recipe_class(tmp_path, ARM=arm, BODY_LR_SCALE=0.2)()


def test_each_arm_keeps_what_the_other_arms_refuse(tmp_path):
    recipe_class(tmp_path, ARM="simple_mil", BODY_LR_SCALE=0.2)()
    for arm in ("abmil", "clam_mb", "dtfd_mil"):
        recipe_class(tmp_path, ARM=arm, MAX_TILES=512, KEEP_FRAC=0.5)()


@pytest.mark.parametrize(("arm", "constants"), [
    ("simple_mil", {"MAX_TILES": None, "KEEP_FRAC": 0}),
    ("abmil", {"BODY_LR_SCALE": None}),
    ("dtfd_mil", {"BODY_LR_SCALE": False}),
])
def test_a_refused_switch_that_is_off_is_not_refused(tmp_path, arm, constants):
    recipe_class(tmp_path, ARM=arm, **constants)()
