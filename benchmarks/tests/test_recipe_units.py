"""The composite recipe's optimizer wrapper, held to closed forms on tiny models.

The weight average, the per-step learning-rate factors and the regrouping that
gives the feature projection and the body their own factors run through the
calls the trainers make. ``test_recipe_seams.py`` holds the policy's other seams;
``test_recipes.py`` drives the policy through the real arms.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

import autobench.pipeline.nnmil._imports  # noqa: F401, E402  (puts benchmarks/lib on sys.path)
from _recipe_probe import (  # noqa: E402
    WIDTH,
    RecordingAdamW,
    RecordingSGD,
    body_head_model,
    recipe_class,
    train_epochs,
)
from automil.registry._state import _clear_registry  # noqa: E402

BASE_LR = 0.1


@pytest.fixture(autouse=True)
def _isolated_registry():
    _clear_registry()
    yield
    _clear_registry()


def _wrapped(tmp_path, *, role="main", **constants):
    """A recipe policy with ``constants``, and the recording SGD it wraps for a body-and-head model."""
    policy = recipe_class(tmp_path, **constants)()
    model = body_head_model()
    raw = RecordingSGD(model.parameters(), lr=BASE_LR)
    return policy, model, raw, policy.wrap_optimizer_for(raw, role=role)


def _lrs_of(raw, param):
    """The learning rate ``param``'s group stepped with, step by step."""
    index = next(i for i, g in enumerate(raw.param_groups) if any(p is param for p in g["params"]))
    return [lrs[index] for lrs in raw.lrs]


def _closed_form_average(steps, decay):
    """sum_i (1 - d) d^(N-1-i) w_i / (1 - d^N) over the weights after each of N steps."""
    n = len(steps)
    share = [(1 - decay) * decay ** (n - 1 - i) / (1 - decay**n) for i in range(n)]
    return [sum(s * step[k] for s, step in zip(share, steps)) for k in range(len(steps[0]))]


def _ids(*params):
    return {id(p) for p in params}


def _group_params(optimizer, param):
    group = next(g for g in optimizer.param_groups if any(p is param for p in g["params"]))
    return _ids(*group["params"])


def _simple_mil():
    from nnMIL.network_architecture.models.simple_mil import SimpleMIL

    return SimpleMIL(input_dim=96, hidden_dim=24, pred_num=2)


def _nnmil_adamw(model, cls=RecordingAdamW):
    """The two AdamW groups nnMIL builds: 1-D parameters without decay, the rest with it."""
    named = list(model.named_parameters())
    flat = [p for _, p in named if p.ndim < 2]
    rest = [p for _, p in named if p.ndim >= 2]
    return cls([{"params": flat, "weight_decay": 0.0}, {"params": rest, "weight_decay": 0.01}], lr=1e-3)


# --- the weight average --------------------------------------------------------------------


@pytest.mark.parametrize("steps", [1, 2, 7])
def test_the_swapped_in_average_is_the_bias_corrected_exponential_average(tmp_path, steps):
    policy, model, raw, optimizer = _wrapped(tmp_path, EMA_DECAY=0.9)
    train_epochs(policy, model, optimizer, epochs=1, steps_per_epoch=steps, validate=False)
    last = [p.detach().clone() for p in model.parameters()]

    policy.before_validation(epoch=0)
    for param, expected in zip(model.parameters(), _closed_form_average(raw.weights, 0.9)):
        assert torch.allclose(param, expected, atol=1e-6)
    policy.should_stop(default=False, epoch=0, metrics={})
    for param, value in zip(model.parameters(), last):
        assert torch.equal(param, value)


def test_two_roles_are_averaged_swapped_and_restored_together(tmp_path):
    policy = recipe_class(tmp_path, EMA_DECAY=0.5)()
    models = [body_head_model(1), body_head_model(2)]
    raws = [RecordingSGD(m.parameters(), lr=BASE_LR) for m in models]
    tiers = [
        policy.wrap_optimizer_for(raws[0], role="tier1"),
        policy.wrap_optimizer_for(raws[1], role="tier2"),
    ]
    for _ in range(3):
        for model, tier in zip(models, tiers):
            tier.zero_grad()
            model(torch.ones(2, WIDTH)).sum().backward()
            tier.step()
    last = [[p.detach().clone() for p in m.parameters()] for m in models]

    policy.before_validation(epoch=0)
    for model, raw in zip(models, raws):
        for param, expected in zip(model.parameters(), _closed_form_average(raw.weights, 0.5)):
            assert torch.allclose(param, expected, atol=1e-6)
    policy.should_stop(default=False, epoch=0, metrics={})
    for model, weights in zip(models, last):
        for param, value in zip(model.parameters(), weights):
            assert torch.equal(param, value)


def test_a_role_is_wrapped_once(tmp_path):
    policy, model, raw, _ = _wrapped(tmp_path, EMA_DECAY=0.9)
    with pytest.raises(ValueError, match="already wrapped"):
        policy.wrap_optimizer_for(RecordingSGD(model.parameters(), lr=BASE_LR), role="main")


def test_restoring_what_was_never_swapped_is_a_no_op_and_swapping_twice_is_refused(tmp_path):
    policy, model, raw, optimizer = _wrapped(tmp_path, EMA_DECAY=0.9)
    train_epochs(policy, model, optimizer, epochs=1, steps_per_epoch=2, validate=False)
    before = [p.detach().clone() for p in model.parameters()]
    policy.should_stop(default=False, epoch=0, metrics={})
    assert all(torch.equal(p, b) for p, b in zip(model.parameters(), before))
    policy.before_validation(epoch=0)
    with pytest.raises(RuntimeError, match="already in place"):
        policy.before_validation(epoch=1)


def test_a_step_the_grad_scaler_skips_never_moves_the_average(tmp_path):
    policy, model, raw, optimizer = _wrapped(tmp_path, EMA_DECAY=0.9)
    scaler = torch.amp.GradScaler("cpu")
    # nnMIL's wikg_mil.py turns autograd's anomaly mode on for the whole process when it is
    # imported, and that mode refuses the NaN an overflowing backward pass produces.
    with torch.autograd.set_detect_anomaly(False):
        for factor in (1.0, float("inf"), 1.0):  # the middle gradient overflows: the scaler skips it
            optimizer.zero_grad()
            scaler.scale(model(torch.ones(2, WIDTH)).sum() * factor).backward()
            scaler.step(optimizer)
            scaler.update()
    assert len(raw.weights) == 2, "the overflowing step must never reach the optimizer"

    policy.before_validation(epoch=0)
    for param, expected in zip(model.parameters(), _closed_form_average(raw.weights, 0.9)):
        assert torch.allclose(param, expected, atol=1e-6)


def test_validating_before_any_optimizer_step_swaps_nothing_in(tmp_path):
    policy, model, raw, optimizer = _wrapped(tmp_path, EMA_DECAY=0.9)  # e.g. the scaler skipped every step
    before = [p.detach().clone() for p in model.parameters()]
    policy.before_validation(epoch=0)
    assert all(torch.equal(p, b) for p, b in zip(model.parameters(), before))
    policy.should_stop(default=False, epoch=0, metrics={})
    assert all(torch.equal(p, b) for p, b in zip(model.parameters(), before))


def test_with_no_average_the_hooks_leave_the_weights_alone(tmp_path):
    policy, model, raw, optimizer = _wrapped(tmp_path)
    train_epochs(policy, model, optimizer, epochs=1, steps_per_epoch=2, validate=False)
    before = [p.detach().clone() for p in model.parameters()]
    policy.before_validation(epoch=0)
    policy.should_stop(default=False, epoch=0, metrics={})
    assert all(torch.equal(p, b) for p, b in zip(model.parameters(), before))


# --- the learning-rate factors -------------------------------------------------------------


class _Boom(RecordingSGD):
    def step(self, *args, **kwargs):
        raise RuntimeError("boom")


def test_the_learning_rates_are_restored_after_every_step(tmp_path):
    policy, model, raw, optimizer = _wrapped(
        tmp_path, ARM="simple_mil", WARMUP_STEPS=3, WARMUP_START=0.5, UNFREEZE=(1, 2, 0.5),
        BODY_LR_SCALE=0.5,
    )
    train_epochs(policy, model, optimizer, epochs=2, steps_per_epoch=3, validate=False)
    assert any(lr != BASE_LR for lrs in raw.lrs for lr in lrs), "the factors never reached a step"
    assert all(group["lr"] == BASE_LR for group in raw.param_groups)

    failing = _Boom(model.parameters(), lr=BASE_LR)
    broken = recipe_class(tmp_path, WARMUP_STEPS=3, WARMUP_START=0.5)().wrap_optimizer(failing)
    with pytest.raises(RuntimeError, match="boom"):
        broken.step()
    assert failing.param_groups[0]["lr"] == BASE_LR


def _step_lrs(tmp_path, *, epochs, steps_per_epoch, **constants):
    policy, model, raw, optimizer = _wrapped(tmp_path, **constants)
    train_epochs(policy, model, optimizer, epochs=epochs, steps_per_epoch=steps_per_epoch)
    return model, raw


def test_the_warmup_is_linear_from_its_start_factor(tmp_path):
    _, raw = _step_lrs(tmp_path, epochs=2, steps_per_epoch=4, WARMUP_STEPS=4, WARMUP_START=0.2)
    assert [lrs[0] for lrs in raw.lrs] == pytest.approx(
        [BASE_LR * f for f in (0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0)]
    )


def test_the_cosine_runs_to_zero_over_its_epochs_counted_from_epoch_zeros_steps(tmp_path):
    _, raw = _step_lrs(tmp_path, epochs=3, steps_per_epoch=4, COSINE_EPOCHS=2)
    # Epoch 0 steps at the full rate (its length is not known yet); from the first step of
    # epoch 1 the factor follows 0.5 (1 + cos(pi t / 8)) and is 0 from t = 8.
    expected = [1.0] * 4 + [0.5 * (1 + math.cos(math.pi * t / 8)) for t in range(4, 8)] + [0.0] * 4
    assert [lrs[0] for lrs in raw.lrs] == pytest.approx([BASE_LR * f for f in expected], abs=1e-12)


def test_the_cosine_starts_where_the_warmup_ends(tmp_path):
    _, raw = _step_lrs(
        tmp_path, epochs=3, steps_per_epoch=4, WARMUP_STEPS=2, WARMUP_START=0.5, COSINE_EPOCHS=2,
    )
    # span = 2 epochs * 4 steps - 2 warmup steps = 6; progress = (t - 2) / 6 from t = 4.
    expected = [0.5, 0.75, 1.0, 1.0] + [0.5 * (1 + math.cos(math.pi * (t - 2) / 6)) for t in range(4, 8)]
    expected += [0.0] * 4
    assert [lrs[0] for lrs in raw.lrs] == pytest.approx([BASE_LR * f for f in expected], abs=1e-12)


def test_the_step_anneal_multiplies_the_rate_from_its_epoch(tmp_path):
    _, raw = _step_lrs(tmp_path, epochs=4, steps_per_epoch=2, STEP_ANNEAL=(2, 0.1))
    expected = [1.0] * 4 + [0.1] * 4
    assert [lrs[0] for lrs in raw.lrs] == pytest.approx([BASE_LR * f for f in expected])


def test_the_factors_multiply_whatever_rate_the_group_holds(tmp_path):
    """DTFD's own MultiStepLR and nnMIL's cosine_lr keep rewriting the group's rate between steps."""
    policy, model, raw, optimizer = _wrapped(tmp_path, STEP_ANNEAL=(0, 0.5))
    raw.param_groups[0]["lr"] = 0.04  # a native scheduler's value
    train_epochs(policy, model, optimizer, epochs=1, steps_per_epoch=1, validate=False)
    assert raw.lrs == [[pytest.approx(0.02)]] and raw.param_groups[0]["lr"] == 0.04


def test_the_projection_ramps_from_its_start_factor_while_the_rest_trains_at_full_rate(tmp_path):
    model, raw = _step_lrs(tmp_path, epochs=4, steps_per_epoch=2, UNFREEZE=(1, 3, 0.5))
    projection, head = model[0].weight, model[2].weight
    ramp = [0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0]  # epoch 0, 1: start; 2: halfway; 3: full
    assert _lrs_of(raw, projection) == pytest.approx([BASE_LR * f for f in ramp])
    assert _lrs_of(raw, model[0].bias) == pytest.approx([BASE_LR * f for f in ramp])
    assert _lrs_of(raw, head) == pytest.approx([BASE_LR] * 8)
    assert _lrs_of(raw, model[2].bias) == pytest.approx([BASE_LR] * 8)


class _BagLogits(torch.nn.Module):
    """An ABMIL as ``train_epochs`` drives a model: one bag in, its logits out."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, bag):
        return self.model(bag.unsqueeze(0))["logits"]


def test_the_projection_bias_of_an_abmil_shaped_model_ramps_with_its_weight(tmp_path):
    from autobench.pipeline.abmil.model import ABMIL

    model = ABMIL(in_dim=WIDTH, M=4, L=3)
    policy = recipe_class(tmp_path, UNFREEZE=(1, 3, 0.5))()
    raw = RecordingSGD(model.parameters(), lr=BASE_LR)
    optimizer = policy.wrap_optimizer(raw)
    train_epochs(policy, _BagLogits(model), optimizer, epochs=4, steps_per_epoch=2, validate=False)

    ramp = [0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0]
    projection = model.feature_extractor[0]
    for param in (projection.weight, projection.bias):
        assert _lrs_of(raw, param) == pytest.approx([BASE_LR * f for f in ramp])
    for param in model.parameters():
        if param is not projection.weight and param is not projection.bias:
            assert _lrs_of(raw, param) == pytest.approx([BASE_LR] * 8)


@pytest.mark.parametrize("model", [
    # the next parameter is a matrix, not a 1-D bias (DTFD's projection: nothing but weights)
    torch.nn.Sequential(torch.nn.Linear(WIDTH, 4, bias=False), torch.nn.ReLU(), torch.nn.Linear(4, 2)),
    torch.nn.Linear(WIDTH, 4, bias=False),  # no next parameter at all
], ids=["a matrix follows", "nothing follows"])
def test_a_projection_without_a_bias_takes_no_other_parameter_with_it(tmp_path, model):
    projection = next(model.parameters())
    optimizer = recipe_class(tmp_path, UNFREEZE=(1, 2, 0.5))().wrap_optimizer(
        torch.optim.SGD(model.parameters(), lr=BASE_LR)
    )
    assert _group_params(optimizer, projection) == _ids(projection)


def test_the_body_trains_at_its_scale_and_the_last_layer_at_full_rate(tmp_path):
    model, raw = _step_lrs(tmp_path, epochs=2, steps_per_epoch=2, ARM="simple_mil", BODY_LR_SCALE=0.2)
    for body in (model[0].weight, model[0].bias):
        assert _lrs_of(raw, body) == pytest.approx([BASE_LR * 0.2] * 4)
    for head in (model[2].weight, model[2].bias):
        assert _lrs_of(raw, head) == pytest.approx([BASE_LR] * 4)


def test_the_unfreeze_ramp_and_the_body_scale_multiply_on_the_projection(tmp_path):
    model, raw = _step_lrs(
        tmp_path, epochs=3, steps_per_epoch=1, ARM="simple_mil", UNFREEZE=(0, 2, 0.5),
        BODY_LR_SCALE=0.2,
    )
    for projection in (model[0].weight, model[0].bias):
        assert _lrs_of(raw, projection) == pytest.approx([BASE_LR * f * 0.2 for f in (0.5, 0.75, 1.0)])
    assert _lrs_of(raw, model[2].weight) == pytest.approx([BASE_LR] * 3)


def test_parameter_groups_are_regrouped_only_when_a_switch_asks_for_it(tmp_path):
    for constants in ({}, {"EMA_DECAY": 0.9, "COSINE_EPOCHS": 3, "STEP_ANNEAL": (1, 0.5), "WARMUP_STEPS": 2}):
        _, _, raw, optimizer = _wrapped(tmp_path, **constants)
        assert optimizer.param_groups is raw.param_groups and len(raw.param_groups) == 1
    _, _, raw, _ = _wrapped(tmp_path, UNFREEZE=(1, 2, 0.5))
    assert len(raw.param_groups) == 2
    # projection (weight and bias together), a middle layer that is body only, and the head
    deeper = torch.nn.Sequential(
        torch.nn.Linear(WIDTH, 4), torch.nn.ReLU(), torch.nn.Linear(4, 4), torch.nn.ReLU(),
        torch.nn.Linear(4, 2),
    )
    raw = RecordingSGD(deeper.parameters(), lr=BASE_LR)
    recipe_class(tmp_path, ARM="simple_mil", UNFREEZE=(1, 2, 0.5), BODY_LR_SCALE=0.5)().wrap_optimizer(raw)
    assert len(raw.param_groups) == 3


def test_a_split_part_keeps_every_setting_of_the_group_it_came_from(tmp_path):
    model = _simple_mil()
    flat = [p for p in model.parameters() if p.ndim < 2]
    rest = [p for p in model.parameters() if p.ndim >= 2]
    raw = torch.optim.AdamW([
        {"params": flat, "weight_decay": 0.0, "betas": (0.8, 0.9)},
        {"params": rest, "weight_decay": 0.05, "eps": 1e-6, "lr": 2e-3},
    ], lr=1e-3)
    settings = {id(p): {k: v for k, v in g.items() if k != "params"} for g in raw.param_groups for p in g["params"]}
    optimizer = recipe_class(tmp_path, ARM="simple_mil", UNFREEZE=(1, 2, 0.1), BODY_LR_SCALE=0.1)().wrap_optimizer(raw)
    assert len(optimizer.param_groups) > 2, "nothing was split"
    for group in optimizer.param_groups:
        for param in group["params"]:
            assert {k: v for k, v in group.items() if k != "params"} == settings[id(param)]


def test_the_average_covers_every_parameter_group(tmp_path):
    policy = recipe_class(tmp_path, EMA_DECAY=0.9)()
    model = body_head_model()
    raw = _nnmil_adamw(model)  # the 1-D parameters in one group, the matrices in another
    optimizer = policy.wrap_optimizer(raw)
    train_epochs(policy, model, optimizer, epochs=1, steps_per_epoch=4, validate=False)
    policy.before_validation(epoch=0)
    in_group_order = [p for group in raw.param_groups for p in group["params"]]
    for param, expected in zip(in_group_order, _closed_form_average(raw.weights, 0.9)):
        assert torch.allclose(param, expected, atol=1e-6)


def test_the_wrapper_delegates_what_the_arms_read_and_hides_nothing(tmp_path):
    _, _, raw, optimizer = _wrapped(tmp_path, EMA_DECAY=0.9)
    assert optimizer.state_dict() == raw.state_dict() and optimizer.defaults is raw.defaults
    assert getattr(optimizer, "_step_supports_amp_scaling", False) is False  # GradScaler probes this
    with pytest.raises(AttributeError):
        optimizer.no_such_attribute


# --- which layers are the projection and the head, on production shapes ---------------------


def test_on_simple_mil_the_projection_is_the_two_attention_inputs_not_the_classification_head(tmp_path):
    model = _simple_mil()
    for arm, with_head in (("simple_mil", False), ("abmil", True)):
        policy = recipe_class(tmp_path, ARM=arm, UNFREEZE=(1, 2, 0.1))()
        optimizer = policy.wrap_optimizer(_nnmil_adamw(model))
        projection = next(g for g in optimizer.param_groups if any(p is model.V.weight for p in g["params"]))
        assert _ids(*projection["params"]) == _ids(
            model.V.weight, model.U.weight, *([model.cls.weight] if with_head else []),
        )
        assert projection["weight_decay"] == 0.01, "the split keeps the group's own settings"
        assert _group_params(optimizer, model.V.bias) == _ids(model.V.bias, model.U.bias, model.w.bias, model.cls.bias)


def test_on_simple_mil_the_head_keeps_the_full_rate_and_the_attention_takes_the_body_scale(tmp_path):
    model = _simple_mil()
    raw = _nnmil_adamw(model)
    optimizer = recipe_class(tmp_path, ARM="simple_mil", BODY_LR_SCALE=0.1)().wrap_optimizer(raw)
    for param in model.parameters():
        param.grad = torch.zeros_like(param)
    optimizer.step()
    body = [model.V.weight, model.V.bias, model.U.weight, model.U.bias, model.w.weight, model.w.bias]
    head = [model.cls.weight, model.cls.bias]
    for param in body:
        assert _lrs_of(raw, param) == [pytest.approx(1e-4)]
    for param in head:
        assert _lrs_of(raw, param) == [pytest.approx(1e-3)]


def test_the_projection_is_the_widest_layer_of_every_arms_real_model(tmp_path):
    from autobench.pipeline.abmil.model import ABMIL
    from autobench.pipeline.clam._imports import CLAM_MB
    from autobench.pipeline.dtfd.config import DTFDConfig
    from autobench.pipeline.dtfd.model import build_dtfd_bundle

    width = 1536
    abmil = ABMIL(in_dim=width, M=500, L=128)
    clam = CLAM_MB(embed_dim=width)
    bundle = build_dtfd_bundle(width, 2, DTFDConfig())
    cases = [  # the projection's weight, then its bias: DTFD's projection has none
        ("abmil", "main", abmil.parameters(), (abmil.feature_extractor[0].weight, abmil.feature_extractor[0].bias)),
        ("clam_mb", "main", clam.parameters(), (clam.attention_net[0].weight, clam.attention_net[0].bias)),
        ("dtfd_mil", "tier1", bundle.tier1_parameters(), (bundle.dim_reduction.fc1.weight,)),
    ]
    for arm, role, parameters, projection in cases:
        policy = recipe_class(tmp_path, ARM=arm, UNFREEZE=(1, 2, 0.1))()
        optimizer = policy.wrap_optimizer_for(torch.optim.Adam(parameters, lr=1e-3), role=role)
        assert _group_params(optimizer, projection[0]) == _ids(*projection), arm
        assert len(optimizer.param_groups) == 2, arm

    tier2 = torch.optim.Adam(bundle.att_cls.parameters(), lr=1e-3)
    wrapped = recipe_class(tmp_path, ARM="dtfd_mil", UNFREEZE=(1, 2, 0.1))().wrap_optimizer_for(tier2, role="tier2")
    assert len(wrapped.param_groups) == 1, "the second DTFD tier has no feature projection"


def test_unfreeze_refuses_a_model_whose_widest_layer_is_not_the_feature_projection(tmp_path):
    # The features are WIDTH wide, but the layer taking the most inputs is the 9-wide second one.
    model = torch.nn.Sequential(torch.nn.Linear(WIDTH, 9), torch.nn.ReLU(), torch.nn.Linear(9, 2))
    policy = recipe_class(tmp_path, UNFREEZE=(1, 2, 0.1))()
    optimizer = policy.wrap_optimizer(torch.optim.SGD(model.parameters(), lr=0.1))
    with pytest.raises(ValueError, match="UNFREEZE"):
        train_epochs(policy, model, optimizer, epochs=1, steps_per_epoch=1)


def test_unfreeze_and_body_scale_refuse_a_model_with_nothing_to_act_on(tmp_path):
    none = torch.nn.Linear(WIDTH, 2)
    with pytest.raises(ValueError, match="BODY_LR_SCALE found no body"):
        recipe_class(tmp_path, ARM="simple_mil", BODY_LR_SCALE=0.1)().wrap_optimizer(
            torch.optim.SGD(none.parameters(), lr=0.1)
        )
    no_matrix = [torch.nn.Parameter(torch.zeros(3))]
    with pytest.raises(ValueError, match="UNFREEZE"):
        recipe_class(tmp_path, UNFREEZE=(1, 2, 0.1))().wrap_optimizer(torch.optim.SGD(no_matrix, lr=0.1))


# --- off means off ------------------------------------------------------------------------


@pytest.mark.parametrize("constants", [
    {}, {"EMA_DECAY": 0, "LS_EPS": 0.0, "POSITIVE_SHARE": False, "MAX_TILES": 0, "WARMUP_STEPS": 0,
         "COSINE_EPOCHS": 0, "MIN_EPOCHS": 0, "BODY_LR_SCALE": None, "KEEP_FRAC": None},
])
def test_none_zero_and_false_all_mean_off(tmp_path, constants):
    policy, model, raw, optimizer = _wrapped(tmp_path, **constants)
    train_epochs(policy, model, optimizer, epochs=2, steps_per_epoch=2)
    assert all(lrs == [BASE_LR] for lrs in raw.lrs) and optimizer.param_groups is raw.param_groups
