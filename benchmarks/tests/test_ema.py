"""Protocol v5: the weight-averaging helper a policy composes from its seams."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from autobench.pipeline.ema import WeightAverage  # noqa: E402
from autobench.pipeline.policy_dispatch import PolicyRuntime  # noqa: E402


def _model_and_optimizer(seed: int = 0):
    torch.manual_seed(seed)
    model = torch.nn.Linear(3, 2)
    return model, torch.optim.SGD(model.parameters(), lr=0.1)


def _train_step(model, optimizer):
    optimizer.zero_grad()
    model(torch.ones(4, 3)).sum().backward()
    optimizer.step()


def test_the_average_follows_each_real_step():
    model, optimizer = _model_and_optimizer()
    average = WeightAverage(decay=0.5)
    tracked = average.track(optimizer)
    expected = [p.detach().clone() for p in model.parameters()]
    for _ in range(3):
        _train_step(model, tracked)
        expected = [0.5 * e + 0.5 * p.detach() for e, p in zip(expected, model.parameters())]
    raw = [p.detach().clone() for p in model.parameters()]
    average.swap_in()
    for param, value in zip(model.parameters(), expected):
        assert torch.allclose(param, value)
    average.restore()
    for param, value in zip(model.parameters(), raw):
        assert torch.equal(param, value)


def test_a_step_that_never_reaches_the_optimizer_leaves_the_average_alone():
    # GradScaler skips optimizer.step() on an overflowing step; the average must not move.
    model, optimizer = _model_and_optimizer()
    average = WeightAverage(decay=0.9)
    tracked = average.track(optimizer)
    before = [p.detach().clone() for p in model.parameters()]
    tracked.zero_grad()
    model(torch.ones(4, 3)).sum().backward()  # gradient computed, step skipped
    average.swap_in()
    for param, value in zip(model.parameters(), before):
        assert torch.equal(param, value)
    average.restore()


def test_two_optimizer_roles_are_averaged_and_swapped_together():
    first, first_opt = _model_and_optimizer(seed=1)
    second, second_opt = _model_and_optimizer(seed=2)
    average = WeightAverage(decay=0.5)
    tier1 = average.track(first_opt, role="tier1")
    tier2 = average.track(second_opt, role="tier2")
    _train_step(first, tier1)
    _train_step(second, tier2)
    raw = [p.detach().clone() for p in [*first.parameters(), *second.parameters()]]
    average.swap_in()
    assert not any(
        torch.equal(p, r) for p, r in zip([*first.parameters(), *second.parameters()], raw)
    )
    average.restore()
    for param, value in zip([*first.parameters(), *second.parameters()], raw):
        assert torch.equal(param, value)


def test_misuse_fails_loudly():
    model, optimizer = _model_and_optimizer()
    average = WeightAverage(decay=0.9)
    average.track(optimizer)
    with pytest.raises(ValueError):
        average.track(optimizer)
    average.restore()  # nothing swapped in yet: a no-op
    average.swap_in()
    with pytest.raises(RuntimeError):
        average.swap_in()
    average.restore()


@pytest.mark.parametrize("decay", [0.0, 1.0, -0.5, 1.5, True])
def test_decay_must_lie_strictly_between_zero_and_one(decay):
    with pytest.raises(ValueError):
        WeightAverage(decay=decay)


def test_a_learning_rate_scheduler_still_attaches_through_the_runtime():
    model, optimizer = _model_and_optimizer()
    tracked = WeightAverage(decay=0.9).track(optimizer)
    target = PolicyRuntime().scheduler_target(tracked, optimizer)
    scheduler = torch.optim.lr_scheduler.StepLR(target, step_size=1, gamma=0.5)
    _train_step(model, tracked)
    scheduler.step()
    assert tracked.param_groups[0]["lr"] == pytest.approx(0.05)
