"""Protocol v6: a policy's ``train_step`` recombines each arm's own pass, step and reset.

The arms run on the tiny CPU fixtures of ``test_policy_seam_arms.py`` and
``test_policy_seam_nnmil.py``. Every arm is held to the same facts:

* a policy whose ``train_step`` is the native body leaves the fold exactly as a
  native run left it;
* a soft target, a per-bag weight and a second pass each reach the arm's loss
  and change the run;
* the limits hold: two passes, one step, a step only after a pass, no in-place
  change to the batch, a valid target and weight.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from _seam_probe import fold_runtime  # noqa: E402
from automil.registry.variants.policy import PolicyVariant  # noqa: E402
from autobench.pipeline.policy_step import Objective, TrainStep  # noqa: E402
from tests.test_policy_seam_arms import ARMS  # noqa: E402
from tests.test_policy_seam_nnmil import _train as train_nnmil  # noqa: E402


def _policy(train_step, transform_bag=None) -> type[PolicyVariant]:
    namespace = {"wrap_optimizer": lambda self, opt: opt, "train_step": train_step}
    if transform_bag is not None:
        namespace["transform_bag"] = transform_bag
    return type("StepProbe", (PolicyVariant,), namespace)


def _native_body(self, step):
    step.zero_grad()
    step.backward_on()
    step.step()


def _smoothed(self, step):
    hard = torch.nn.functional.one_hot(step.labels, step.n_classes).float()
    step.zero_grad()
    step.backward_on(target=0.8 * hard + 0.2 / step.n_classes)
    step.step()


def _half_weight(self, step):
    step.zero_grad()
    step.backward_on(weight=0.5)
    step.step()


def _two_passes(self, step):
    """Sharpness-aware shape: a pass, a second pass at shifted weights, one step."""
    step.zero_grad()
    step.backward_on()
    step.backward_on(target=step.labels)
    step.step()


def _arm_runs():
    runs = [pytest.param(arm.run, id=arm.name) for arm in ARMS]
    for model_type in ("simple_mil", "vision_transformer"):
        def run(workdir, runtime, model_type=model_type):
            return train_nnmil(workdir, runtime, model_type=model_type)[1]
        runs.append(pytest.param(run, id=f"nnmil-{model_type}"))
    return runs


@pytest.mark.parametrize("run", _arm_runs())
def test_the_native_body_as_a_policy_changes_nothing(run, tmp_path):
    native = run(tmp_path / "native", fold_runtime())
    assert run(tmp_path / "policy", fold_runtime(_policy(_native_body))) == native


@pytest.mark.parametrize("body", [_smoothed, _half_weight, _two_passes],
                         ids=["soft-target", "weight", "two-passes"])
@pytest.mark.parametrize("run", _arm_runs())
def test_each_lever_reaches_the_arm(run, body, tmp_path):
    native = run(tmp_path / "native", fold_runtime())
    assert run(tmp_path / "policy", fold_runtime(_policy(body))) != native


def _first_loss(run, workdir, weight):
    """The loss the arm trains on for its first batch, at a per-bag weight."""
    seen: list[float] = []

    def body(self, step):
        step.zero_grad()
        _, loss = step.backward_on(weight=weight)
        seen.append(float(loss))
        step.step()

    run(workdir, fold_runtime(_policy(body)))
    return seen[0]


@pytest.mark.parametrize("run", _arm_runs())
def test_a_weight_scales_the_loss_the_arm_trains_on_once(run, tmp_path):
    """The first batch meets the same weights either way, so the loss scales by the weight
    exactly (CLAM: its bag and instance terms together, keeping their mixture)."""
    unit = _first_loss(run, tmp_path / "unit", 1.0)
    assert _first_loss(run, tmp_path / "half", 0.5) == pytest.approx(0.5 * unit, rel=1e-5)


def _every_other_batch(self, step):
    self.count = getattr(self, "count", 0) + 1
    if self.count % 2:
        step.zero_grad()
        step.backward_on()
        step.step()


@pytest.mark.parametrize("run", _arm_runs())
def test_a_policy_may_skip_batches_on_every_arm(run, tmp_path):
    native = run(tmp_path / "native", fold_runtime())
    assert run(tmp_path / "skipping", fold_runtime(_policy(_every_other_batch))) != native


def test_a_second_dtfd_pass_reuses_the_slides_pseudo_bag_split(tmp_path):
    """A second pass draws nothing from the trainer's RNG, so the slide order is unchanged."""
    dtfd = next(arm for arm in ARMS if arm.name == "dtfd")

    def bag_order(body):
        seen: list[float] = []

        def transform_bag(self, features, *, label, epoch, generator):
            seen.append(float(features.sum()))
            return features

        dtfd.run(tmp_path / body.__name__, fold_runtime(_policy(body, transform_bag)))
        return seen

    assert bag_order(_two_passes) == bag_order(_native_body)


def _stepped_twice(self, step):
    step.backward_on()
    step.step()
    step.step()


def _three_passes(self, step):
    for _ in range(3):
        step.backward_on()


def _step_first(self, step):
    step.step()


def _in_place(self, step):
    step.features.mul_(2.0)
    step.backward_on()


def _returns_value(self, step):
    step.backward_on()
    return step.loss


@pytest.mark.parametrize("body, error, message", [
    (_stepped_twice, RuntimeError, "stepped twice"),
    (_three_passes, RuntimeError, "more than 2 passes"),
    (_step_first, RuntimeError, "before any backward_on"),
    (_in_place, RuntimeError, "in place"),
    (_returns_value, TypeError, "must return None"),
])
def test_the_limits_hold_on_a_real_arm(body, error, message, tmp_path):
    abmil = next(arm for arm in ARMS if arm.name == "abmil")
    with pytest.raises(error, match=message):
        abmil.run(tmp_path / "run", fold_runtime(_policy(body)))


def _ctx(labels, *, n_classes=3):
    passes: list[Objective] = []
    ctx = TrainStep(
        torch.zeros(4, 2), torch.tensor(labels), n_classes=n_classes, epoch=0,
        policy="probe",
        backward_on=lambda f, objective: (passes.append(objective) or (torch.zeros(len(labels), n_classes), torch.tensor(0.0))),
        step=lambda: None, zero_grad=lambda: None, generator=lambda: None,
    )
    return ctx, passes


@pytest.mark.parametrize("target", [
    torch.tensor([3]),                      # class out of range
    torch.tensor([0, 1]),                   # wrong batch size
    torch.tensor([[0.5, 0.6, 0.0]]),        # does not sum to 1
    torch.tensor([[1.5, -0.5, 0.0]]),       # negative probability
    torch.tensor([[0.5, 0.5]]),             # wrong class count
    torch.tensor([[float("nan"), 1.0, 0.0]]),
    "1",
])
def test_an_invalid_target_is_refused(target):
    ctx, _ = _ctx([1])
    with pytest.raises((ValueError, TypeError)):
        ctx.backward_on(target=target)


@pytest.mark.parametrize("weight", [-1.0, float("inf"), torch.tensor([1.0, 1.0]), True, "1"])
def test_an_invalid_weight_is_refused(weight):
    ctx, _ = _ctx([1])
    with pytest.raises((ValueError, TypeError)):
        ctx.backward_on(weight=weight)


def test_a_valid_target_and_weight_reach_the_objective():
    ctx, passes = _ctx([2, 0])
    ctx.backward_on(target=torch.tensor([[0.0, 0.5, 0.5], [1.0, 0.0, 0.0]]), weight=2.0)
    assert passes[0].weight.tolist() == [2.0, 2.0]
    assert passes[0].classes.tolist() == [1, 0]
    ctx, passes = _ctx([1])
    ctx.backward_on(target=2)
    assert passes[0].target.tolist() == [2] and passes[0].classes.tolist() == [2]


class TestObjective:
    logits = torch.tensor([[2.0, -1.0, 0.5], [0.1, 0.3, -0.2]])
    labels = torch.tensor([0, 2])

    def test_without_a_target_or_weight_it_is_the_arms_own_loss(self):
        loss_fn = torch.nn.CrossEntropyLoss()
        assert torch.equal(
            Objective(self.labels).loss(loss_fn, self.logits), loss_fn(self.logits, self.labels),
        )

    def test_a_one_hot_target_with_unit_weight_equals_the_native_loss(self):
        loss_fn = torch.nn.CrossEntropyLoss()
        one_hot = torch.nn.functional.one_hot(self.labels, 3).float()
        objective = Objective(self.labels, one_hot, torch.ones(2))
        assert torch.allclose(objective.loss(loss_fn, self.logits), loss_fn(self.logits, self.labels))

    def test_a_weight_scales_each_bags_loss(self):
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        weighted = Objective(self.labels, None, torch.tensor([2.0, 0.0])).loss(loss_fn, self.logits)
        assert torch.allclose(weighted, loss_fn(self.logits, self.labels) * torch.tensor([2.0, 0.0]))

    def test_pseudo_bags_repeat_the_slides_target(self):
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        target = torch.tensor([[0.2, 0.8, 0.0]])
        rows = torch.randn(4, 3)
        repeated = Objective(torch.tensor([1]), target).loss(loss_fn, rows, repeat=4)
        assert torch.allclose(repeated, torch.nn.functional.cross_entropy(rows, target.repeat(4, 1), reduction="none"))

    def test_a_non_cross_entropy_loss_refuses_a_target(self):
        with pytest.raises(TypeError, match="cross-entropy"):
            Objective(self.labels, None, torch.ones(2)).loss(torch.nn.NLLLoss(), self.logits)
