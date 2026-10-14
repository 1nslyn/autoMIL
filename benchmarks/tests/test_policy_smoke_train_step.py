"""The policy smoke harness drives a policy's own ``train_step`` on each arm's batch.

A classification trainer trains every batch through the runtime's
``train_step``. A policy that writes its own must survive the arm's batch
(one bag; a CLAM bag; TITAN's batch of slide vectors; nnMIL's padded batch
through a GradScaler; both DTFD tiers on one bag) before an attempt is
charged, train reproducibly from the generator it is handed, and stay off the
survival cells, whose adapters never call it.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from autobench.pipeline import policy_smoke  # noqa: E402
from tests.test_policy_smoke import HEADER, _isolated_registry, _main, _write  # noqa: E402, F401


def _policy(name: str, body: str) -> str:
    return HEADER.format(name=name) + f'''class Stepper(PolicyVariant):
    def wrap_optimizer(self, opt):
        self.params = [p for group in opt.param_groups for p in group["params"]]
        return opt

    def train_step(self, step):
''' + "".join(f"        {line}\n" for line in body.strip().splitlines())


#: Sharpness-aware minimization: ascend to the worst nearby weights, take the
#: gradient there, come back, step. Two passes, one step; scale-free in the
#: gradient norm, so the nnMIL loss scale needs no unscaling.
SAM = _policy("sam", """
import torch
step.zero_grad()
step.backward_on()
grads = [p.grad for p in self.params if p.grad is not None]
norm = torch.norm(torch.stack([g.norm() for g in grads])) + 1e-12
shifts = []
with torch.no_grad():
    for p in self.params:
        shift = torch.zeros_like(p) if p.grad is None else 0.05 * p.grad / norm
        p.add_(shift)
        shifts.append(shift)
step.zero_grad()
step.backward_on()
with torch.no_grad():
    for p, shift in zip(self.params, shifts):
        p.sub_(shift)
step.step()
""")

#: Label smoothing as a probability target per bag, on every arm's batch size.
SMOOTHED = _policy("smoothed", """
import torch
hard = torch.nn.functional.one_hot(step.labels, step.n_classes).float()
step.zero_grad()
step.backward_on(target=0.9 * hard + 0.1 / step.n_classes, weight=1.0)
step.step()
""")

#: Gradient accumulation: one step every second batch.
ACCUMULATE = _policy("accumulate", """
self.count = getattr(self, "count", 0) + 1
if self.count % 2 == 1:
    step.zero_grad()
step.backward_on(weight=0.5)
if self.count % 2 == 0:
    step.step()
""")

#: A random subset of the bag, drawn from the generator the step is handed.
OWN_GENERATOR = _policy("own_generator", """
import torch
step.zero_grad()
keep = torch.rand(step.features.shape[0], generator=step.generator) < 2.0
step.backward_on(step.features[keep])
step.step()
""")

GLOBAL_RNG = _policy("global_rng", """
import torch
step.zero_grad()
step.backward_on(weight=float(torch.rand(1)) + 0.5)
step.step()
""")

#: A second pass on half the bag: legal on the batch arms, refused by DTFD,
#: whose pseudo-bag split is drawn once per slide.
SMALLER_SECOND_PASS = _policy("smaller_second_pass", """
step.zero_grad()
step.backward_on()
step.backward_on(step.features[: step.features.shape[0] // 2])
step.step()
""")

THREE_PASSES = _policy("three_passes", """
step.zero_grad()
for _ in range(3):
    step.backward_on()
step.step()
""")

ONE_CLASS_PER_BAG = _policy("one_class", """
step.zero_grad()
step.backward_on(target=1)
step.step()
""")


@pytest.mark.parametrize("source", [SAM, SMOOTHED, ACCUMULATE, OWN_GENERATOR],
                         ids=["sam", "label-smoothing", "accumulation", "own-generator"])
@pytest.mark.parametrize("arm", [None, *policy_smoke.ARMS])
def test_a_legitimate_training_step_passes_on_every_arm(tmp_path, capsys, source, arm):
    path = str(_write(tmp_path, "stepper", source))
    assert _main([path] + (["--arm", arm] if arm else [])) == 0, capsys.readouterr().err


@pytest.mark.parametrize("source, message", [
    (GLOBAL_RNG, "global RNG"),
    (THREE_PASSES, "more than 2 passes"),
])
def test_a_step_that_breaks_the_rules_is_refused(tmp_path, capsys, source, message):
    assert _main([str(_write(tmp_path, "stepper", source))]) == 1
    assert message in capsys.readouterr().err


def test_a_single_bag_target_fails_on_the_batch_arms_only(tmp_path, capsys):
    path = str(_write(tmp_path, "stepper", ONE_CLASS_PER_BAG))
    for arm in ("abmil", "clam", "dtfd"):
        assert _main(["--arm", arm, path]) == 0
    for arm in ("titan", "nnmil"):
        assert _main(["--arm", arm, path]) == 1
        assert "target" in capsys.readouterr().err


def test_a_survival_cell_refuses_a_policy_that_steps_itself(tmp_path, capsys):
    path = str(_write(tmp_path, "stepper", SMOOTHED))
    assert _main(["--task-family", "survival", path]) == 1
    assert "train_step() is not wired for survival trainers" in capsys.readouterr().err


def test_the_step_check_runs_each_arms_batch_once_per_shape():
    labels = [label for label, _ in policy_smoke._step_checks(object, None, "classification")]
    assert labels == [f"train_step on {what}" for what, _, _ in policy_smoke.STEP_CASES.values()]


def test_dtfd_refuses_a_second_pass_on_a_bag_of_another_size(tmp_path, capsys):
    path = str(_write(tmp_path, "stepper", SMALLER_SECOND_PASS))
    for arm in ("abmil", "clam"):
        assert _main(["--arm", arm, path]) == 0
    assert _main(["--arm", "dtfd", path]) == 1
    assert "first pass's size" in capsys.readouterr().err
