"""The policy smoke harness drives the train-only data seams a policy overrides.

``test_policy_smoke.py`` covers the optimizer, scheduler and stopping seams.
A policy that overrides ``transform_bag`` or ``before_validation`` must also
survive them before an attempt is charged: the transform on the bag its arm
hands it (CLAM keeps eight instances, TITAN keeps one vector per slide), the
hook after a wrapped optimizer has stepped, and a survival cell, whose
trainers call neither seam, must refuse a policy that relies on one.
"""
from __future__ import annotations

import textwrap
from collections import Counter

import pytest

torch = pytest.importorskip("torch")

from automil.registry._state import _clear_registry  # noqa: E402
from autobench.pipeline import policy_smoke  # noqa: E402
from tests.test_policy_smoke import (  # noqa: E402, F401
    HEADER,
    IDENTITY,
    _isolated_registry,
    _main,
    _write,
)

#: Two runs of three epochs each, per bag case the harness drives.
CALLS_PER_BAG_CASE = 2 * policy_smoke.STEPS_PER_ORDER
SEAM_CHECKS = ("transform_bag seam", "before_validation seam")

#: Instance dropout from the generator the policy is handed, and a hook that returns nothing.
CLEAN = HEADER.format(name="clean_seams") + '''class CleanSeams(PolicyVariant):
    def wrap_optimizer(self, opt):
        return opt

    def transform_bag(self, features, *, label, epoch, generator):
        import torch
        count = features.shape[0]
        if count < 16:                      # a slide vector: there is nothing to drop
            return features.clone()
        keep = torch.randperm(count, generator=generator)[: count // 2 + 1].sort().values
        return features[keep]

    def before_validation(self, *, epoch):
        self.last_epoch = epoch
'''

#: Writes every seam call to a file: what the harness really drove.
RECORDER = HEADER.format(name="seam_recorder") + '''class SeamRecorder(PolicyVariant):
    def wrap_optimizer(self, opt):
        return opt

    def _note(self, line):
        with open(LOG_PATH, "a") as handle:
            handle.write(line + "\\n")

    def transform_bag(self, features, *, label, epoch, generator):
        self._note("bag %d %d %s" % (epoch, label, "x".join(map(str, features.shape))))
        return features.clone()

    def before_validation(self, *, epoch):
        self._note("validate %d" % epoch)
'''

#: Weights averaged along the run, swapped in when the hook is called.
EMA_SWAP = HEADER.format(name="ema_swap") + '''class EmaSwap(PolicyVariant):
    def wrap_optimizer(self, opt):
        policy = self
        policy.params = [p for g in opt.param_groups for p in g["params"]]
        policy.average = [p.detach().clone() for p in policy.params]

        class _Wrapped:
            def __init__(self, inner):
                self.inner = inner

            @property
            def param_groups(self):
                return self.inner.param_groups

            def zero_grad(self, *a, **kw):
                self.inner.zero_grad(*a, **kw)

            def step(self, *a, **kw):
                import torch
                self.inner.step(*a, **kw)
                with torch.no_grad():
                    for average, p in zip(policy.average, policy.params):
                        average.mul_(0.9).add_(p, alpha=0.1)

        return _Wrapped(opt)

    def before_validation(self, *, epoch):
        import torch
        with torch.no_grad():
            for average, p in zip(self.average, self.params):
                p.copy_(average)
'''


def _transform_policy(name: str, body: str) -> str:
    """A policy whose ``transform_bag`` runs ``body``; everything else is native."""
    return (
        HEADER.format(name=name)
        + "class Probe(PolicyVariant):\n"
        + "    def wrap_optimizer(self, opt):\n        return opt\n\n"
        + "    def transform_bag(self, features, *, label, epoch, generator):\n"
        + textwrap.indent(textwrap.dedent(body), " " * 8)
    )


def _hook_policy(name: str, body: str) -> str:
    """A policy whose ``before_validation`` runs ``body``; everything else is native."""
    return (
        HEADER.format(name=name)
        + "class Probe(PolicyVariant):\n"
        + "    def wrap_optimizer(self, opt):\n        return opt\n\n"
        + "    def before_validation(self, *, epoch):\n"
        + textwrap.indent(textwrap.dedent(body), " " * 8)
    )


def _smoke(path, family="classification", arm=None) -> list[str]:
    """The harness's failures for one policy file, on a fresh registry."""
    _clear_registry()
    return policy_smoke.smoke(path, family, arm)


def _seam_labels(path, family="classification", arm=None) -> list[str]:
    """The data-seam checks the harness would run for one policy file."""
    _clear_registry()
    policy_cls = policy_smoke.load_policy_class(path)
    return [
        label for label, _ in policy_smoke._checks(policy_cls, arm, family)
        if label.startswith(SEAM_CHECKS)
    ]


class TestOnlyAnOverriddenSeamIsDriven:
    def test_a_policy_that_overrides_neither_seam_gets_neither_check(self, tmp_path):
        assert _seam_labels(_write(tmp_path, "identity", IDENTITY)) == []

    def test_overriding_the_bag_transform_adds_one_check_per_bag_case(self, tmp_path):
        path = _write(tmp_path, "transform_only", _transform_policy("transform_only", "return features\n"))
        labels = _seam_labels(path)
        assert len(labels) == len(policy_smoke.BAG_CASES)
        assert all(label.startswith("transform_bag seam") for label in labels)

    def test_an_arm_is_driven_on_its_own_bag_only(self, tmp_path):
        path = _write(tmp_path, "transform_only", _transform_policy("transform_only", "return features\n"))
        for arm in policy_smoke.ARMS:
            assert len(_seam_labels(path, arm=arm)) == 1, arm

    def test_overriding_the_hook_adds_one_check(self, tmp_path):
        path = _write(tmp_path, "hook_only", _hook_policy("hook_only", "return None\n"))
        assert _seam_labels(path) == ["before_validation seam"]

    def test_a_policy_overriding_both_gets_both(self, tmp_path):
        labels = _seam_labels(_write(tmp_path, "clean", CLEAN), arm="clam")
        assert [label.split(" seam")[0] for label in labels] == ["transform_bag", "before_validation"]


class TestTheBagTransformIsDrivenOnEachArmsBag:
    @pytest.mark.parametrize("arm", [*policy_smoke.ARMS, None])
    def test_a_clean_policy_passes_on_every_arm(self, tmp_path, capsys, arm):
        path = str(_write(tmp_path, "clean", CLEAN))
        assert _main(["--arm", arm, path] if arm else [path]) == 0, capsys.readouterr().err

    @pytest.mark.parametrize("arm, shapes", [
        ("abmil", {"16x8": CALLS_PER_BAG_CASE}),
        ("dtfd", {"16x8": CALLS_PER_BAG_CASE}),
        ("nnmil", {"16x8": CALLS_PER_BAG_CASE}),
        ("clam", {"16x8": CALLS_PER_BAG_CASE}),
        ("titan", {"1x8": CALLS_PER_BAG_CASE}),
        (None, {"16x8": 2 * CALLS_PER_BAG_CASE, "1x8": CALLS_PER_BAG_CASE}),
    ])
    def test_the_policy_sees_the_arms_bag_each_epoch_and_labels_both_classes(
        self, tmp_path, arm, shapes,
    ):
        log = tmp_path / "seams.log"
        source = RECORDER.replace("LOG_PATH", repr(str(log)))
        assert _smoke(_write(tmp_path, "seam_recorder", source), arm=arm) == []
        bags = [line.split() for line in log.read_text().splitlines() if line.startswith("bag")]
        assert Counter(shape for _, _, _, shape in bags) == shapes
        assert {epoch for _, epoch, _, _ in bags} == set(map(str, range(policy_smoke.STEPS_PER_ORDER)))
        assert {label for _, _, label, _ in bags} == {"0", "1"}

    def test_clam_needs_eight_instances_to_survive_the_transform(self, tmp_path):
        path = _write(tmp_path, "keeps_three", _transform_policy("keeps_three", "return features[:3]\n"))
        (failure,) = _smoke(path, arm="clam")
        assert failure.startswith("[transform_bag seam") and "at least 8" in failure
        assert _smoke(path, arm="abmil") == []
        assert _smoke(path, arm="titan") == []          # one vector stays one vector
        assert len(_smoke(path)) == 1                   # no arm named: every bag case, CLAM's fails

    def test_clam_asks_for_the_instance_count_its_model_samples(self):
        from autobench.pipeline.config import ModelConfig

        assert policy_smoke.CLAM_MIN_INSTANCES == ModelConfig(model_type="clam_sb").B

    def test_titan_must_keep_its_slide_vector_shape(self, tmp_path):
        body = "import torch\nreturn torch.cat([features, features])\n"
        path = _write(tmp_path, "doubles", _transform_policy("doubles", body))
        (failure,) = _smoke(path, arm="titan")
        assert failure.startswith("[transform_bag seam") and "keep this arm's bag shape" in failure
        assert _smoke(path, arm="abmil") == []          # a longer bag is a legal bag elsewhere

    @pytest.mark.parametrize("body, message", [
        ("features.mul_(2.0)\nreturn features\n", "modified a training bag in place"),
        ("return features.tolist()\n", "must return a tensor"),
        ("return None\n", "must return a tensor"),
        ("return features.double()\n", "must return a tensor"),
        ("return features[:, :4]\n", "must return a tensor"),
        ("return features.sum(dim=0)\n", "must return a tensor"),
    ], ids=["in_place", "list", "none", "dtype", "width", "rank"])
    def test_a_bag_the_runtime_rejects_is_refused(self, tmp_path, body, message):
        path = _write(tmp_path, "bad_return", _transform_policy("bad_return", body))
        (failure,) = _smoke(path, arm="abmil")
        assert failure.startswith("[transform_bag seam") and message in failure


class TestTheBagTransformDrawsFromItsOwnGenerator:
    @pytest.mark.parametrize("draw", [
        "import torch\ntorch.rand(1)",
        "import numpy\nnumpy.random.rand()",
        "import random\nrandom.random()",
    ], ids=["torch", "numpy", "random"])
    def test_a_draw_from_a_global_rng_is_refused(self, tmp_path, draw):
        path = _write(
            tmp_path, "global_draw",
            _transform_policy("global_draw", draw + "\nreturn features.clone()\n"),
        )
        (failure,) = _smoke(path, arm="abmil")
        assert "drew from a global RNG" in failure

    def test_a_transform_that_differs_between_two_runs_is_refused(self, tmp_path):
        body = (
            "import torch\n"
            "fresh = torch.Generator()\n"
            "fresh.seed()                  # a seed from the OS: new on every run\n"
            "return features + 0.01 * torch.randn(features.shape, generator=fresh)\n"
        )
        path = _write(tmp_path, "unseeded", _transform_policy("unseeded", body))
        (failure,) = _smoke(path, arm="abmil")
        assert "not reproducible" in failure

    def test_a_policy_using_the_generator_it_is_handed_is_reproducible(self, tmp_path):
        body = (
            "import torch\n"
            "return features + 0.01 * torch.randn(features.shape, generator=generator)\n"
        )
        path = _write(tmp_path, "seeded", _transform_policy("seeded", body))
        assert _smoke(path, arm="abmil") == []


class TestTheHookRunsAfterAWrappedOptimizerHasStepped:
    def test_it_is_called_exactly_once_with_an_epoch(self, tmp_path):
        log = tmp_path / "seams.log"
        source = RECORDER.replace("LOG_PATH", repr(str(log)))
        assert _smoke(_write(tmp_path, "seam_recorder", source), arm="abmil") == []
        assert [line for line in log.read_text().splitlines() if line.startswith("validate")] == ["validate 0"]

    def test_a_weight_swap_through_the_wrapped_optimizer_passes(self, tmp_path):
        path = _write(tmp_path, "ema_swap", EMA_SWAP)
        for arm in policy_smoke.ARMS:
            assert _smoke(path, arm=arm) == [], arm

    def test_a_hook_that_returns_something_is_refused(self, tmp_path):
        path = _write(tmp_path, "returns_one", _hook_policy("returns_one", "return 1\n"))
        (failure,) = _smoke(path, arm="abmil")
        assert failure.startswith("[before_validation seam]") and "must return None" in failure

    def test_a_hook_that_raises_is_refused(self, tmp_path):
        path = _write(tmp_path, "raises", _hook_policy("raises", 'raise RuntimeError("no model")\n'))
        (failure,) = _smoke(path, arm="abmil")
        assert failure.startswith("[before_validation seam]") and "no model" in failure


class TestSurvivalTrainersCallNeitherSeam:
    @pytest.mark.parametrize("seam, source", [
        ("transform_bag", _transform_policy("survival_transform", "return features\n")),
        ("before_validation", _hook_policy("survival_hook", "return None\n")),
    ])
    def test_a_policy_overriding_the_seam_is_refused_on_a_survival_cell(self, tmp_path, seam, source):
        path = _write(tmp_path, "survival_seam", source)
        for arm in (None, *policy_smoke.ARMS):
            (failure,) = _smoke(path, "survival", arm)
            assert failure.startswith(f"[{seam} seam (survival)]"), arm
            assert "not wired for survival trainers" in failure
        assert _smoke(path, "classification") == []

    def test_both_overrides_are_named(self, tmp_path):
        failures = _smoke(_write(tmp_path, "clean", CLEAN), "survival", "nnmil")
        assert len(failures) == 2
        assert {failure.split(" seam")[0] for failure in failures} == {
            "[transform_bag", "[before_validation",
        }

    def test_the_refusal_drives_nothing(self, tmp_path):
        log = tmp_path / "seams.log"
        source = RECORDER.replace("LOG_PATH", repr(str(log)))
        assert len(_smoke(_write(tmp_path, "seam_recorder", source), "survival", "abmil")) == 2
        assert not log.exists()

    def test_a_policy_overriding_neither_seam_still_passes_a_survival_cell(self, tmp_path):
        assert _smoke(_write(tmp_path, "identity", IDENTITY), "survival", "abmil") == []
