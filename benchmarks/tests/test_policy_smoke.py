"""The policy smoke harness: a policy file must survive the trainers' seams
before an attempt is charged for it.

Two of the five rehearsal cells lost an attempt (plus a retry) to policy
files that crashed on the seam's call order: the ABMIL trainer calls
``zero_grad`` between the forward and the backward pass, so a policy that
restored weights in place inside ``zero_grad`` tripped autograd's version
check. The harness runs a policy through a tiny model under each of the
three call orders the trainers use, through the same ``PolicyRuntime`` the
trainers use, plus the scheduler and stopping seams.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from autobench.pipeline import policy_smoke  # noqa: E402

HEADER = '''from automil.registry import PolicyVariant, VariantSpec, register

@register(VariantSpec(
    name="{name}", kind="policy", parent=None, base_commit="abc1234",
    primary_value=0.0, node_id="node_0001", created_at="2026-01-01T00:00:00+00:00",
))
'''

IDENTITY = HEADER.format(name="identity") + '''class Identity(PolicyVariant):
    def wrap_optimizer(self, opt):
        return opt
'''

LOOKAHEAD = HEADER.format(name="lookahead") + '''class Lookahead(PolicyVariant):
    """Slow weights updated every k steps: a legitimate single-point wrapper."""

    def wrap_optimizer(self, opt):
        class _Wrapped:
            def __init__(self, inner):
                self.inner = inner
                self.k = 0
                self.slow = [p.detach().clone() for g in inner.param_groups for p in g["params"]]

            @property
            def param_groups(self):
                return self.inner.param_groups

            def zero_grad(self, *a, **kw):
                self.inner.zero_grad(*a, **kw)

            def step(self, *a, **kw):
                import torch
                self.inner.step(*a, **kw)
                self.k += 1
                if self.k % 2 == 0:
                    with torch.no_grad():
                        params = [p for g in self.inner.param_groups for p in g["params"]]
                        for slow, fast in zip(self.slow, params):
                            slow.add_(0.5 * (fast - slow))
                            fast.copy_(slow)

            def state_dict(self):
                return self.inner.state_dict()

        return _Wrapped(opt)
'''

RESTORE_IN_ZERO_GRAD = HEADER.format(name="ema_in_place") + '''class EmaInPlace(PolicyVariant):
    """The rehearsal crash: weights restored IN PLACE inside zero_grad, which
    the ABMIL trainer calls between the forward and the backward pass."""

    def wrap_optimizer(self, opt):
        class _Wrapped:
            def __init__(self, inner):
                self.inner = inner
                self.raw = [p.detach().clone() for g in inner.param_groups for p in g["params"]]

            @property
            def param_groups(self):
                return self.inner.param_groups

            def zero_grad(self, *a, **kw):
                import torch
                with torch.no_grad():
                    params = [p for g in self.inner.param_groups for p in g["params"]]
                    for raw, p in zip(self.raw, params):
                        p.copy_(raw)
                self.inner.zero_grad(*a, **kw)

            def step(self, *a, **kw):
                self.inner.step(*a, **kw)

        return _Wrapped(opt)
'''

RETURNS_NONE = HEADER.format(name="returns_none") + '''class ReturnsNone(PolicyVariant):
    def wrap_optimizer(self, opt):
        return None
'''

NON_BOOL_STOP = HEADER.format(name="non_bool_stop") + '''class NonBoolStop(PolicyVariant):
    def wrap_optimizer(self, opt):
        return opt

    def should_stop(self, *, default, epoch, metrics):
        return "no"
'''

NO_POLICY = '''"""Not a policy module at all."""
X = 1
'''

READS_C_INDEX = HEADER.format(name="c_index_stop") + '''class CIndexStop(PolicyVariant):
    """A survival stopping rule: legal on a survival cell, whose trainers pass
    val_c_index and val_loss to should_stop."""

    def wrap_optimizer(self, opt):
        return opt

    def should_stop(self, *, default, epoch, metrics):
        return bool(default) or metrics["val_c_index"] > 0.99
'''

READS_AUC = HEADER.format(name="auc_stop") + '''class AucStop(PolicyVariant):
    def wrap_optimizer(self, opt):
        return opt

    def should_stop(self, *, default, epoch, metrics):
        return bool(default) or metrics["val_auc"] > 0.99
'''

READS_MILESTONES = HEADER.format(name="milestones") + '''class Milestones(PolicyVariant):
    """A scheduler tweak written against the scheduler DTFD really passes
    (MultiStepLR, one per tier)."""

    def wrap_optimizer(self, opt):
        return opt

    def wrap_scheduler(self, sched):
        sched.milestones = type(sched.milestones)({m + 1: 1 for m in sched.milestones})
        return sched
'''

NO_PARAM_GROUPS = HEADER.format(name="no_param_groups") + '''class NoParamGroups(PolicyVariant):
    """A wrapper that delegates only zero_grad and step: nnMIL reads
    optimizer.param_groups[0]["lr"] every epoch and would crash at once."""

    def wrap_optimizer(self, opt):
        class _Wrapped:
            def __init__(self, inner):
                self.inner = inner

            def zero_grad(self, *a, **kw):
                self.inner.zero_grad(*a, **kw)

            def step(self, *a, **kw):
                self.inner.step(*a, **kw)

        return _Wrapped(opt)
'''

SHARED_SLOW = HEADER.format(name="shared_slow") + '''class SharedSlow(PolicyVariant):
    """Lookahead with the slow weights kept on the POLICY, not the wrapper:
    DTFD wraps both tiers with one policy instance before training, so the
    second wrap overwrites the first tier's buffers."""

    def wrap_optimizer(self, opt):
        import torch
        policy = self
        policy.slow = [p.detach().clone() for g in opt.param_groups for p in g["params"]]

        class _Wrapped:
            def __init__(self, inner):
                self.inner = inner

            @property
            def param_groups(self):
                return self.inner.param_groups

            def zero_grad(self, *a, **kw):
                self.inner.zero_grad(*a, **kw)

            def step(self, *a, **kw):
                self.inner.step(*a, **kw)
                with torch.no_grad():
                    params = [p for g in self.inner.param_groups for p in g["params"]]
                    for slow, fast in zip(policy.slow, params):
                        slow.add_(0.5 * (fast - slow))
                        fast.copy_(slow)

        return _Wrapped(opt)
'''

COPIES_PARAM_GROUPS = HEADER.format(name="copies_param_groups") + '''class CopiesParamGroups(PolicyVariant):
    """Delegates zero_grad and step but hands out COPIES of param_groups: the
    plain path never notices; nnMIL's GradScaler unscales the copies, finds
    no gradients on them and refuses the step."""

    def wrap_optimizer(self, opt):
        import copy

        class _Wrapped:
            def __init__(self, inner):
                self.inner = inner

            @property
            def param_groups(self):
                return copy.deepcopy(self.inner.param_groups)

            def zero_grad(self, *a, **kw):
                self.inner.zero_grad(*a, **kw)

            def step(self, *a, **kw):
                self.inner.step(*a, **kw)

        return _Wrapped(opt)
'''

CAPTURES_IN_SCHEDULER = HEADER.format(name="captures_in_scheduler") + '''class CapturesInScheduler(PolicyVariant):
    """Buffers made per optimizer wrap, captured per scheduler wrap: DTFD
    wraps BOTH optimizers before EITHER scheduler, so tier 1's scheduler
    captures tier 2's buffers and its first step mismatches shapes."""

    def wrap_optimizer(self, opt):
        self.buffers = [p.detach().clone() for g in opt.param_groups for p in g["params"]]
        return opt

    def wrap_scheduler(self, sched):
        import torch
        captured = self.buffers

        class _Wrapped:
            def __init__(self, inner):
                self.inner = inner

            def step(self, *a, **kw):
                self.inner.step(*a, **kw)
                params = [p for g in self.inner.optimizer.param_groups for p in g["params"]]
                with torch.no_grad():
                    for buffer, p in zip(captured, params, strict=True):
                        buffer.add_(p - buffer)

        return _Wrapped(sched)
'''

FORGETS_TIER2 = HEADER.format(name="forgets_tier2") + '''class ForgetsTier2(PolicyVariant):
    def wrap_optimizer(self, opt):
        return opt

    def wrap_scheduler_for(self, sched, *, role):
        if role == "tier1":
            return sched
        return None      # tier2 falls through: DTFD would crash before training
'''


@pytest.fixture(autouse=True)
def _isolated_registry():
    from automil.registry._state import _clear_registry
    _clear_registry()
    yield
    _clear_registry()


def _write(tmp_path, name, source):
    path = tmp_path / f"{name}.py"
    path.write_text(source)
    return path


def _main(argv):
    """One harness run on a fresh registry: production runs one policy per
    subprocess, a test that judges one file twice must not trip the duplicate
    registration."""
    from automil.registry._state import _clear_registry
    _clear_registry()
    return policy_smoke.main(argv)


class TestHarnessVerdicts:
    def test_an_identity_policy_passes(self, tmp_path, capsys):
        assert policy_smoke.main([str(_write(tmp_path, "identity", IDENTITY))]) == 0
        assert "passed" in capsys.readouterr().out

    def test_a_lookahead_style_wrapper_passes(self, tmp_path):
        assert policy_smoke.main([str(_write(tmp_path, "lookahead", LOOKAHEAD))]) == 0

    def test_the_rehearsal_crash_is_caught(self, tmp_path, capsys):
        code = policy_smoke.main([str(_write(tmp_path, "ema_in_place", RESTORE_IN_ZERO_GRAD))])
        assert code == 1
        err = capsys.readouterr().err
        assert "forward -> zero_grad -> backward -> step" in err
        assert "inplace" in err.lower() or "in-place" in err.lower() or "modified" in err.lower()

    def test_a_wrapper_that_returns_none_is_caught(self, tmp_path, capsys):
        assert policy_smoke.main([str(_write(tmp_path, "returns_none", RETURNS_NONE))]) == 1
        assert "returned None for optimizer role" in capsys.readouterr().err

    def test_a_non_bool_stop_decision_is_caught(self, tmp_path, capsys):
        assert policy_smoke.main([str(_write(tmp_path, "non_bool_stop", NON_BOOL_STOP))]) == 1
        assert "should_stop" in capsys.readouterr().err

    def test_a_module_without_a_policy_is_a_usage_error(self, tmp_path, capsys):
        assert policy_smoke.main([str(_write(tmp_path, "no_policy", NO_POLICY))]) == 2
        assert "PolicyVariant" in capsys.readouterr().err

    def test_a_missing_file_is_a_usage_error(self, tmp_path):
        assert policy_smoke.main([str(tmp_path / "absent.py")]) == 2


class TestTheStoppingSeamIsJudgedByTaskFamily:
    def test_a_survival_stopping_rule_passes_on_a_survival_cell(self, tmp_path, capsys):
        path = str(_write(tmp_path, "c_index_stop", READS_C_INDEX))
        assert _main(["--task-family", "survival", path]) == 0
        assert _main([path]) == 1
        assert "val_c_index" in capsys.readouterr().err

    def test_a_classification_stopping_rule_passes_on_a_classification_cell(self, tmp_path, capsys):
        path = str(_write(tmp_path, "auc_stop", READS_AUC))
        assert _main([path]) == 0
        assert _main(["--task-family", "survival", path]) == 1
        assert "val_auc" in capsys.readouterr().err

    def test_an_unknown_task_family_is_a_usage_error(self, tmp_path):
        path = str(_write(tmp_path, "auc_stop", READS_AUC))
        assert policy_smoke.main(["--task-family", "regression", path]) == 2


class TestTheOptimizerSeamIsTheTrainers:
    def test_a_wrapper_without_param_groups_is_refused(self, tmp_path, capsys):
        assert policy_smoke.main([str(_write(tmp_path, "no_param_groups", NO_PARAM_GROUPS))]) == 1
        assert "param_groups" in capsys.readouterr().err

    def test_per_wrap_state_kept_on_the_policy_is_refused(self, tmp_path, capsys):
        assert policy_smoke.main([str(_write(tmp_path, "shared_slow", SHARED_SLOW))]) == 1
        assert "tier" in capsys.readouterr().err

    def test_per_wrapper_state_passes(self, tmp_path):
        assert policy_smoke.main([str(_write(tmp_path, "lookahead", LOOKAHEAD))]) == 0

    def test_a_wrapper_handing_the_scaler_copies_is_refused(self, tmp_path, capsys):
        assert policy_smoke.main([str(_write(tmp_path, "copies_param_groups", COPIES_PARAM_GROUPS))]) == 1
        err = capsys.readouterr().err
        assert "GradScaler" in err and "inf checks" in err


class TestTheSchedulerSeamIsDTFDs:
    def test_a_multistep_milestone_tweak_passes(self, tmp_path):
        assert policy_smoke.main([str(_write(tmp_path, "milestones", READS_MILESTONES))]) == 0

    def test_buffers_captured_per_scheduler_wrap_are_refused(self, tmp_path, capsys):
        """Passes when each tier's optimizer and scheduler are wrapped
        together; refused under DTFD's real order."""
        assert policy_smoke.main([str(_write(tmp_path, "captures", CAPTURES_IN_SCHEDULER))]) == 1
        assert "tier" in capsys.readouterr().err

    def test_a_wrapper_that_forgets_tier2_is_refused_by_name(self, tmp_path, capsys):
        assert policy_smoke.main([str(_write(tmp_path, "forgets_tier2", FORGETS_TIER2))]) == 1
        err = capsys.readouterr().err
        assert "tier2" in err and "scheduler" in err


class TestHarnessCoverage:
    @pytest.mark.parametrize("family, stop_keys", [
        ("classification", "val_auc,val_loss"), ("survival", "val_c_index,val_loss"),
    ])
    def test_every_call_order_and_seam_is_exercised(self, tmp_path, family, stop_keys):
        """A policy that records what the harness did: all three call orders,
        the stopping seam with the family's metrics, and the DTFD tiers each
        with their MultiStepLR scheduler."""
        recorder = tmp_path / "record.txt"
        source = HEADER.format(name="recorder") + f'''class Recorder(PolicyVariant):
    def wrap_optimizer_for(self, opt, *, role):
        with open({str(recorder)!r}, "a") as fh:
            fh.write("optimizer:" + role + "\\n")
        return opt

    def wrap_optimizer(self, opt):
        return opt

    def wrap_scheduler_for(self, sched, *, role):
        with open({str(recorder)!r}, "a") as fh:
            fh.write("scheduler:" + role + ":" + type(sched).__name__ + "\\n")
        return sched

    def should_stop(self, *, default, epoch, metrics):
        with open({str(recorder)!r}, "a") as fh:
            fh.write("stop:" + ",".join(sorted(metrics)) + "\\n")
        return default
'''
        path = str(_write(tmp_path, "recorder", source))
        assert policy_smoke.main(["--task-family", family, path]) == 0
        lines = recorder.read_text().splitlines()
        # one wrap per call order, one for the scaled order, plus the stopping run
        assert lines.count("optimizer:main") == len(policy_smoke.CALL_ORDERS) + 2
        assert "optimizer:tier1" in lines and "optimizer:tier2" in lines
        assert "scheduler:tier1:MultiStepLR" in lines and "scheduler:tier2:MultiStepLR" in lines
        assert not any(line.startswith("scheduler:main") for line in lines)
        # DTFD wraps both tiers, then builds both schedulers, before any step
        dtfd = [line for line in lines if line.endswith(":tier1") or line.endswith(":tier2")
                or ":tier1:" in line or ":tier2:" in line]
        assert dtfd == ["optimizer:tier1", "optimizer:tier2",
                        "scheduler:tier1:MultiStepLR", "scheduler:tier2:MultiStepLR"]
        assert lines.count(f"stop:{stop_keys}") == policy_smoke.STEPS_PER_ORDER
