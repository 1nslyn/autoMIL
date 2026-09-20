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

READS_BACC = HEADER.format(name="bacc_stop") + '''class BaccStop(PolicyVariant):
    """nnMIL passes val_bacc and val_f1 beside val_auc; ABMIL does not."""

    def wrap_optimizer(self, opt):
        return opt

    def should_stop(self, *, default, epoch, metrics):
        return bool(default) or metrics["val_bacc"] > 0.99
'''

HOLDS_SCHEDULER = HEADER.format(name="holds_scheduler") + '''class HoldsScheduler(PolicyVariant):
    """A DTFD policy keeps its scheduler and reads the learning rate when
    asked to stop: legal on DTFD, where the schedulers exist before the
    first stopping call."""

    def wrap_optimizer(self, opt):
        return opt

    def wrap_scheduler(self, sched):
        self.scheduler = sched
        return sched

    def should_stop(self, *, default, epoch, metrics):
        return bool(default) or self.scheduler.get_last_lr()[0] < 1e-9
'''

STATEFUL_FROM_EPOCH_ZERO = HEADER.format(name="stateful_from_zero") + '''class StatefulFromZero(PolicyVariant):
    """Initializes at epoch 0 and compares afterwards: nnMIL's survival
    trainers ask from epoch 2, so the state never exists there."""

    def wrap_optimizer(self, opt):
        return opt

    def should_stop(self, *, default, epoch, metrics):
        if epoch == 0:
            self.previous = metrics["val_loss"]
            return bool(default)
        stalled = metrics["val_loss"] >= self.previous
        self.previous = metrics["val_loss"]
        return bool(default) or (stalled and epoch > 50)
'''

SYNC_ARMED_BY_STOP = HEADER.format(name="sync_armed_by_stop") + '''class SyncArmedByStop(PolicyVariant):
    """Epoch-based Lookahead: should_stop arms a slow-weight sync that the
    next zero_grad performs in place. Legal where zero_grad precedes the
    forward pass; on ABMIL zero_grad sits between forward and backward, so
    the in-place sync trips autograd on the following backward."""

    def wrap_optimizer(self, opt):
        import torch
        policy = self
        policy.armed = False

        class _Wrapped:
            def __init__(self, inner):
                self.inner = inner
                self.slow = [p.detach().clone() for g in inner.param_groups for p in g["params"]]

            @property
            def param_groups(self):
                return self.inner.param_groups

            def zero_grad(self, *a, **kw):
                if policy.armed:
                    with torch.no_grad():
                        params = [p for g in self.inner.param_groups for p in g["params"]]
                        for slow, fast in zip(self.slow, params):
                            slow.add_(0.5 * (fast - slow))
                            fast.copy_(slow)
                self.inner.zero_grad(*a, **kw)

            def step(self, *a, **kw):
                self.inner.step(*a, **kw)

        return _Wrapped(opt)

    def should_stop(self, *, default, epoch, metrics):
        self.armed = True
        return bool(default)
'''

RESTORES_TIER2_IN_ZERO_GRAD = HEADER.format(name="restores_tier2") + '''class RestoresTier2(PolicyVariant):
    """The rehearsal crash aimed at one role: weights restored in place inside
    zero_grad for tier2 only, which DTFD calls between forward and backward."""

    def wrap_optimizer_for(self, opt, *, role):
        import torch
        if role != "tier2":
            return opt

        class _Wrapped:
            def __init__(self, inner):
                self.inner = inner
                self.raw = [p.detach().clone() for g in inner.param_groups for p in g["params"]]

            @property
            def param_groups(self):
                return self.inner.param_groups

            def zero_grad(self, *a, **kw):
                with torch.no_grad():
                    for raw, p in zip(self.raw, [p for g in self.inner.param_groups for p in g["params"]]):
                        p.copy_(raw)
                self.inner.zero_grad(*a, **kw)

            def step(self, *a, **kw):
                self.inner.step(*a, **kw)

        return _Wrapped(opt)

    def wrap_optimizer(self, opt):
        return opt
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
        assert _main(["--arm", "resnet", path]) == 2

    def test_the_arm_decides_which_keys_a_stopping_rule_may_read(self, tmp_path, capsys):
        path = str(_write(tmp_path, "bacc_stop", READS_BACC))
        assert _main(["--arm", "nnmil", path]) == 0
        assert _main(["--arm", "abmil", path]) == 1
        assert "val_bacc" in capsys.readouterr().err

    def test_a_dtfd_policy_may_read_its_scheduler_when_asked_to_stop(self, tmp_path, capsys):
        path = str(_write(tmp_path, "holds_scheduler", HOLDS_SCHEDULER))
        assert _main(["--arm", "dtfd", path]) == 0
        assert _main(["--arm", "abmil", path]) == 1          # no scheduler ever exists there
        assert "scheduler" in capsys.readouterr().err

    def test_the_stopping_run_uses_the_arms_own_call_order(self, tmp_path, capsys):
        path = str(_write(tmp_path, "sync_armed_by_stop", SYNC_ARMED_BY_STOP))
        assert _main(["--arm", "titan", path]) == 0          # zero_grad before forward
        assert _main(["--arm", "abmil", path]) == 1          # zero_grad between forward and backward
        err = capsys.readouterr().err
        assert "stopping seam" in err and "inplace" in err.lower() or "modified" in err.lower()
        # the ABMIL survival adapter zeroes before the forward pass: legal there
        assert _main(["--arm", "abmil", "--task-family", "survival", path]) == 0
        assert _main(["--arm", "clam", "--task-family", "survival", path]) == 0

    def test_nnmil_survival_asks_from_epoch_two(self, tmp_path, capsys):
        path = str(_write(tmp_path, "stateful_from_zero", STATEFUL_FROM_EPOCH_ZERO))
        assert _main(["--arm", "nnmil", "--task-family", "classification", path]) == 0
        assert _main(["--arm", "nnmil", "--task-family", "survival", path]) == 1
        assert "previous" in capsys.readouterr().err


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

    def test_an_in_place_restore_aimed_at_tier2_is_refused(self, tmp_path, capsys):
        assert _main(["--arm", "dtfd", str(_write(tmp_path, "restores_tier2", RESTORES_TIER2_IN_ZERO_GRAD))]) == 1
        err = capsys.readouterr().err
        assert "inplace" in err.lower() or "modified" in err.lower()

    def test_a_wrapper_that_forgets_tier2_is_refused_by_name(self, tmp_path, capsys):
        assert policy_smoke.main([str(_write(tmp_path, "forgets_tier2", FORGETS_TIER2))]) == 1
        err = capsys.readouterr().err
        assert "tier2" in err and "scheduler" in err


class TestHarnessCoverage:
    @pytest.mark.parametrize("arm, family, stop_keys", [
        (None, "classification", "val_auc,val_loss"), (None, "survival", "val_c_index,val_loss"),
        ("dtfd", "classification", "val_auc,val_loss"),
        ("nnmil", "classification", "val_auc,val_bacc,val_f1,val_loss"),
        ("clam", "classification", "val_auc,val_error,val_loss"),
    ])
    def test_every_call_order_and_seam_is_exercised(self, tmp_path, arm, family, stop_keys):
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
        argv = ["--task-family", family, path] + (["--arm", arm] if arm else [])
        assert policy_smoke.main(argv) == 0
        lines = recorder.read_text().splitlines()
        # one wrap per call order, one for the scaled order, plus the stopping
        # run (which on DTFD wraps the tiers instead of a main optimizer)
        assert lines.count("optimizer:main") == len(policy_smoke.CALL_ORDERS) + (1 if arm == "dtfd" else 2)
        dtfd = [line for line in lines if line.endswith(":tier1") or line.endswith(":tier2")
                or ":tier1:" in line or ":tier2:" in line]
        wrap_order = ["optimizer:tier1", "optimizer:tier2",
                      "scheduler:tier1:MultiStepLR", "scheduler:tier2:MultiStepLR"]
        if arm is None:
            assert dtfd == wrap_order                       # the DTFD seam alone
        elif arm == "dtfd":
            assert dtfd == wrap_order * 2                   # the seam, then the stopping run
        else:
            assert dtfd == []                               # no scheduler exists on this arm
        assert not any(line.startswith("scheduler:main") for line in lines)
        assert lines.count(f"stop:{stop_keys}") == policy_smoke.STEPS_PER_ORDER
