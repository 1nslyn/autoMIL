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


class TestHarnessCoverage:
    def test_every_call_order_and_seam_is_exercised(self, tmp_path):
        """A policy that records what the harness did: all three call orders,
        the scheduler seam, the stopping seam, and a non-main role."""
        recorder = tmp_path / "record.txt"
        source = HEADER.format(name="recorder") + f'''class Recorder(PolicyVariant):
    def wrap_optimizer_for(self, opt, *, role):
        with open({str(recorder)!r}, "a") as fh:
            fh.write("optimizer:" + role + "\\n")
        return opt

    def wrap_optimizer(self, opt):
        return opt

    def wrap_scheduler(self, sched):
        with open({str(recorder)!r}, "a") as fh:
            fh.write("scheduler\\n")
        return sched

    def should_stop(self, *, default, epoch, metrics):
        with open({str(recorder)!r}, "a") as fh:
            fh.write("stop:" + ",".join(sorted(metrics)) + "\\n")
        return default
'''
        assert policy_smoke.main([str(_write(tmp_path, "recorder", source))]) == 0
        lines = recorder.read_text().splitlines()
        # one wrap per call order, plus the scheduler-and-stopping run
        assert lines.count("optimizer:main") == len(policy_smoke.CALL_ORDERS) + 1
        assert "optimizer:tier1" in lines and "optimizer:tier2" in lines
        assert "scheduler" in lines
        assert any(line.startswith("stop:val_auc,val_loss") for line in lines)
