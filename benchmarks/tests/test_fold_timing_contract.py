"""Every fold trainer reports ``elapsed_seconds`` over its WHOLE body.

autoMIL reports agent-worktime alongside eval-count, and `elapsed_seconds`
sums into `elapsed_seconds_total` per cell. That number is only meaningful if
every arm measures the same span — and they did not:

* five of seven trainers stopped the clock before the checkpoint restore and
  the final test/val evaluation, which re-reads every bag from disk;
* ``clam/survival_train.py`` reported no ``elapsed_seconds`` at all, so every
  CLAM survival fold — and every CLAM survival cell's total — read 0;
* DTFD classification was timed by its *runner* while its survival sibling
  timed itself, two mechanisms that drifted apart.

These tests pin the contract structurally (every trainer returns the key) and
behaviourally (the span really does include final evaluation).
"""

from __future__ import annotations

import ast
import inspect

import pytest

from autobench import BENCHMARKS_ROOT

#: Every fold trainer across the five arms and both task types.
_FOLD_TRAINERS = (
    "autobench.pipeline.clam.train:train_fold",
    "autobench.pipeline.clam.survival_train:train_survival_fold",
    "autobench.pipeline.abmil.train:train_abmil_fold",
    "autobench.pipeline.abmil.survival_train:train_abmil_survival_fold",
    "autobench.pipeline.dtfd.train:train_dtfd_fold",
    "autobench.pipeline.dtfd.survival_train:train_dtfd_survival_fold",
    "autobench.pipeline.nnmil.train:train_nnmil_fold",
    "autobench.pipeline.titan.train:train_titan_fold",
    "autobench.pipeline.titan.survival_train:train_titan_survival_fold",
)


def _resolve(dotted: str):
    module_path, _, attr = dotted.partition(":")
    module = __import__(module_path, fromlist=[attr])
    return getattr(module, attr)


@pytest.mark.parametrize("dotted", sorted(_FOLD_TRAINERS))
def test_fold_trainer_reports_elapsed_seconds(dotted: str) -> None:
    """Every fold trainer's returned dict carries ``elapsed_seconds``.

    ``clam/survival_train.py`` did not, and the omission was invisible: the
    runner reads it with ``result.get("elapsed_seconds", 0) or 0``, so a
    missing key silently became a reported zero.
    """
    function = _resolve(dotted)
    source = inspect.getsource(function)
    tree = ast.parse(inspect.cleandoc(source))

    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys |= {
                k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)
            }
    assert "elapsed_seconds" in keys, (
        f"{dotted} never puts 'elapsed_seconds' in a returned dict; the "
        "runner's `.get(..., 0) or 0` would silently report 0 for every fold"
    )


@pytest.mark.parametrize("dotted", sorted(_FOLD_TRAINERS))
def test_elapsed_is_computed_after_the_last_evaluation(dotted: str) -> None:
    """The clock stops at the END of the body, not at the end of the epoch loop.

    Structural proxy for the span: the statement producing ``elapsed_seconds``
    must not appear before the final evaluation. Every trainer now computes it
    inline in its result dict, which is by construction the last thing built,
    so the check is simply that no *earlier* statement binds it.
    """
    function = _resolve(dotted)
    tree = ast.parse(inspect.cleandoc(inspect.getsource(function)))

    binds: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in (
                    "elapsed_seconds", "elapsed",
                ):
                    binds.append(node.lineno)

    assert not binds, (
        f"{dotted} binds elapsed_seconds/elapsed at line(s) {binds} instead of "
        "computing it where the result dict is built. A separate binding is "
        "how five of these trainers ended up excluding the checkpoint restore "
        "and the final test/val evaluation from the measured span."
    )


def test_dtfd_runner_does_not_time_folds_itself() -> None:
    """One mechanism, not two.

    DTFD's runner used to time the classification branch while the survival
    branch timed itself, so the two task types measured different spans of the
    same arm. Both branches now report their own.
    """
    source = (
        BENCHMARKS_ROOT / "src" / "autobench" / "pipeline" / "dtfd" / "runner.py"
    ).read_text()
    assert "time.time()" not in source, (
        "dtfd/runner.py times a fold itself again; fold timing belongs to the "
        "trainers so both task types measure the same span"
    )


def test_dtfd_survival_span_includes_final_evaluation(tmp_path, monkeypatch):
    """Behavioural proof, not just structural: slow the final scoring down.

    The structural checks above say *where* the clock stops. This one proves
    *what* it covers, by making the post-loop concordance passes expensive and
    requiring the reported time to absorb that cost. Before the fix the clock
    stopped at the epoch loop, so this delay landed entirely outside it.
    """
    import time as _time

    import numpy as np
    import torch

    from autobench.pipeline.dtfd import survival_train as st
    from autobench.pipeline.dtfd.config import DTFDConfig
    from autobench.pipeline.dtfd.dataset import DTFDSurvivalSlide

    import h5py

    def _bag(path, rng, status):
        feats = rng.standard_normal((20, 64)).astype("float32") + status * 3.0
        with h5py.File(path, "w") as f:
            f.create_dataset("features", data=feats)

    rng = np.random.default_rng(5)
    bags = tmp_path / "bags"
    bags.mkdir()

    def _split(prefix, n):
        out = []
        for i in range(n):
            path = str(bags / f"{prefix}{i}.h5")
            _bag(path, rng, i % 2)
            out.append(DTFDSurvivalSlide(
                slide_id=f"{prefix}{i}", h5_path=path, status=i % 2,
                time=10.0 + 5.0 * (i % 4), patient_id=f"{prefix}{i}",
            ))
        return out

    train, val, test = _split("tr", 8), _split("va", 4), _split("te", 4)

    # Patch a TAIL-ONLY call. `_risk_records` looks like the obvious target but
    # the epoch loop's own val c-index goes through it too, so a delay there
    # lands inside the old span as well and the test passes against the bug.
    # `_restore` runs exactly once, after the loop, on the way to final scoring.
    delay = 0.25
    original = st._restore

    def _slow(*args, **kwargs):
        _time.sleep(delay)
        return original(*args, **kwargs)

    monkeypatch.setattr(st, "_restore", _slow)

    result = st.train_dtfd_survival_fold(
        train, val, test,
        embed_dim=64, nll_bins=4,
        cfg=DTFDConfig(numGroup=2, mDim=32, max_epochs=1, lr=1e-3,
                       early_stopping=False),
        device=torch.device("cpu"), seed=0,
    )

    assert result["elapsed_seconds"] >= delay, (
        f"elapsed_seconds={result['elapsed_seconds']:.3f} did not absorb a "
        f"{delay}s delay injected into the post-loop scoring, so the measured "
        "span still stops at the epoch loop"
    )


class TestSchedulerTarget:
    """A policy may return a duck-typed wrapper; a torch scheduler may not."""

    @staticmethod
    def _runtime():
        from autobench.pipeline.policy_dispatch import PolicyRuntime
        return PolicyRuntime()

    @staticmethod
    def _optimizer():
        import torch
        return torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)

    def test_replacement_optimizer_is_scheduled(self):
        """A policy that REPLACES the optimizer gets the replacement scheduled."""
        original, replacement = self._optimizer(), self._optimizer()
        target = self._runtime().scheduler_target(replacement, original)
        assert target is replacement

    def test_wrapper_falls_back_to_the_real_optimizer(self):
        """A wrapper delegates and shares param_groups, so schedule the inner one."""
        original = self._optimizer()

        class Proxy:
            def __init__(self, inner):
                self._inner = inner
                self.param_groups = inner.param_groups

            def zero_grad(self, *a, **k):
                return self._inner.zero_grad(*a, **k)

            def step(self, *a, **k):
                return self._inner.step(*a, **k)

        target = self._runtime().scheduler_target(Proxy(original), original)
        assert target is original

    def test_a_real_scheduler_accepts_what_it_returns(self):
        """End to end: MultiStepLR must actually construct from the result."""
        import torch

        original = self._optimizer()

        class Proxy:
            def zero_grad(self, *a, **k): ...
            def step(self, *a, **k): ...

        runtime = self._runtime()
        proxy = Proxy()
        with pytest.raises(TypeError, match="is not an Optimizer"):
            torch.optim.lr_scheduler.MultiStepLR(proxy, [1])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            runtime.scheduler_target(proxy, original), [1],
        )
        assert scheduler.optimizer is original

    def test_neither_real_fails_at_the_seam_naming_the_role(self):
        """No usable optimizer: report it here, not from deep inside torch."""
        class Proxy:
            def zero_grad(self, *a, **k): ...
            def step(self, *a, **k): ...

        with pytest.raises(TypeError, match="role 'tier1'"):
            self._runtime().scheduler_target(Proxy(), Proxy(), role="tier1")
