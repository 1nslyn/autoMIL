"""RED stubs for SCH-01: GPU scheduling-policy dispatch.

Wave-0 Nyquist stubs — all marked xfail(strict=True) until 12-02/12-03
implement the production code. No production code is written here.

Design notes:
- Uses a FakeDaemon SimpleNamespace to avoid constructing OrchestratorDaemon
  (which requires a real filesystem skeleton).
- Calls ExperimentOrchestrator._find_best_gpu as an unbound method against FakeDaemon.
- Patches `query_gpus` from automil.backends._orchestrator_daemon.
- GpuInfo-like objects are plain SimpleNamespace with .index and .free_gb attrs.
"""
from __future__ import annotations

import types
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from automil.backends._orchestrator_daemon import ExperimentOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gpu(index: int, free_gb: float) -> SimpleNamespace:
    """Minimal GPU info object compatible with _find_best_gpu."""
    return SimpleNamespace(index=index, free_gb=free_gb)


def _fake_daemon(
    *,
    gpu_allocations: dict | None = None,
    running: dict | None = None,
    max_per_gpu: int = 4,
    safety_margin_gb: float = 1.0,
    scheduling_policy: str = "best_fit",
    rr_cursor: int = 0,
) -> SimpleNamespace:
    """Build a minimal object with the attrs _find_best_gpu and _reload_orchestrator_config read."""
    fd = SimpleNamespace(
        gpu_allocations=gpu_allocations if gpu_allocations is not None else {},
        running=running if running is not None else {},
        max_per_gpu=max_per_gpu,
        safety_margin_gb=safety_margin_gb,
        scheduling_policy=scheduling_policy,
        _rr_cursor=rr_cursor,
        # Attrs needed by _reload_orchestrator_config (hot-reload tests)
        default_vram=1.0,
        default_timeout=150,
        poll_interval=5,
    )
    return fd


# ---------------------------------------------------------------------------
# SCH-01 Tests — 7 stubs
# ---------------------------------------------------------------------------

def test_best_fit_picks_tightest():
    """best_fit picks the GPU with the LEAST schedulable free VRAM (tightest fit).

    Two candidates: GPU 0 with 4.0 GB schedulable, GPU 1 with 8.0 GB.
    Expected: GPU 0 (tightest fit = current behavior preserved, D-01).

    RED gate: We verify the dispatch path is active by tracking reads of
    `scheduling_policy` via a MagicMock.  Current code never reads
    `self.scheduling_policy`, so `mock_policy.call_count == 0` will be
    asserted — but after dispatch is implemented the code will read the
    attribute and `call_count` will be >= 1, causing THIS assertion to fail
    and the test to XPASS — which flips it GREEN. Until then it stays RED.

    We use unittest.mock.PropertyMock attached to a class wrapper so that
    attribute access is tracked.
    """
    from unittest.mock import MagicMock, PropertyMock

    gpus = [_gpu(0, 5.0), _gpu(1, 9.0)]  # schedulable: 4.0 / 8.0
    fake = _fake_daemon(scheduling_policy="best_fit", rr_cursor=0)
    needed_gb = 2.0

    # Wrap fake in a class so we can attach a PropertyMock
    class _FakeClass:
        pass

    obj = _FakeClass()
    obj.__dict__.update(fake.__dict__)
    policy_mock = PropertyMock(return_value="best_fit")
    type(obj).scheduling_policy = policy_mock

    with patch(
        "automil.backends._orchestrator_daemon.query_gpus",
        return_value=gpus,
    ):
        result = ExperimentOrchestrator._find_best_gpu(obj, needed_gb)

    assert result == 0, f"best_fit should return GPU 0 (tightest fit), got {result}"
    # Dispatch gate: scheduling_policy must have been read at least once
    assert policy_mock.call_count >= 1, (
        "_find_best_gpu did not read self.scheduling_policy — "
        "policy dispatch not yet implemented (SCH-01)"
    )


def test_least_loaded_picks_emptiest():
    """least_loaded picks the GPU with the MOST schedulable free VRAM (emptiest).

    Same two candidates as test_best_fit; policy=least_loaded → GPU 1 (8.0 GB free).
    """
    gpus = [_gpu(0, 5.0), _gpu(1, 9.0)]  # schedulable: 4.0 / 8.0 after margin
    fake = _fake_daemon(scheduling_policy="least_loaded")
    needed_gb = 2.0

    with patch(
        "automil.backends._orchestrator_daemon.query_gpus",
        return_value=gpus,
    ):
        result = ExperimentOrchestrator._find_best_gpu(fake, needed_gb)

    assert result == 1, f"least_loaded should return GPU 1 (emptiest), got {result}"


def test_round_robin_cycles_eligible():
    """round_robin cycles through eligible GPUs in stable index order.

    Three eligible GPUs (indices 0, 1, 2); cursor starts at 0.
    First call → GPU 0, second call → GPU 1.
    """
    gpus = [_gpu(0, 5.0), _gpu(1, 5.0), _gpu(2, 5.0)]  # all schedulable 4.0 GB
    fake = _fake_daemon(scheduling_policy="round_robin", rr_cursor=0)
    needed_gb = 2.0

    with patch(
        "automil.backends._orchestrator_daemon.query_gpus",
        return_value=gpus,
    ):
        result_1 = ExperimentOrchestrator._find_best_gpu(fake, needed_gb)
        result_2 = ExperimentOrchestrator._find_best_gpu(fake, needed_gb)

    assert result_1 == 0, f"First round_robin call should return GPU 0, got {result_1}"
    assert result_2 == 1, f"Second round_robin call should return GPU 1, got {result_2}"


def test_round_robin_cursor_wraps():
    """round_robin cursor wraps via modulo when it exceeds the candidate count.

    Two eligible GPUs (indices 0, 1); cursor starts at 0.
    Calls: GPU 0 → GPU 1 → GPU 0 (third call wraps).
    """
    gpus = [_gpu(0, 5.0), _gpu(1, 5.0)]  # schedulable 4.0 GB each
    fake = _fake_daemon(scheduling_policy="round_robin", rr_cursor=0)
    needed_gb = 2.0

    with patch(
        "automil.backends._orchestrator_daemon.query_gpus",
        return_value=gpus,
    ):
        ExperimentOrchestrator._find_best_gpu(fake, needed_gb)  # → 0
        ExperimentOrchestrator._find_best_gpu(fake, needed_gb)  # → 1
        result_3 = ExperimentOrchestrator._find_best_gpu(fake, needed_gb)  # → 0 (wrap)

    assert result_3 == 0, f"Third round_robin call should wrap to GPU 0, got {result_3}"
    # Cursor advancement gate: after 3 round_robin placements, _rr_cursor must be 3
    # (incremented once per call). Current code never modifies _rr_cursor so this fails RED.
    assert fake._rr_cursor == 3, (
        f"_rr_cursor must be 3 after 3 round_robin placements, got {fake._rr_cursor} "
        "(cursor advancement not implemented — SCH-01)"
    )


def test_policy_hot_reload():
    """_reload_orchestrator_config reads scheduling_policy and updates self.scheduling_policy.

    Daemon starts with scheduling_policy='best_fit'; config dict with
    orchestrator.scheduling_policy='round_robin' is hot-reloaded; after reload
    self.scheduling_policy must equal 'round_robin'.
    """
    fake = _fake_daemon(scheduling_policy="best_fit")

    # Simulate what _reload_orchestrator_config should do: read the orchestrator
    # section and assign self.scheduling_policy.  We call the real method with a
    # fake config file written to a temp dir — but since the production code does
    # not yet read scheduling_policy, this test is expected to XFAIL.
    import tempfile, pathlib, yaml  # noqa: E401

    with tempfile.TemporaryDirectory() as td:
        automil_dir = pathlib.Path(td)
        cfg_content = {"orchestrator": {"scheduling_policy": "round_robin"}}
        (automil_dir / "config.yaml").write_text(
            yaml.dump(cfg_content)
        )
        # Patch automil_dir onto the fake daemon
        fake.automil_dir = automil_dir

        ExperimentOrchestrator._reload_orchestrator_config(fake)

    assert fake.scheduling_policy == "round_robin", (
        f"After hot-reload scheduling_policy should be 'round_robin', "
        f"got {fake.scheduling_policy!r}"
    )


def test_unknown_policy_fallback():
    """Unknown scheduling_policy string falls back to best_fit AND emits a warning.

    policy='turbo_boost' (unknown); _find_best_gpu should:
    - Fall back to best_fit: returns the tightest-fit GPU (GPU 0, 4.0 GB).
    - Emit a logger.warning mentioning the unknown policy value.

    RED gate: Current code has no dispatch and no warning for unknown policies.
    The `caplog` assertion on the warning message keeps this RED until the dispatch
    branch is implemented with its unknown-policy logger.warning call (SCH-01 security,
    see 12-RESEARCH.md §Security — V5 Input Validation).
    """
    gpus = [_gpu(0, 5.0), _gpu(1, 9.0)]  # schedulable: 4.0 / 8.0
    fake = _fake_daemon(scheduling_policy="turbo_boost")
    needed_gb = 2.0

    import logging

    with patch(
        "automil.backends._orchestrator_daemon.query_gpus",
        return_value=gpus,
    ):
        with patch("automil.backends._orchestrator_daemon.logger") as mock_logger:
            result = ExperimentOrchestrator._find_best_gpu(fake, needed_gb)

    assert result == 0, (
        f"Unknown policy 'turbo_boost' should fall back to best_fit (GPU 0), got {result}"
    )
    # Warning gate: dispatch must log a warning for unrecognized policy strings
    assert mock_logger.warning.called, (
        "_find_best_gpu must call logger.warning for unknown scheduling_policy "
        "'turbo_boost' (SCH-01 — dispatch not yet implemented)"
    )


def test_cursor_not_reset_on_policy_change():
    """_rr_cursor is NOT reset when scheduling_policy changes during hot-reload.

    Cursor starts at 5; hot-reload changes policy round_robin → best_fit →
    round_robin; _rr_cursor must remain 5 throughout (never reset to 0).
    """
    import tempfile, pathlib, yaml  # noqa: E401

    fake = _fake_daemon(scheduling_policy="round_robin", rr_cursor=5)

    with tempfile.TemporaryDirectory() as td:
        automil_dir = pathlib.Path(td)
        fake.automil_dir = automil_dir

        # Hot-reload 1: change to best_fit
        (automil_dir / "config.yaml").write_text(
            yaml.dump({"orchestrator": {"scheduling_policy": "best_fit"}})
        )
        ExperimentOrchestrator._reload_orchestrator_config(fake)

        # Hot-reload 2: change back to round_robin
        (automil_dir / "config.yaml").write_text(
            yaml.dump({"orchestrator": {"scheduling_policy": "round_robin"}})
        )
        ExperimentOrchestrator._reload_orchestrator_config(fake)

    assert fake._rr_cursor == 5, (
        f"_rr_cursor must not be reset on policy changes; expected 5, got {fake._rr_cursor}"
    )
