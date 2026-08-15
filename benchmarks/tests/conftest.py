"""Shared test fixtures for autobench tests."""

import os
import sys

import pytest

# Ensure the tests directory is importable for _helpers
sys.path.insert(0, os.path.dirname(__file__))

from _helpers import make_test_ds  # noqa: E402


@pytest.fixture
def test_ds():
    """Pytest fixture returning a standard test DatasetConfig."""
    return make_test_ds()


@pytest.fixture(autouse=True)
def _bag_cache_is_opt_in(monkeypatch):
    """No test may inherit the operator's bag-cache setting.

    ``AUTOBENCH_BAG_CACHE`` changes real behaviour: with it set, bags stop being
    re-read per epoch. Tests that assert the uncached contract -- notably
    ``test_lazy_bag_loading.py::test_bags_are_reread_every_epoch`` -- passed only
    because the variable happened to be unset in the shell running them, and
    failed outright once it was exported. On the campaign host, where it IS
    exported, that surfaced as two failures in an unrelated file with a message
    that misdiagnosed the cause.

    Also resets the module's process-global latch and warn-set. Without that,
    one test exercising a degrade path silently switches the cache off for every
    test after it, and those tests still pass -- for entirely the wrong reason.
    Tests that want the cache opt in explicitly via ``monkeypatch.setenv``.
    """
    from autobench.pipeline import bag_cache

    monkeypatch.delenv("AUTOBENCH_BAG_CACHE", raising=False)
    monkeypatch.setattr(bag_cache, "_warned", set())
    monkeypatch.setattr(bag_cache, "_disabled", False)
    monkeypatch.setattr(bag_cache, "_checked_dir", None)
