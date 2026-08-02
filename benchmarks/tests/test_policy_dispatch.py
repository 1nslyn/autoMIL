from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from autobench.pipeline.policy_dispatch import (
    PolicyRuntime,
    resolve_policy_name,
    runtime_automil_dir,
)


def test_native_runtime_is_exact_identity():
    runtime = PolicyRuntime()
    optimizer = Mock()
    scheduler = Mock()
    assert runtime.wrap_optimizer(optimizer) is optimizer
    assert runtime.wrap_scheduler(scheduler) is scheduler
    assert runtime.should_stop(False, epoch=3, metrics={"val_loss": 1.0}) is False
    assert runtime.should_stop(True, epoch=3, metrics={"val_loss": 1.0}) is True


def test_nested_runtime_automil_dir_comes_from_orchestrator_env(monkeypatch):
    monkeypatch.setenv(
        "AUTOMIL_DIR_REL", "benchmarks/experiments/ccrcc/automil",
    )
    assert runtime_automil_dir().as_posix() == (
        "benchmarks/experiments/ccrcc/automil"
    )


def test_explicit_and_archived_policy_must_agree(tmp_path):
    automil_dir = tmp_path / "automil"
    automil_dir.mkdir()
    (automil_dir / "applied_variant.json").write_text(json.dumps({
        "policy": {"variant": "archived"},
    }))
    with pytest.raises(ValueError, match="disagrees"):
        resolve_policy_name(SimpleNamespace(policy_variant="explicit"), automil_dir)


def test_runtime_loads_registered_policy_from_overlay(tmp_path):
    automil_dir = tmp_path / "automil"
    policy_dir = automil_dir / "variants" / "_policies"
    policy_dir.mkdir(parents=True)
    (policy_dir / "identity.py").write_text('''
from automil.registry import PolicyVariant, VariantSpec, register

@register(VariantSpec(
    name="identity", kind="policy", parent=None, base_commit="abc",
    composite=0.5, node_id="n1", created_at="2026-08-02T00:00:00+00:00",
))
class Identity(PolicyVariant):
    def wrap_optimizer(self, opt):
        return opt
    def should_stop(self, *, default, epoch, metrics):
        return epoch >= 7
''')
    runtime = PolicyRuntime.from_experiment(
        SimpleNamespace(policy_variant="identity"), automil_dir,
    )
    optimizer = Mock()
    optimizer.zero_grad = Mock()
    optimizer.step = Mock()
    assert runtime.wrap_optimizer(optimizer, role="tier1") is optimizer
    assert runtime.should_stop(False, epoch=6) is False
    assert runtime.should_stop(False, epoch=7) is True


def test_runtime_loads_policy_from_nested_project_env(tmp_path, monkeypatch):
    automil_dir = tmp_path / "benchmarks" / "experiments" / "ccrcc" / "automil"
    policy_dir = automil_dir / "variants" / "_policies"
    policy_dir.mkdir(parents=True)
    (policy_dir / "nested_identity.py").write_text('''
from automil.registry import PolicyVariant, VariantSpec, register
@register(VariantSpec(
    name="nested_identity", kind="policy", parent=None, base_commit="abc",
    composite=0.5, node_id="n2", created_at="2026-08-02T00:00:00+00:00",
))
class NestedIdentity(PolicyVariant):
    def wrap_optimizer(self, opt):
        return opt
''')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "AUTOMIL_DIR_REL", "benchmarks/experiments/ccrcc/automil",
    )
    runtime = PolicyRuntime.from_experiment(
        SimpleNamespace(policy_variant="nested_identity"),
    )
    assert runtime.name == "nested_identity"


def test_runtime_fails_loudly_for_missing_policy(tmp_path):
    automil_dir = tmp_path / "automil"
    (automil_dir / "variants" / "_policies").mkdir(parents=True)
    with pytest.raises(ValueError, match="was not registered"):
        PolicyRuntime.from_experiment(
            SimpleNamespace(policy_variant="missing"), automil_dir,
        )


def test_runtime_rejects_non_boolean_stop_decision():
    policy = Mock()
    policy.should_stop.return_value = 1
    runtime = PolicyRuntime(name="bad", policy=policy)
    with pytest.raises(TypeError, match="must return bool"):
        runtime.should_stop(False, epoch=1)
