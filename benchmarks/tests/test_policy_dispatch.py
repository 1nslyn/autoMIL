from __future__ import annotations

import json
import random
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
    assert runtime.for_fold() is runtime


def test_each_fold_gets_an_independent_policy_instance():
    class StatefulPolicy:
        def __init__(self):
            self.epochs = []

        def wrap_optimizer_for(self, opt, *, role):
            return opt

    template = PolicyRuntime(name="stateful", policy=StatefulPolicy())
    fold_zero = template.for_fold()
    fold_one = template.for_fold()
    optimizer = Mock(zero_grad=Mock(), step=Mock())
    fold_zero.wrap_optimizer(optimizer)
    fold_one.wrap_optimizer(optimizer)
    fold_zero.policy.epochs.append(3)

    assert fold_zero.policy is not fold_one.policy
    assert fold_one.policy.epochs == []
    assert template.policy.epochs == []


def test_fold_policy_construction_is_lazy_and_seed_deterministic():
    class RandomizedPolicy:
        def __init__(self):
            self.draw = random.random()

        def wrap_optimizer_for(self, opt, *, role):
            return opt

    template = PolicyRuntime(name="randomized", policy=RandomizedPolicy())
    fold_zero = template.for_fold()
    fold_one = template.for_fold()
    assert fold_zero.policy is None
    assert fold_one.policy is None

    optimizer = Mock(zero_grad=Mock(), step=Mock())
    random.seed(17)
    fold_zero.wrap_optimizer(optimizer)
    random.seed(17)
    fold_one.wrap_optimizer(optimizer)
    assert fold_zero.policy is not fold_one.policy
    assert fold_zero.policy.draw == fold_one.policy.draw


def test_each_fold_gets_an_isolated_policy_class_namespace():
    class ClassStatePolicy:
        COUNT = 0

        def wrap_optimizer_for(self, opt, *, role):
            type(self).COUNT += 1
            return opt

    template = PolicyRuntime(name="class-state", policy=ClassStatePolicy())
    fold_zero = template.for_fold()
    fold_one = template.for_fold()
    optimizer = Mock(zero_grad=Mock(), step=Mock())
    fold_zero.wrap_optimizer(optimizer)
    fold_one.wrap_optimizer(optimizer)

    assert type(fold_zero.policy) is not type(fold_one.policy)
    assert type(fold_zero.policy).COUNT == 1
    assert type(fold_one.policy).COUNT == 1
    assert ClassStatePolicy.COUNT == 0


def test_each_fold_gets_an_isolated_module_helper_namespace():
    namespace = {"__name__": "_fold_policy_fixture"}
    exec('''
class Helper:
    COUNT = 0

class ModulePolicy:
    def wrap_optimizer_for(self, opt, *, role):
        alias = Helper
        alias.COUNT += 1
        return opt
''', namespace)
    policy_type = namespace["ModulePolicy"]
    helper_type = namespace["Helper"]
    template = PolicyRuntime(name="module-state", policy=policy_type())
    fold_zero = template.for_fold()
    fold_one = template.for_fold()
    optimizer = Mock(zero_grad=Mock(), step=Mock())
    fold_zero.wrap_optimizer(optimizer)
    fold_one.wrap_optimizer(optimizer)

    helper_zero = fold_zero.policy.wrap_optimizer_for.__globals__["Helper"]
    helper_one = fold_one.policy.wrap_optimizer_for.__globals__["Helper"]
    assert helper_zero is not helper_one
    assert helper_zero.COUNT == 1
    assert helper_one.COUNT == 1
    assert helper_type.COUNT == 0


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
