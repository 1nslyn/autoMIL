"""Every runner->trainer call site must bind against its callee's signature.

The DTFD survival path shipped broken: ``runner.py`` passed ``policy_runtime=``
to ``train_dtfd_survival_fold``, which never declared that parameter, so the
call raised ``TypeError`` before a single epoch ran. Nothing caught it —
``py_compile`` is happy with a bad keyword, the classification sibling *does*
accept it, and no test exercised survival.

These tests bind each fold-trainer call site found in the runners against the
real callee signature, so a keyword that no longer exists fails at collection
time rather than an hour into a cluster job.
"""

from __future__ import annotations

import ast
import inspect
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from autobench import BENCHMARKS_ROOT  # noqa: E402

#: ``runner module path`` -> ``{called name: dotted import of the callee}``.
#: Only fold trainers are listed: they are the boundary where a runner hands a
#: whole fold to an arm, and the boundary that broke.
_CALL_SITES = {
    "autobench/pipeline/dtfd/runner.py": {
        "train_dtfd_fold": "autobench.pipeline.dtfd.train:train_dtfd_fold",
        "train_dtfd_survival_fold": (
            "autobench.pipeline.dtfd.survival_train:train_dtfd_survival_fold"
        ),
    },
}


def _import_callee(dotted: str):
    module_path, _, attr = dotted.partition(":")
    module = __import__(module_path, fromlist=[attr])
    return getattr(module, attr)


def _call_sites(source_rel: str, names: set[str]):
    """Yield ``(name, lineno, n_positional, keyword_names)`` per matching call."""
    source = (BENCHMARKS_ROOT / "src" / source_rel).read_text()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id not in names:
            continue
        if any(k.arg is None for k in node.keywords):
            pytest.fail(f"{source_rel}:{node.lineno} uses **kwargs; cannot bind")
        yield (
            node.func.id,
            node.lineno,
            len(node.args),
            [k.arg for k in node.keywords],
        )


@pytest.mark.parametrize("source_rel", sorted(_CALL_SITES))
def test_fold_trainer_call_sites_bind(source_rel: str) -> None:
    """Each call's arguments bind against the function it actually resolves to."""
    targets = _CALL_SITES[source_rel]
    seen: set[str] = set()

    for name, lineno, n_positional, keywords in _call_sites(source_rel, set(targets)):
        seen.add(name)
        signature = inspect.signature(_import_callee(targets[name]))
        # Placeholder values: binding validates arity and names, never types.
        try:
            signature.bind(
                *[object()] * n_positional,
                **{keyword: object() for keyword in keywords},
            )
        except TypeError as exc:
            pytest.fail(
                f"{source_rel}:{lineno} calls {name}() with arguments it does "
                f"not accept: {exc}\n"
                f"  positional={n_positional} keywords={keywords}\n"
                f"  signature={signature}"
            )

    missing = set(targets) - seen
    assert not missing, (
        f"{source_rel} no longer calls {sorted(missing)}; update _CALL_SITES so "
        "this guard keeps covering the fold-trainer boundary."
    )


def test_survival_trainer_accepts_policy_runtime() -> None:
    """The survival trainer's body reads ``policy_runtime``, so it must take it.

    Dropping the keyword at the call site would silence the ``TypeError`` and
    trade it for an ``UnboundLocalError`` from the body's
    ``policy_runtime = policy_runtime or PolicyRuntime()``. Pin the parameter
    itself so neither failure can come back.
    """
    from autobench.pipeline.dtfd.survival_train import train_dtfd_survival_fold

    parameters = inspect.signature(train_dtfd_survival_fold).parameters
    assert "policy_runtime" in parameters, (
        "train_dtfd_survival_fold reads policy_runtime in its body but does "
        "not declare it as a parameter"
    )
    assert parameters["policy_runtime"].default is None


def _walk_own_scope(function: ast.AST):
    """Walk ``function``'s body without descending into nested function scopes.

    A nested ``def``/``lambda`` has its own parameters, so ``metric =
    metric.get("mean")`` inside one is safe even when the enclosing function
    never heard of ``metric``. The caller visits every function separately, so
    stopping at the boundary loses no coverage.
    """
    stack = list(ast.iter_child_nodes(function))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(node))


def _self_referential_first_assignments(function: ast.AST) -> list[tuple[str, int]]:
    """``name = <expr reading name>`` where that assignment is the first bind.

    The right-hand side evaluates before the target binds, so reading a name
    there is an ``UnboundLocalError`` unless the name arrived as a parameter
    (or was bound earlier in the body). This is the exact shape of the DTFD
    defect: ``policy_runtime = policy_runtime or PolicyRuntime()`` inside a
    function that never declared ``policy_runtime``.
    """
    args = getattr(function, "args", None)
    declared = set()
    if args is not None:
        declared = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
        if args.vararg:
            declared.add(args.vararg.arg)
        if args.kwarg:
            declared.add(args.kwarg.arg)
    for statement in _walk_own_scope(function):
        if isinstance(statement, (ast.Global, ast.Nonlocal)):
            declared.update(statement.names)

    # Every *other* way a local name can come into existence. A self-referential
    # assignment is only a defect when it is the name's sole binding site — a
    # ``for embeddings, ... in loader:`` loop makes the ``embeddings =
    # embeddings.to(device)`` inside it perfectly safe. Checking "is there any
    # other binder" needs no execution-order model, which ``ast.walk`` (breadth
    # first) could not give us anyway.
    self_assign_targets: dict[str, int] = {}
    other_binders: set[str] = set(declared)

    def _names(target: ast.AST) -> set[str]:
        return {
            n.id for n in ast.walk(target)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }

    for node in _walk_own_scope(function):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            other_binders |= _names(node.target)
        elif isinstance(node, ast.comprehension):
            other_binders |= _names(node.target)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            other_binders |= _names(node.optional_vars)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            other_binders.add(node.name)
        elif isinstance(node, ast.NamedExpr):
            other_binders |= _names(node.target)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            other_binders |= _names(node.target)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            other_binders |= {a.asname or a.name.split(".")[0] for a in node.names}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node is not function:
                other_binders.add(node.name)
        elif isinstance(node, ast.Assign):
            read = {
                n.id for n in ast.walk(node.value)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            }
            for target in node.targets:
                for name in _names(target):
                    if name in read:
                        self_assign_targets.setdefault(name, node.lineno)
                    else:
                        other_binders.add(name)

    return sorted(
        (name, lineno)
        for name, lineno in self_assign_targets.items()
        if name not in other_binders
    )


def test_no_function_reads_a_name_it_never_bound() -> None:
    """Catches the second half of the DTFD defect, generalized.

    Had the call site simply dropped the bad keyword, the ``TypeError`` would
    have become an ``UnboundLocalError`` from the trainer's own
    ``policy_runtime = policy_runtime or PolicyRuntime()``. Neither failure can
    return while this holds.
    """
    offenders: list[str] = []
    pipeline = BENCHMARKS_ROOT / "src" / "autobench" / "pipeline"
    for path in sorted(pipeline.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for name, lineno in _self_referential_first_assignments(node):
                rel = path.relative_to(BENCHMARKS_ROOT / "src")
                offenders.append(f"{rel}:{lineno} {node.name}() reads `{name}`")

    assert not offenders, (
        "these assignments read a name on the right-hand side before anything "
        f"binds it (UnboundLocalError at runtime): {offenders}"
    )


def test_no_dead_policy_runtime_parameters() -> None:
    """A declared-but-unused ``policy_runtime`` is a policy that silently no-ops.

    ``_risk_records`` accepted one and ignored it, which reads as "policy is
    honoured here" at every call site. Keep the parameter list honest.
    """
    offenders: list[str] = []
    for path in sorted((BENCHMARKS_ROOT / "src" / "autobench" / "pipeline").rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = node.args
            declared = {
                a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)
            }
            if "policy_runtime" not in declared:
                continue
            uses = sum(
                1
                for inner in ast.walk(node)
                if isinstance(inner, ast.Name) and inner.id == "policy_runtime"
            )
            if uses == 0:
                rel = path.relative_to(BENCHMARKS_ROOT / "src")
                offenders.append(f"{rel}:{node.lineno} {node.name}()")

    assert not offenders, (
        f"these functions take `policy_runtime` and never use it: {offenders}"
    )
