"""Static purity validator: rejects top-level I/O, network, mutable globals (REG-03 / D-30).

Pure AST walk — never imports the module. Safe to run on untrusted code because
no user code is executed. The submit hook (Plan 01-07) runs this before the
also-static InterfaceValidator; neither validator imports candidate modules
(T-01-14).
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path

from automil.registry.errors import ValidationError

logger = logging.getLogger(__name__)

# Banned top-level call names: open(), print(), exec(), eval(), etc.
_BANNED_BUILTINS: frozenset[str] = frozenset({
    "open", "print", "input", "exec", "eval", "compile",
})

# Banned top-level module prefixes — any `<mod>.<attr>(...)` at module scope is rejected.
_BANNED_MODULES: frozenset[str] = frozenset({
    "requests", "urllib", "socket", "http", "ftplib", "smtplib",
    "subprocess",
})

# Specific os.* attributes that imply filesystem/process side effects.
_BANNED_OS_ATTRS: frozenset[str] = frozenset({
    "system", "popen", "remove", "unlink", "mkdir", "rmdir", "rename",
    "makedirs", "removedirs", "chmod", "chown",
})

# Imports execute module code.  Variant modules therefore get only the trusted
# registry API plus annotation-only standard-library helpers at module scope;
# numerical/framework imports belong inside methods where they cannot execute
# during registry scanning.
_TRUSTED_IMPORTS: frozenset[str] = frozenset({
    "__future__", "automil.registry", "typing", "collections.abc",
})


def _is_immutable_literal(node: ast.AST) -> bool:
    """Return whether an AST expression is an immutable constant.

    Allowed: Constant (str/int/float/bool/None), Tuple of immutables,
    BinOp/UnaryOp on Constants (e.g., -1, 3.14 * 2), well-known names
    (None, True, False), frozenset/tuple calls.

    Disallowed: List, Dict, Set, comprehensions, arbitrary Calls.
    """
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp):
        return _is_immutable_literal(node.operand)
    if isinstance(node, ast.Tuple):
        return all(_is_immutable_literal(elt) for elt in node.elts)
    if isinstance(node, ast.BinOp):
        return _is_immutable_literal(node.left) and _is_immutable_literal(node.right)
    if isinstance(node, ast.Name):
        # Only accept well-known immutable built-in names.
        return node.id in {"None", "True", "False"}
    if isinstance(node, ast.Call):
        func = node.func
        # Allow tuple(...) and frozenset(...) — both produce immutable results.
        if (
            isinstance(func, ast.Name)
            and func.id in {"tuple", "frozenset"}
            and not node.keywords
        ):
            return all(_is_immutable_literal(arg) for arg in node.args)
        return False
    return False


class PurityValidator:
    """Static AST walk; never imports the module under inspection.

    D-30 purity rules:
      - No top-level I/O (open, print, exec, eval, compile, input).
      - No top-level network/process calls (requests.*, urllib.*, socket.*,
        subprocess.*, http.*, os.system, os.popen).
      - No top-level filesystem mutations (Path.write_*, .read_*, .append_*).
      - No top-level mutable module-level globals (list, dict, set,
        comprehensions).
      - No `if __name__ == "__main__":` blocks — variants are libraries.
      - Top-level subscript-target assignment rejected
        (e.g., os.environ["X"] = "y").
      - Constants, class/function definitions, imports, and the docstring
        are all allowed.
    """

    def check(self, module_path: Path) -> None:
        """Validate a variant module for purity. Raises ValidationError on failure."""
        if not module_path.exists():
            raise ValidationError(
                validator_name="purity",
                path=module_path,
                reason="module not found",
                fix_suggestion="Verify the path exists and is readable.",
            )

        try:
            source = module_path.read_text()
            tree = ast.parse(source, filename=str(module_path))
        except SyntaxError as e:
            raise ValidationError(
                validator_name="purity",
                path=module_path,
                line=e.lineno,
                column=e.offset,
                reason=f"syntax error: {e.msg}",
                fix_suggestion="Fix the Python syntax error in the module.",
            ) from e

        for node in tree.body:
            self._check_top_level_node(module_path, node)

    def _check_top_level_node(self, module_path: Path, node: ast.AST) -> None:
        """Inspect one top-level statement for purity violations."""

        # --- trusted imports only (imports execute code at module load) ---
        if isinstance(node, ast.Import):
            denied = [
                alias.name
                for alias in node.names
                if alias.name not in _TRUSTED_IMPORTS
            ]
            if denied:
                self._definition_error(
                    module_path, node, f"untrusted top-level import(s) {denied}",
                )
            return
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module not in _TRUSTED_IMPORTS:
                self._definition_error(
                    module_path,
                    node,
                    f"untrusted top-level import {node.module!r}",
                )
            return
        if isinstance(node, ast.ClassDef):
            self._check_class_definition(module_path, node)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._check_function_header(module_path, node)
            return

        # --- module-level expressions ---
        if isinstance(node, ast.Expr):
            # Module docstring (str constant) is OK.
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return
            # Top-level Call: only the @register decorator is sanctioned, but
            # @register appears on ClassDef nodes (not as a bare Expr).  Any
            # bare top-level call expression is a side-effect smell.
            if isinstance(node.value, ast.Call):
                self._reject_call(module_path, node.value)
            else:
                raise ValidationError(
                    validator_name="purity",
                    path=module_path,
                    line=node.lineno,
                    column=node.col_offset,
                    reason=f"unexpected top-level expression: {ast.dump(node.value)[:60]}",
                    fix_suggestion="Move the expression into a function body or remove it.",
                )
            return

        # --- module-level assignments ---
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            if isinstance(node, ast.AugAssign):
                self._definition_error(
                    module_path, node, "module constants cannot be mutated",
                )
            # Subscript-target assignments are banned:
            # e.g., os.environ["X"] = "y"   (Assign with Subscript target)
            targets = getattr(node, "targets", None) or [getattr(node, "target", None)]
            for tgt in (t for t in targets if t is not None):
                if isinstance(tgt, ast.Subscript):
                    raise ValidationError(
                        validator_name="purity",
                        path=module_path,
                        line=node.lineno,
                        column=node.col_offset,
                        reason=(
                            "banned top-level subscript assignment "
                            "(e.g., os.environ[...] = ...) — mutable module-level state (D-30)"
                        ),
                        fix_suggestion=(
                            "Move the assignment into a function/method body."
                        ),
                    )

            value = getattr(node, "value", None)
            if value is None:
                return  # bare AnnAssign (`x: int`) is metadata, OK
            if not _is_immutable_literal(value):
                # Innermost first preserves the specific root-cause diagnostic
                # for chains such as ``open(...).read()``.
                for sub_node in reversed(list(ast.walk(value))):
                    if isinstance(sub_node, ast.Call):
                        self._reject_call(module_path, sub_node)
                self._definition_error(
                    module_path,
                    value,
                    "module attributes must be immutable literals",
                )
            return

        # --- if blocks ---
        if isinstance(node, ast.If):
            test = node.test
            # Reject `if __name__ == "__main__":` — variants are libraries, not scripts.
            is_main_block = (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == "__main__"
            )
            if is_main_block:
                raise ValidationError(
                    validator_name="purity",
                    path=module_path,
                    line=node.lineno,
                    column=node.col_offset,
                    reason=(
                        '`if __name__ == "__main__":` block detected; '
                        "variant modules are libraries, not scripts (D-30)"
                    ),
                    fix_suggestion=(
                        'Remove the `__main__` block — variants are imported, not run '
                        "directly.  If you need a runnable demo, put it in a sibling "
                        "tests/ file."
                    ),
                )
            self._definition_error(
                module_path,
                node,
                "top-level conditional blocks are not declarative",
            )

        # --- anything else (Try, With, While, For, ...) is suspect ---
        raise ValidationError(
            validator_name="purity",
            path=module_path,
            line=getattr(node, "lineno", None),
            column=getattr(node, "col_offset", None),
            reason=f"unsupported top-level construct: {type(node).__name__}",
            fix_suggestion=(
                "Move this construct into a function/method body, or restructure "
                "the module so all top-level code is class/function definitions, "
                "imports, and immutable constants."
            ),
        )

    def _definition_error(
        self, module_path: Path, node: ast.AST, reason: str,
    ) -> None:
        raise ValidationError(
            validator_name="purity",
            path=module_path,
            line=getattr(node, "lineno", None),
            column=getattr(node, "col_offset", None),
            reason=f"unsafe definition-time expression: {reason}",
            fix_suggestion=(
                "Keep decorators, bases, annotations, defaults, and class-body "
                "constants declarative; move executable work into a method body."
            ),
        )

    def _check_function_header(
        self,
        module_path: Path,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        safe_decorators = {"staticmethod", "classmethod", "property", "abstractmethod"}
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Name) or decorator.id not in safe_decorators:
                self._definition_error(
                    module_path, decorator, "function decorator is not allowlisted",
                )
        defaults = [*node.args.defaults, *node.args.kw_defaults]
        for default in defaults:
            if default is not None and not _is_immutable_literal(default):
                self._definition_error(
                    module_path, default, "function default is executable or mutable",
                )
        annotations = [
            arg.annotation
            for arg in [
                *node.args.posonlyargs, *node.args.args,
                *node.args.kwonlyargs,
            ]
            if arg.annotation is not None
        ]
        if node.args.vararg and node.args.vararg.annotation is not None:
            annotations.append(node.args.vararg.annotation)
        if node.args.kwarg and node.args.kwarg.annotation is not None:
            annotations.append(node.args.kwarg.annotation)
        if node.returns is not None:
            annotations.append(node.returns)
        for annotation in annotations:
            if any(isinstance(part, ast.Call) for part in ast.walk(annotation)):
                self._definition_error(
                    module_path, annotation, "annotation contains a call",
                )
        for statement in ast.walk(node):
            if isinstance(statement, (ast.Global, ast.Nonlocal)):
                self._definition_error(
                    module_path,
                    statement,
                    "methods cannot mutate global or enclosing-scope state",
                )

    def _check_register_decorator(
        self, module_path: Path, decorator: ast.AST,
    ) -> None:
        if not (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Name)
            and decorator.func.id == "register"
            and len(decorator.args) == 1
            and not decorator.keywords
        ):
            self._definition_error(
                module_path, decorator, "class decorator must be register(VariantSpec(...))",
            )
        spec = decorator.args[0]
        if not (
            isinstance(spec, ast.Call)
            and isinstance(spec.func, ast.Name)
            and spec.func.id == "VariantSpec"
            and not spec.args
            and all(keyword.arg is not None for keyword in spec.keywords)
            and all(_is_immutable_literal(keyword.value) for keyword in spec.keywords)
        ):
            self._definition_error(
                module_path, spec, "VariantSpec fields must be immutable literals",
            )

    def _check_class_definition(self, module_path: Path, node: ast.ClassDef) -> None:
        register_decorators = [
            decorator for decorator in node.decorator_list
            if isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Name)
            and decorator.func.id == "register"
        ]
        if register_decorators and len(node.decorator_list) != 1:
            self._definition_error(
                module_path, node, "variant class must have exactly one register decorator",
            )
        if register_decorators:
            self._check_register_decorator(module_path, register_decorators[0])
        elif node.decorator_list:
            self._definition_error(
                module_path, node.decorator_list[0],
                "helper class decorators are not allowed",
            )
        if node.keywords:
            self._definition_error(
                module_path, node.keywords[0], "class keywords/metaclasses are not allowed",
            )
        for base in node.bases:
            if not isinstance(base, (ast.Name, ast.Attribute)):
                self._definition_error(
                    module_path, base, "class base must be a dotted name",
                )
        for statement in node.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._check_function_header(module_path, statement)
                continue
            if isinstance(statement, ast.Pass):
                continue
            if (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                continue
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                value = getattr(statement, "value", None)
                if value is None:
                    continue
                if not _is_immutable_literal(value):
                    self._definition_error(
                        module_path,
                        value,
                        "class attributes must be immutable literals",
                    )
                continue
            self._definition_error(
                module_path, statement,
                f"unsupported class-body construct {type(statement).__name__}",
            )

    def _reject_call(self, module_path: Path, call: ast.Call) -> None:
        """Raise ValidationError for banned call patterns."""
        func = call.func

        # Bare-name calls: open(), print(), exec(), eval(), compile(), input()
        if isinstance(func, ast.Name) and func.id in _BANNED_BUILTINS:
            raise ValidationError(
                validator_name="purity",
                path=module_path,
                line=call.lineno,
                column=call.col_offset,
                reason=f"banned top-level builtin call: {func.id}()",
                fix_suggestion=(
                    f"Move {func.id}() into a method body, or remove it. "
                    "Top-level I/O on import time leaks side effects to every "
                    "consumer that imports the module."
                ),
            )

        # Attribute calls: requests.get(), urllib.request.urlopen(),
        # socket.socket(), os.system(), subprocess.run(), etc.
        if isinstance(func, ast.Attribute):
            # Walk to the leftmost Name to find the root module.
            attr_chain: list[str] = []
            cur: ast.AST = func
            while isinstance(cur, ast.Attribute):
                attr_chain.append(cur.attr)
                cur = cur.value

            if isinstance(cur, ast.Name):
                root = cur.id
                full = ".".join(reversed([root] + attr_chain))

                if root in _BANNED_MODULES:
                    raise ValidationError(
                        validator_name="purity",
                        path=module_path,
                        line=call.lineno,
                        column=call.col_offset,
                        reason=f"banned top-level network/process call: {full}(...)",
                        fix_suggestion=(
                            f"Move {full}(...) into a method body. "
                            "Top-level network or process side effects on import "
                            "violate D-30 module-level purity."
                        ),
                    )

                if root == "os" and attr_chain and attr_chain[-1] in _BANNED_OS_ATTRS:
                    raise ValidationError(
                        validator_name="purity",
                        path=module_path,
                        line=call.lineno,
                        column=call.col_offset,
                        reason=f"banned top-level os.* call: {full}(...)",
                        fix_suggestion=f"Move {full}(...) into a method body.",
                    )

            # Path("/x").write_text("y") — the attr is write_text/read_text etc.
            # Detect: any top-level call whose method name starts with
            # "write", "read", or "append" is a filesystem I/O smell.
            if func.attr.startswith(("write_", "read_", "append_")):
                raise ValidationError(
                    validator_name="purity",
                    path=module_path,
                    line=call.lineno,
                    column=call.col_offset,
                    reason=f"banned top-level filesystem/I/O call: .{func.attr}(...)",
                    fix_suggestion="Move the I/O into a method body.",
                )

        raise ValidationError(
            validator_name="purity",
            path=module_path,
            line=call.lineno,
            column=call.col_offset,
            reason="top-level calls are not declarative and execute during import",
            fix_suggestion="Move the call into a function or method body.",
        )
