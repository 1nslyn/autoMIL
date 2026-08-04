"""Static interface validator: ABC base + required-method signature check.

Run order in the submit hook (Plan 01-07):
  1. PurityValidator (pure AST, no import — fast, safe).
  2. InterfaceValidator (AST-only; never imports agent-authored code).

Both validators are AST-only. This is the T-01-14 mitigation: validation never
executes decorators, defaults, annotations, class bodies, or imported modules.
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Optional

from automil.registry.errors import ValidationError

logger = logging.getLogger(__name__)


_ABC_NAMES = {
    "model": "ModelVariant",
    "loss": "LossVariant",
    "policy": "PolicyVariant",
}

_ABC_POSITIONAL = {
    ("model", "forward"): ("features", "coords"),
    ("loss", "__call__"): ("logits", "targets"),
    ("policy", "wrap_optimizer"): ("opt",),
}


def _required_methods(kind: str) -> list[str]:
    """The methods the ABC marks @abstractmethod for each kind."""
    return {
        "model": ["forward"],
        "loss": ["__call__"],
        "policy": ["wrap_optimizer"],
    }[kind]


def _find_register_calls(tree: ast.Module) -> list[tuple[ast.ClassDef, ast.Call]]:
    """Return [(class_def_ast, register_call_ast), ...] for every @register-decorated class.

    D-26: each variant module should contain exactly one.
    """
    out: list[tuple[ast.ClassDef, ast.Call]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call):
                func = dec.func
                if isinstance(func, ast.Name) and func.id == "register":
                    out.append((node, dec))
                elif isinstance(func, ast.Attribute) and func.attr == "register":
                    out.append((node, dec))
    return out


def _extract_kind_from_register_call(call: ast.Call) -> Optional[str]:
    """Extract the kind='...' literal from @register(VariantSpec(kind=...)).

    Returns None if the kind cannot be statically determined. The validator is
    fail-closed and never falls back to importing the candidate module.
    """
    if not call.args:
        return None
    spec_call = call.args[0]
    if not isinstance(spec_call, ast.Call):
        return None
    for kw in spec_call.keywords:
        if kw.arg == "kind" and isinstance(kw.value, ast.Constant):
            v = kw.value.value
            if isinstance(v, str):
                return v
    return None


def _trusted_imports(tree: ast.Module) -> dict[str, str]:
    """Return local names imported directly from the trusted registry API."""
    trusted: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module != "automil.registry":
            continue
        for name in node.names:
            trusted[name.asname or name.name] = name.name
    return trusted


def _ast_signature_compatible(
    kind: str, method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[bool, str]:
    """Apply the existing required-positional rule without importing code."""
    positional = [*method.args.posonlyargs, *method.args.args]
    if positional and positional[0].arg in {"self", "cls"}:
        positional = positional[1:]
    required_count = max(0, len(positional) - len(method.args.defaults))
    required = {arg.arg for arg in positional[:required_count]}
    allowed = set(_ABC_POSITIONAL[(kind, method.name)])
    unknown = sorted(required - allowed)
    if unknown:
        return False, (
            f"variant introduces new required positional parameter(s) {unknown} "
            "not present in the ABC"
        )
    return True, ""


class InterfaceValidator:
    """ABC subclass + required-method signature check.

    Run order within check():
      1. AST scan (cheap):  parse → find @register classes → count them.
      2. Static trusted-base + method signature checks. No candidate import.
    """

    def check(self, module_path: Path) -> None:
        """Validate a variant module. Raises ValidationError on first failure."""
        if not module_path.exists():
            raise ValidationError(
                validator_name="interface",
                path=module_path,
                reason="module not found",
                fix_suggestion="Verify the path exists and is readable.",
            )

        # --- Phase 1: AST scan (cheap, no import) ---
        try:
            source = module_path.read_text()
            tree = ast.parse(source, filename=str(module_path))
        except SyntaxError as e:
            raise ValidationError(
                validator_name="interface",
                path=module_path,
                line=e.lineno,
                column=e.offset,
                reason=f"syntax error: {e.msg}",
                fix_suggestion="Fix the Python syntax error.",
            ) from e

        register_classes = _find_register_calls(tree)

        if not register_classes:
            raise ValidationError(
                validator_name="interface",
                path=module_path,
                reason="no @register decorator found; not a variant module",
                fix_suggestion=(
                    "Decorate the variant class with @register(VariantSpec(...)). "
                    "See src/automil/registry/registrar.py for the API."
                ),
            )

        if len(register_classes) > 1:
            first_class, _ = register_classes[0]
            second_class, _ = register_classes[1]
            raise ValidationError(
                validator_name="interface",
                path=module_path,
                line=second_class.lineno,
                column=second_class.col_offset,
                reason=(
                    f"multiple @register-decorated classes detected: "
                    f"{first_class.name!r} and {second_class.name!r}. "
                    f"D-26 requires a single .py file per variant "
                    f"('one variant, one file')."
                ),
                fix_suggestion=(
                    f"Split into two files: one for {first_class.name} and one "
                    f"for {second_class.name}."
                ),
            )

        class_def, register_call = register_classes[0]
        kind_hint = _extract_kind_from_register_call(register_call)

        kind = kind_hint
        if kind not in _ABC_NAMES:
            raise ValidationError(
                validator_name="interface",
                path=module_path,
                line=class_def.lineno,
                reason=(
                    f"unknown or undetermined kind {kind!r} for class "
                    f"{class_def.name!r}"
                ),
                fix_suggestion=(
                    "Set kind='model' | 'loss' | 'policy' in the VariantSpec "
                    "decorator argument."
                ),
            )

        trusted = _trusted_imports(tree)
        expected_base = _ABC_NAMES[kind]
        actual_bases = [
            base.id for base in class_def.bases if isinstance(base, ast.Name)
        ]
        trusted_bases = [trusted.get(name) for name in actual_bases]
        if expected_base not in trusted_bases:
            raise ValidationError(
                validator_name="interface",
                path=module_path,
                line=class_def.lineno,
                reason=(
                    f"class {class_def.name!r} declared kind={kind!r} but is not "
                    f"a direct subclass of trusted {expected_base} "
                    f"(actual bases: {actual_bases[:3]})"
                ),
                fix_suggestion=(
                    f"Import {expected_base} from automil.registry and use "
                    f"`class {class_def.name}({expected_base}):` "
                    f"or update kind= in the VariantSpec."
                ),
            )

        # Required-method existence + signature compatibility.
        required = _required_methods(kind)
        for method_name in required:
            variant_method = next((
                node for node in class_def.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == method_name
            ), None)
            if variant_method is None:
                raise ValidationError(
                    validator_name="interface",
                    path=module_path,
                    line=class_def.lineno,
                    reason=(
                        f"missing required method {method_name!r} on "
                        f"{class_def.name!r} (kind={kind!r} requires "
                        f"{required})"
                    ),
                    fix_suggestion=(
                        f"Add `def {method_name}(self, ...)` to the class. "
                        f"See {expected_base}.{method_name} for the "
                        f"expected signature. ABC: {expected_base}"
                    ),
                )

            ok, reason = _ast_signature_compatible(kind, variant_method)
            if not ok:
                raise ValidationError(
                    validator_name="interface",
                    path=module_path,
                    line=class_def.lineno,
                    reason=(
                        f"method {method_name!r} signature incompatible with "
                        f"{expected_base}.{method_name}: {reason}"
                    ),
                    fix_suggestion=(
                        f"Match the ABC's signature: see "
                        f"{expected_base}.{method_name} for the expected "
                        f"parameter list."
                    ),
                )

        logger.debug(
            "InterfaceValidator: %s passed (kind=%r, class=%r)",
            module_path.name, kind, class_def.name,
        )
