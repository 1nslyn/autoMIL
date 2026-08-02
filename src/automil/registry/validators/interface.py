"""Static interface validator: ABC subclass + required-method signature check (REG-03 / D-30).

Run order in the submit hook (Plan 01-07):
  1. PurityValidator (pure AST, no import — fast, safe).
  2. InterfaceValidator (imports the module for reflection — only after purity passes).

This ordering is the mitigation for T-01-14 (privilege elevation via malicious
module import).  Plan 01-07 MUST enforce purity-before-interface.
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from automil.registry.errors import ValidationError

logger = logging.getLogger(__name__)


@contextmanager
def _isolated_registry_state() -> Iterator[None]:
    """Run reflection imports without reading or mutating live registrations.

    ``@register`` necessarily writes to process-global dictionaries at import
    time.  Interface validation is a property of one module, however, and may
    run repeatedly inside a long-lived CLI/test process.  Snapshotting, clearing,
    and restoring all registry stores makes validation idempotent while keeping
    the decorator's kind/spec checks active for the module under inspection.
    Cross-module name collisions remain the scanner/runtime registry's job.
    """
    from automil.registry._state import (
        LOSS_VARIANTS,
        MODEL_VARIANTS,
        POLICY_VARIANTS,
        SPEC_STORE,
    )

    stores = (MODEL_VARIANTS, LOSS_VARIANTS, POLICY_VARIANTS, SPEC_STORE)
    snapshots = tuple(dict(store) for store in stores)
    for store in stores:
        store.clear()
    try:
        yield
    finally:
        for store, snapshot in zip(stores, snapshots):
            store.clear()
            store.update(snapshot)


def _abcs() -> dict[str, type]:
    """Lazy import to avoid circular at module import time."""
    from automil.registry.variants import LossVariant, ModelVariant, PolicyVariant
    return {"model": ModelVariant, "loss": LossVariant, "policy": PolicyVariant}


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

    Returns None if the kind cannot be statically determined (e.g., stored in
    a variable), in which case the validator falls back to runtime introspection.
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


def _import_module_from_path(module_path: Path) -> Any:
    """Import a Python file as a module using a unique name to avoid sys.modules pollution."""
    spec = importlib.util.spec_from_file_location(
        f"_automil_validator_{module_path.stem}_{id(module_path)}", module_path
    )
    if spec is None or spec.loader is None:
        raise ValidationError(
            validator_name="interface",
            path=module_path,
            reason="cannot create module spec — file may not be a valid Python source",
            fix_suggestion="Verify the file is a valid Python module with a .py extension.",
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _signature_compatible(abc_method: Any, variant_method: Any) -> tuple[bool, str]:
    """Check whether variant_method's signature is compatible with abc_method's.

    Compatibility rules:
      - Variant must not add NEW positional parameters not present in the ABC
        (callers that follow the ABC's interface have no way to supply them).
      - Variant MAY tighten a default (e.g., ABC has `coords=None`, variant
        has `coords` with no default) — this is valid narrowing.
      - Variant MAY have fewer parameters (dropping optional ABC params).
      - Variant MAY add keyword-only (**kwargs) parameters.

    Returns (ok, reason_if_not_ok).
    """
    try:
        abc_sig = inspect.signature(abc_method)
        var_sig = inspect.signature(variant_method)
    except (ValueError, TypeError) as e:
        return False, f"could not inspect signature: {e}"

    abc_params = list(abc_sig.parameters.values())
    var_params = list(var_sig.parameters.values())

    # Skip `self` for both.
    if abc_params and abc_params[0].name == "self":
        abc_params = abc_params[1:]
    if var_params and var_params[0].name == "self":
        var_params = var_params[1:]

    # Build name-sets: positional-or-keyword params only (not *args/**kwargs).
    _positional_kinds = {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }
    abc_positional_names = {
        p.name for p in abc_params if p.kind in _positional_kinds
    }
    var_positional = [p for p in var_params if p.kind in _positional_kinds]
    var_positional_names = {p.name for p in var_positional}

    # Variant must NOT introduce positional params unknown to the ABC.
    new_params = var_positional_names - abc_positional_names
    # Filter: only truly NEW required ones are a problem (no default, and
    # callers won't know to pass them).
    truly_new_required = [
        p for p in var_positional
        if p.name in new_params and p.default is inspect.Parameter.empty
    ]
    if truly_new_required:
        names = [p.name for p in truly_new_required]
        return False, (
            f"variant introduces new required positional parameter(s) "
            f"{names} not present in the ABC; callers following the ABC "
            f"interface cannot provide them"
        )

    return True, ""


class InterfaceValidator:
    """ABC subclass + required-method signature check.

    Run order within check():
      1. AST scan (cheap):  parse → find @register classes → count them.
      2. Dynamic import + reflection (heavier): ABC subclass check + method
         existence + signature compatibility.

    This is NOT a fully static validator (import is needed for reflection), but
    it is safe because Plan 01-07 runs PurityValidator first.
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

        # --- Phase 2: Dynamic import + reflection ---
        try:
            with _isolated_registry_state():
                module = _import_module_from_path(module_path)
        except SyntaxError as e:
            raise ValidationError(
                validator_name="interface",
                path=module_path,
                line=e.lineno,
                reason=f"syntax error during import: {e.msg}",
                fix_suggestion="Fix the Python syntax error.",
            ) from e
        except ValidationError:
            raise
        except Exception as e:
            # RegistrationError from @register means the class failed a runtime
            # check (ABC mismatch, duplicate, bad kind).  Translate to a
            # ValidationError that names kind + the offending class + LossVariant/etc.
            from automil.registry.registrar import RegistrationError
            if isinstance(e, RegistrationError):
                # Best-effort: include the class name + kind + base classes from AST.
                raise ValidationError(
                    validator_name="interface",
                    path=module_path,
                    line=class_def.lineno,
                    reason=(
                        f"@register raised RegistrationError for class "
                        f"{class_def.name!r} (kind={kind_hint!r}): {e}. "
                        f"Check that the class subclasses the right ABC for "
                        f"kind={kind_hint!r} and that the base class "
                        f"({', '.join(b.id for b in class_def.bases if isinstance(b, ast.Name))}) "
                        f"matches."
                    ),
                    fix_suggestion=(
                        f"Set kind= to match the actual base class in the VariantSpec, "
                        f"or change the class to subclass the correct ABC for "
                        f"kind={kind_hint!r}."
                    ),
                ) from e
            raise ValidationError(
                validator_name="interface",
                path=module_path,
                reason=f"import failed: {type(e).__name__}: {e}",
                fix_suggestion="Inspect the import-time error in the module.",
            ) from e

        cls = getattr(module, class_def.name, None)
        if cls is None:
            raise ValidationError(
                validator_name="interface",
                path=module_path,
                line=class_def.lineno,
                reason=f"class {class_def.name!r} not found in module after import",
                fix_suggestion="Confirm the class is defined at module top level.",
            )

        abcs = _abcs()

        # Resolve kind: prefer AST hint, fall back to runtime introspection.
        kind = kind_hint
        if kind is None:
            for k, abc in abcs.items():
                if isinstance(cls, type) and issubclass(cls, abc):
                    kind = k
                    break

        if kind not in abcs:
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

        abc_class = abcs[kind]

        # ABC subclass check.
        if not (isinstance(cls, type) and issubclass(cls, abc_class)):
            actual_bases = [b.__name__ for b in getattr(cls, "__mro__", [cls])[1:] if b is not object]
            raise ValidationError(
                validator_name="interface",
                path=module_path,
                line=class_def.lineno,
                reason=(
                    f"class {class_def.name!r} declared kind={kind!r} but is not a "
                    f"subclass of {abc_class.__name__} (actual bases: "
                    f"{actual_bases[:3]})"
                ),
                fix_suggestion=(
                    f"Change `class {class_def.name}({abc_class.__name__}):` "
                    f"or update kind= in the VariantSpec."
                ),
            )

        # Required-method existence + signature compatibility.
        required = _required_methods(kind)
        for method_name in required:
            variant_method = getattr(cls, method_name, None)
            # Check if method is still abstract (not overridden).
            if (
                variant_method is None
                or getattr(variant_method, "__isabstractmethod__", False)
            ):
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
                        f"See {abc_class.__name__}.{method_name} for the "
                        f"expected signature. ABC: {abc_class.__name__}"
                    ),
                )

            abc_method = getattr(abc_class, method_name)
            ok, reason = _signature_compatible(abc_method, variant_method)
            if not ok:
                raise ValidationError(
                    validator_name="interface",
                    path=module_path,
                    line=class_def.lineno,
                    reason=(
                        f"method {method_name!r} signature incompatible with "
                        f"{abc_class.__name__}.{method_name}: {reason}"
                    ),
                    fix_suggestion=(
                        f"Match the ABC's signature: see "
                        f"{abc_class.__name__}.{method_name} for the expected "
                        f"parameter list."
                    ),
                )

        logger.debug(
            "InterfaceValidator: %s passed (kind=%r, class=%r)",
            module_path.name, kind, class_def.name,
        )
