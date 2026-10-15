#!/usr/bin/env python3
"""Instantiate the composite recipe template into one policy file with its constants baked in.

``benchmarks/experiments/dev/_recipes/composite.py`` is a single ``Recipe`` policy whose
switches are class constants (None = off). One dev-cell attempt is that template with its
four provenance fields filled in and the attempt's constants written over the template's
defaults, so every attempt's policy file is complete sealed-seam code an agent could have
written. The result is checked the way ``automil submit`` checks a policy: strict purity
under the cell's method-import seal, then the interface validator.

    python benchmarks/scripts/dev_recipe.py instantiate --name ema_cos --node-id node_0002 \\
        --base-commit abc1234 --set EMA_DECAY=0.998 --set ARM=abmil --out ema_cos.py

``--set NAME=VALUE`` parses VALUE as a Python literal (a bare word such as ``abmil`` is the
string); a list becomes a tuple, because policy class constants must be immutable. What a
constant's value may be is the template's own check (``Recipe._validate``), which the
policy smoke runs at submit.
"""
from __future__ import annotations

import argparse
import ast
import math
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from automil.registry import InterfaceValidator, PurityValidator, ValidationError
from automil.registry.config import load_registry_config

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = REPO_ROOT / "benchmarks" / "experiments" / "dev" / "_recipes" / "composite.py"
#: The cohort whose ``registry`` block every dev cell copies: same seal, same locks.
SEAL_CONFIG_DIR = REPO_ROOT / "benchmarks" / "experiments" / "tcga_luad" / "automil"
RECIPE_CLASS = "Recipe"
PLACEHOLDERS = {
    "name": "__NAME__",
    "base_commit": "__BASE_COMMIT__",
    "node_id": "__NODE_ID__",
    "created_at": "__CREATED_AT__",
}
#: The registry scanner skips files starting with ``_``, so a policy name never does.
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
#: Provenance values land inside string literals of the template: nothing that could close one.
_TOKEN = re.compile(r"[A-Za-z0-9_.:+-]+")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _check_fields(fields: Mapping[str, str]) -> None:
    if not _NAME.fullmatch(fields["name"]):
        raise ValueError(
            f"name {fields['name']!r} must be letters, digits, '_' or '-' and must not start "
            "with '_' (the registry scanner skips such files)"
        )
    for field in ("node_id", "base_commit", "created_at"):
        if not _TOKEN.fullmatch(fields[field]):
            raise ValueError(
                f"{field} {fields[field]!r} must be letters, digits or one of . : + _ - only"
            )


def _fill_placeholders(text: str, fields: Mapping[str, str]) -> str:
    """Write the provenance fields over the template's placeholders, each of which appears once."""
    for field, placeholder in PLACEHOLDERS.items():
        found = text.count(placeholder)
        if found != 1:
            raise ValueError(
                f"the template must carry {placeholder} (for {field}) exactly once; found {found}"
            )
    pattern = re.compile("|".join(re.escape(p) for p in PLACEHOLDERS.values()))
    values = {placeholder: fields[field] for field, placeholder in PLACEHOLDERS.items()}
    return pattern.sub(lambda match: values[match.group(0)], text)


def _recipe_class(tree: ast.Module) -> ast.ClassDef:
    found = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == RECIPE_CLASS]
    if len(found) != 1:
        raise ValueError(f"the template must define exactly one class {RECIPE_CLASS}; found {len(found)}")
    return found[0]


def _constant_nodes(recipe: ast.ClassDef) -> dict[str, ast.Assign]:
    """The ``NAME = value`` class constants of the recipe, by name."""
    return {
        stmt.targets[0].id: stmt
        for stmt in recipe.body
        if isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id.isupper()
    }


def _bake(name: str, value: object) -> object:
    """``value`` as an immutable literal the template can hold: lists become tuples."""
    if isinstance(value, (list, tuple)):
        return tuple(_bake(name, item) for item in value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(
        f"constant {name}: {value!r} is not a literal the template can hold "
        "(None, a bool, an int, a finite float, a str, or a list or tuple of them)"
    )


def _bake_constants(text: str, constants: Mapping[str, object]) -> str:
    """Replace the value of each named class constant, leaving its line's comments alone."""
    nodes = _constant_nodes(_recipe_class(ast.parse(text)))
    unknown = sorted(set(constants) - set(nodes))
    if unknown:
        raise ValueError(f"unknown constant(s) {unknown}; the template's constants are {sorted(nodes)}")
    edits = []
    for name, value in constants.items():
        node = nodes[name].value
        if node.lineno != node.end_lineno:
            raise ValueError(f"constant {name} spans several lines in the template")
        edits.append((node.lineno - 1, node.col_offset, node.end_col_offset, repr(_bake(name, value))))
    lines = text.split("\n")
    for row, start, end, source in sorted(edits, reverse=True):
        lines[row] = lines[row][:start] + source + lines[row][end:]
    return "\n".join(lines)


def _spec_fields(recipe: ast.ClassDef) -> dict[str, object]:
    """The literal keywords of the class's ``@register(VariantSpec(...))`` decorator."""
    decorator = recipe.decorator_list[0] if len(recipe.decorator_list) == 1 else None
    spec = decorator.args[0] if isinstance(decorator, ast.Call) and decorator.args else None
    if not isinstance(spec, ast.Call):
        raise ValueError(f"{RECIPE_CLASS} must carry exactly one @register(VariantSpec(...)) decorator")
    return {keyword.arg: ast.literal_eval(keyword.value) for keyword in spec.keywords}


def _verify(text: str, constants: Mapping[str, object], fields: Mapping[str, str]) -> None:
    """Re-parse the result: every baked value and provenance field reads back as given."""
    recipe = _recipe_class(ast.parse(text))
    nodes = _constant_nodes(recipe)
    for name, value in constants.items():
        got, expected = ast.literal_eval(nodes[name].value), _bake(name, value)
        if repr(got) != repr(expected):
            raise ValueError(f"constant {name} reads back as {got!r}, not {expected!r}")
    spec = _spec_fields(recipe)
    for field in fields:
        if spec.get(field) != fields[field]:
            raise ValueError(f"{field} reads back as {spec.get(field)!r}, not {fields[field]!r}")


def _validate_policy(text: str, name: str) -> None:
    """The submit-time validators: strict purity under the cell's method-import seal, then interface."""
    seal = load_registry_config(SEAL_CONFIG_DIR).policy_method_imports
    if seal is None:
        raise ValueError(
            f"{SEAL_CONFIG_DIR / 'config.yaml'} declares no registry.policy_method_imports: "
            "refusing to instantiate a policy without the seal"
        )
    with tempfile.TemporaryDirectory(prefix="dev_recipe_") as directory:
        path = Path(directory) / f"{name}.py"
        path.write_text(text)
        try:
            PurityValidator(strict_policy=True, method_imports=seal).check(path)
            InterfaceValidator().check(path)
        except ValidationError as error:
            raise ValueError(f"recipe {name!r} fails the sealed-policy validators: {error}") from error


def instantiate_recipe(
    template_path: Path,
    *,
    name: str,
    constants: Mapping[str, object],
    node_id: str,
    base_commit: str,
    created_at: str,
) -> str:
    """The template's text with its provenance filled in and ``constants`` baked in.

    Raises ValueError for an unusable name or provenance field, a constant the template
    does not have, a value it cannot hold as an immutable literal, or a result the
    sealed-policy validators refuse.
    """
    fields = {"name": name, "node_id": node_id, "base_commit": base_commit, "created_at": created_at}
    _check_fields(fields)
    text = _fill_placeholders(Path(template_path).read_text(), fields)
    text = _bake_constants(text, constants)
    _verify(text, constants, fields)
    _validate_policy(text, name)
    return text


def _parse_assignment(item: str) -> tuple[str, object]:
    name, equals, raw = item.partition("=")
    if not equals or not name:
        raise ValueError(f"--set expects NAME=VALUE; got {item!r}")
    try:
        return name, ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        if _WORD.fullmatch(raw):
            return name, raw  # a bare word such as abmil is a string
        raise ValueError(f"--set {name}: {raw!r} is not a Python literal") from None


def _collect_constants(items: Sequence[str]) -> dict[str, object]:
    constants: dict[str, object] = {}
    for item in items:
        name, value = _parse_assignment(item)
        if name in constants:
            raise ValueError(f"--set {name} given twice")
        constants[name] = value
    return constants


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dev_recipe.py", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    instantiate = commands.add_parser(
        "instantiate", help="write a policy file: the template with its constants baked in",
    )
    instantiate.add_argument("--name", required=True, help="the policy variant name (and file stem)")
    instantiate.add_argument("--node-id", required=True)
    instantiate.add_argument("--base-commit", required=True)
    instantiate.add_argument("--created-at", help="ISO timestamp (default: now, UTC)")
    instantiate.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    instantiate.add_argument(
        "--set", dest="assignments", action="append", default=[], metavar="NAME=VALUE",
        help="a constant to bake in (repeatable); VALUE is a Python literal",
    )
    instantiate.add_argument("--out", type=Path, help="where to write the policy (default: stdout)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        text = instantiate_recipe(
            args.template,
            name=args.name,
            constants=_collect_constants(args.assignments),
            node_id=args.node_id,
            base_commit=args.base_commit,
            created_at=args.created_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
    except (ValueError, OSError, SyntaxError) as error:
        print(f"dev_recipe: {error}", file=sys.stderr)
        return 1
    if args.out is None:
        sys.stdout.write(text)
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"dev_recipe: wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
