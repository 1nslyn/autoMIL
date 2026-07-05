"""Isolated loader for the vendored ``lib/AttentionDeepMIL`` reference module.

``lib/AttentionDeepMIL/model.py`` is a single flat module with no internal
package imports (unlike DTFD-MIL's ``utils``/``Model`` tangle), so there is no
risk of it clobbering another arm's same-named modules. Still, we load it by
explicit file path via ``importlib`` (mirroring ``dtfd/_imports.py``'s
isolation approach) rather than inserting ``lib/AttentionDeepMIL`` onto
``sys.path`` and doing a bare ``import model`` -- that would register a
generic top-level ``model`` module name that could collide with another
vendored lib's own ``model.py`` under ``max_tasks_per_child > 1``.
"""

from __future__ import annotations

import importlib.util
import sys
import types

from autobench import LIB_ROOT

_ATTENTION_DEEPMIL_DIR = LIB_ROOT / "AttentionDeepMIL"
_MODEL_PATH = _ATTENTION_DEEPMIL_DIR / "model.py"


def _load_module_from_path(name: str, path) -> types.ModuleType:
    """Load a module from an explicit file path under ``name`` and register it."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load AttentionDeepMIL module {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_attention_deepmil_module() -> dict[str, object]:
    """Load AttentionDeepMIL's ``model.py`` in isolation and return its symbols.

    Loaded under a private module name (``_autobench_attention_deepmil_model``)
    so it never collides with another vendored lib's ``model`` module in the
    same process.
    """
    module = _load_module_from_path(
        "_autobench_attention_deepmil_model", _MODEL_PATH
    )
    return {
        "Attention": module.Attention,
        "GatedAttention": module.GatedAttention,
    }


_symbols = _load_attention_deepmil_module()

Attention = _symbols["Attention"]
GatedAttention = _symbols["GatedAttention"]

__all__ = ["Attention", "GatedAttention"]
