"""Isolated loader for the vendored ``lib/DTFD-MIL`` reference modules.

DTFD-MIL's source uses **bare** module names: its ``utils.py`` is imported as
``import utils`` and ``Model/Attention.py`` does ``from Model.network import
Classifier_1fc``. Both ``clam/_imports.py`` and ``smmile/_imports.py`` register
a *different* ``utils`` package (``utils.utils``, ``utils.core_utils``) on
``sys.path``. Under ``max_tasks_per_child > 1`` a CLAM/SMMILe experiment can run
before a DTFD one in the same process, leaving ``sys.modules['utils']`` pointing
at the wrong package -- so a naive ``sys.path.insert`` + ``import utils`` would
silently bind the wrong module (no ``get_cam_1d``) and corrupt the arm.

To stay robust we load DTFD's four modules *by explicit file path* via
``importlib``, under private module names, behind a transient ``Model`` package
shim that is torn down (and any pre-existing ``utils``/``Model`` entries
restored) before this module finishes importing. Nothing DTFD needs leaks into
the shared namespace, and nothing already there is clobbered.
"""

from __future__ import annotations

import importlib.util
import sys
import types

from autobench import LIB_ROOT

_DTFD_DIR = LIB_ROOT / "DTFD-MIL"
_MODEL_DIR = _DTFD_DIR / "Model"


def _load_module_from_path(name: str, path) -> types.ModuleType:
    """Load a module from an explicit file path under ``name`` and register it."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load DTFD module {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_dtfd_modules() -> dict[str, object]:
    """Load DTFD's four reference modules in isolation and return their symbols.

    A transient ``Model`` package (pointing at DTFD's ``Model/`` dir) lets
    ``Model.Attention``'s ``from Model.network import Classifier_1fc`` resolve to
    DTFD's own network during load. We snapshot and restore any pre-existing
    ``Model``/``Model.network``/``Model.Attention``/``utils`` entries so a
    CLAM/SMMILe run in the same process is left exactly as we found it.
    """
    guarded = ("Model", "Model.network", "Model.Attention", "utils")
    saved = {key: sys.modules.get(key) for key in guarded}

    try:
        model_pkg = types.ModuleType("Model")
        model_pkg.__path__ = [str(_MODEL_DIR)]  # mark as a package
        sys.modules["Model"] = model_pkg

        network = _load_module_from_path("Model.network", _MODEL_DIR / "network.py")
        attention = _load_module_from_path("Model.Attention", _MODEL_DIR / "Attention.py")
        # DTFD's top-level utils.py — loaded under a private name so it never
        # collides with CLAM/SMMILe's ``utils`` package.
        dtfd_utils = _load_module_from_path("_autobench_dtfd_utils", _DTFD_DIR / "utils.py")

        return {
            "Classifier_1fc": network.Classifier_1fc,
            "DimReduction": network.DimReduction,
            "Attention_Gated": attention.Attention_Gated,
            "Attention_with_Classifier": attention.Attention_with_Classifier,
            "get_cam_1d": dtfd_utils.get_cam_1d,
        }
    finally:
        # Tear down the shim submodules we introduced, then restore snapshots.
        for key in ("Model.network", "Model.Attention"):
            sys.modules.pop(key, None)
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


_symbols = _load_dtfd_modules()

Classifier_1fc = _symbols["Classifier_1fc"]
DimReduction = _symbols["DimReduction"]
Attention_Gated = _symbols["Attention_Gated"]
Attention_with_Classifier = _symbols["Attention_with_Classifier"]
get_cam_1d = _symbols["get_cam_1d"]

__all__ = [
    "Classifier_1fc",
    "DimReduction",
    "Attention_Gated",
    "Attention_with_Classifier",
    "get_cam_1d",
]
