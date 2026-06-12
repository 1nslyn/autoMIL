#!/usr/bin/env python3
"""sklearn-iris: minimal autoMIL contract demo (DEC-02 / D-203).

Honors the DEC-06 contract: reads automil/config.yaml for data.seed,
accepts CUDA_VISIBLE_DEVICES and AUTOMIL_GPU (no-op on CPU), installs
a SIGTERM handler before compute, and writes result.json conforming to
automil/schemas/result.schema.json. No automil.* imports (consumer-decoupled).
"""
from __future__ import annotations

import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

import yaml
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

RESULTS_DIR = Path(".")  # write to cwd (= worktree when launched by orchestrator)
_state: dict[str, Any] = {"completed": False, "accuracy": 0.0, "f1": 0.0, "variant_dispatched": None}


def _write_result(*, status: str, partial: bool, variant_dispatched: str | None = None) -> None:
    """Write result.json conforming to automil/schemas/result.schema.json."""
    payload: dict = {
        "status": status,
        "composite": float(_state["accuracy"]),
        "metrics": {
            "accuracy": float(_state["accuracy"]),
            "f1": float(_state["f1"]),
        },
        "partial": partial,
    }
    if variant_dispatched is not None:
        payload["variant_dispatched"] = variant_dispatched
    (RESULTS_DIR / "result.json").write_text(json.dumps(payload, indent=2))


def _sigterm_handler(signum: int, frame: object) -> None:
    """SIGTERM clean exit. Idempotent: late SIGTERM after completion writes status=completed."""
    if _state["completed"]:
        _write_result(status="completed", partial=False, variant_dispatched=_state["variant_dispatched"])
    else:
        _write_result(status="budget_killed", partial=True, variant_dispatched=_state["variant_dispatched"])
    sys.exit(0)  # NOT sys.exit(130); 0 signals graceful flush to daemon.


def main() -> None:
    """Train LogisticRegression on iris; write result.json."""
    signal.signal(signal.SIGTERM, _sigterm_handler)  # install before any compute

    # Honor CUDA_VISIBLE_DEVICES + AUTOMIL_GPU (no-op on CPU, D-204 items 2-3).
    _ = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    _ = os.environ.get("AUTOMIL_GPU", "")

    config_path = Path("automil/config.yaml")  # D-204 contract item 1
    seed = 42
    config: dict = {}
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text()) or {}
        seed = int((config.get("data") or {}).get("seed", 42))

    # APL-01: resolve variant name from applied_variant.json (A1 fix: propagated into
    # worktree by apply_overlay), then config.yaml model.variant, then env fallback.
    # No automil.* imports — consumer-decoupled contract (train.py:7). Pure stdlib.
    variant_name: str | None = None
    applied_path = Path("automil/applied_variant.json")
    if applied_path.exists():
        _sel = json.loads(applied_path.read_text()) or {}
        variant_name = (_sel.get("model") or {}).get("variant")
    if not variant_name:
        variant_name = (config.get("model") or {}).get("variant")
    if not variant_name:
        variant_name = os.environ.get("AUTOMIL_VARIANT_MODEL")

    # Path-traversal guard (T-10-08): variant_name must be a plain name, no separators.
    if variant_name is not None:
        import pathlib as _pl
        if _pl.Path(variant_name).name != variant_name:
            raise ValueError(f"Invalid variant name (path traversal): {variant_name!r}")

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )

    # Dispatch: load the variant module via raw importlib (no automil.* imports).
    clf = None
    if variant_name:
        import importlib.util as _ilu
        _variants_dir = Path("automil/variants") / variant_name
        _py_files = sorted(
            f for f in _variants_dir.glob("*.py")
            if not f.name.startswith("_") and f.name != "__init__.py"
        )
        if _py_files:
            _spec = _ilu.spec_from_file_location(variant_name, _py_files[0])
            if _spec is None or _spec.loader is None:
                _write_result(status="crash", partial=False, variant_dispatched=variant_name)
                raise ValueError(
                    f"Could not load variant module at {_py_files[0]}: "
                    "importlib returned no spec or loader. "
                    "Ensure the file has a .py extension and is a valid Python module."
                )
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            if not hasattr(_mod, "make_classifier"):
                _write_result(status="crash", partial=False, variant_dispatched=variant_name)
                raise AttributeError(
                    f"Variant module {_py_files[0]} has no make_classifier() function."
                )
            clf = _mod.make_classifier(seed=seed)
    if clf is None:
        variant_name = None  # baseline path — clear for result.json marker
        clf = LogisticRegression(max_iter=200, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    _state["accuracy"] = float(accuracy_score(y_test, y_pred))
    _state["f1"] = float(f1_score(y_test, y_pred, average="macro"))
    _state["variant_dispatched"] = variant_name  # None for baseline path
    _state["completed"] = True

    _write_result(status="completed", partial=False, variant_dispatched=variant_name)


if __name__ == "__main__":
    main()
