"""Identity of the per-fold results cache (CR-5b).

Every trainer resumes a fold by checking whether
``<results_dir>/fold_N/metrics.json`` exists. CR-5 gave the *orchestrated* path an
isolated ``AUTOMIL_RESULTS_DIR``, but the static grid never sets it, so all five
runners fell back to
``benchmark_dir/results/{framework}/{strategy}/{task}/{encoder}/{model}[/{loss}]``
-- a path built by :attr:`ExperimentConfig.results_subdir`, which carried **no
seed and no hyperparameter**. Two silent failures followed:

* ``--seed 43`` after a seed-42 grid returned seed 42's numbers verbatim, so a
  multi-seed variance study would report zero variance.
* Re-running after correcting a learning rate returned the old numbers.

Seed identity and recipe identity require different cache behavior.

*Seed* belongs in the **path**: seeds are meant to coexist, and evicting seed 42 to
run seed 43 would make multi-seed impossible rather than merely wrong.

*Everything else* is fingerprinted into a sidecar, and a mismatch **fails loudly**
rather than self-healing. That follows the precedent set by the task-CSV cache
guard (PRELAUNCH_REVIEW B2): ``benchmark_dir`` is shared across concurrently
running experiments, and a cache that purges itself can delete folds another
process is training from. Printing the purge command and stopping is the safe
half of the trade.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

__all__ = [
    "FINGERPRINT_FILENAME",
    "StaleResultsCacheError",
    "config_fingerprint",
    "fingerprint_payload",
    "resolve_results_dir",
]

#: Sidecar written next to the ``fold_N/`` directories it describes.
FINGERPRINT_FILENAME = "config_fingerprint.json"

#: Fields excluded from the fingerprint because they cannot change the numbers.
#: ``survival_losses`` is the task's *configured menu*; the loss actually trained
#: is ``ExperimentConfig.survival_loss``, which is both in the path and in the
#: fingerprint. Including the menu would evict unrelated caches whenever a loss is
#: added to a dataset YAML.
_TASK_KEYS_NOT_IN_FINGERPRINT = ("survival_losses",)


class StaleResultsCacheError(RuntimeError):
    """A results directory was produced by a different configuration.

    Raised instead of resuming, because resuming would silently report the old
    configuration's numbers under the new configuration's label.
    """


def _as_plain(obj: Any) -> dict[str, Any] | None:
    """Dataclass / mapping -> plain JSON-able dict."""
    if obj is None:
        return None
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, dict):
        return dict(obj)
    raise TypeError(
        f"arm_cfg must be a dataclass instance or a mapping, got {type(obj).__name__}"
    )


def fingerprint_payload(
    exp_cfg: Any, arm_cfg: Any | None = None,
) -> dict[str, Any]:
    """The configuration this results directory belongs to.

    Args:
        exp_cfg: the experiment config.
        arm_cfg: the arm's own config dataclass (``DTFDConfig``, ``ABMILConfig``,
            ``TitanHeadConfig``) or nnMIL's computed plan dict. These hold knobs
            that live *outside* ``exp_cfg`` -- DTFD's ``numGroup``, ABMIL's ``M`` --
            so a change there must invalidate the cache just the same.
    """
    payload = exp_cfg.to_dict()
    task = payload.get("task")
    if isinstance(task, dict):
        for key in _TASK_KEYS_NOT_IN_FINGERPRINT:
            task.pop(key, None)
    if arm_cfg is not None:
        payload["arm"] = _as_plain(arm_cfg)
    return payload


def config_fingerprint(exp_cfg: Any, arm_cfg: Any | None = None) -> str:
    """Stable digest of everything that can change this experiment's numbers."""
    blob = json.dumps(
        fingerprint_payload(exp_cfg, arm_cfg), sort_keys=True, default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        out[prefix] = obj
    return out


def _changed_fields(
    old: Mapping[str, Any], new: Mapping[str, Any],
) -> list[str]:
    """Dotted field names that differ, for an error message that says *what* moved."""
    flat_old, flat_new = _flatten(old), _flatten(new)
    keys = sorted(set(flat_old) | set(flat_new))
    return [k for k in keys if flat_old.get(k) != flat_new.get(k)]


def _write_atomic(path: str, payload: Mapping[str, Any]) -> None:
    """Concurrent experiments share ``benchmark_dir``; never leave a torn file."""
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".fingerprint-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=str)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _enforce_fingerprint(
    results_dir: str, exp_cfg: Any, arm_cfg: Any | None,
) -> None:
    path = os.path.join(results_dir, FINGERPRINT_FILENAME)
    payload = fingerprint_payload(exp_cfg, arm_cfg)
    digest = config_fingerprint(exp_cfg, arm_cfg)

    if os.path.exists(path):
        try:
            with open(path) as f:
                stored = json.load(f)
        except (OSError, json.JSONDecodeError):
            stored = None  # unreadable sidecar: re-stamp rather than block
        if stored is not None:
            if stored.get("digest") == digest:
                return
            changed = _changed_fields(stored.get("config", {}), payload)
            raise StaleResultsCacheError(
                f"{results_dir} already holds per-fold results produced by a "
                f"DIFFERENT configuration.\n"
                f"  changed: {', '.join(changed) or '(digest differs)'}\n"
                f"Resuming would report the old configuration's numbers under this "
                f"run's label (CR-5b). Purge it and re-run:\n"
                f"    rm -rf {results_dir}\n"
                f"or point this experiment at its own directory via "
                f"AUTOMIL_RESULTS_DIR."
            )

    _write_atomic(path, {"digest": digest, "config": payload})


def resolve_results_dir(
    exp_cfg: Any,
    benchmark_dir: str,
    results_dir: str | None = None,
    *,
    arm_cfg: Any | None = None,
) -> str:
    """Resolve an experiment's results directory and refuse a stale cache.

    Args:
        exp_cfg: the experiment config.
        benchmark_dir: shared benchmark root, used only for the fallback path.
        results_dir: an explicit directory (the orchestrated path passes
            ``AUTOMIL_RESULTS_DIR``); honoured as-is, but still fingerprinted.
        arm_cfg: the arm's own config, if it has one outside ``exp_cfg``.

    Raises:
        StaleResultsCacheError: the directory holds results from a different
            configuration.
    """
    if results_dir is None:
        results_dir = os.path.join(benchmark_dir, "results", exp_cfg.results_subdir)
    os.makedirs(results_dir, exist_ok=True)
    _enforce_fingerprint(results_dir, exp_cfg, arm_cfg)
    return results_dir
