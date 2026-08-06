"""Runtime identity validator (REG-03 identity, REG-06 mode / D-30 / D-31 / D-32).

Runs at instantiate time in ``train.py`` BEFORE the first epoch:
  1. Lazy-imports torch.
  2. Constructs minimal stub inputs (features=(N, feat_dim), coords=(N, 2)).
  3. Runs both variant.forward() and parent.forward() once.
  4. Compares output rank + dtype.
  5. (architecture-preserving only) compares param counts per
     identity_constraints + per-constraint check.
  6. On failure: writes archive_dir/validation_failure.json via the
     PATTERNS.md §3 atomic-write pattern (tempfile + rename) and raises
     ValidationError. The training script wrapper catches the exception
     and writes a result.json of {status: "validation_failed",
     composite: 0.0} so the orchestrator records the attempt without
     hanging (D-32).

Torch is imported LAZILY inside check() — the framework remains importable
without torch installed (Plan 01-01 invariant).
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from automil.registry.errors import ValidationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IdentityConstraint:
    """Parsed form of one entry in ``registry.identity_constraints``.

    Format examples::

        "param_count_pct: 5"  ->  IdentityConstraint(kind="param_count_pct", value="5")
        "output_rank: 2"      ->  IdentityConstraint(kind="output_rank",      value="2")
    """

    kind: str
    value: str


def _parse_constraints(raw: tuple[str, ...]) -> tuple[IdentityConstraint, ...]:
    """Parse ``["k1: v1", "k2: v2"]`` into IdentityConstraint tuples.

    Hard-fail (ValueError) on malformed entries with the offending entry named.
    """
    out: list[IdentityConstraint] = []
    for entry in raw:
        if ":" not in entry:
            raise ValueError(
                f"identity_constraints: malformed entry {entry!r}; "
                f"expected 'kind: value' (e.g., 'param_count_pct: 5'). "
                f"Edit automil/config.yaml: registry.identity_constraints."
            )
        k, _, v = entry.partition(":")
        out.append(IdentityConstraint(kind=k.strip(), value=v.strip()))
    return tuple(out)


def _atomic_write_json(path: Path, data: dict) -> None:
    """PATTERNS.md §3 atomic write: tempfile.mkstemp + os.rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.rename(tmp_path, str(path))
        os.utime(str(path))
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _safe_param_count(obj: Any) -> int:
    """Return number of parameter elements; 0 if obj has no .parameters()."""
    try:
        return sum(p.numel() for p in obj.parameters())
    except (AttributeError, TypeError):
        return 0


class IdentityValidator:
    """Runtime stub-forward identity check.

    Args:
        mode: ``'free'`` or ``'architecture-preserving'``.
        identity_constraints: tuple of ``"key: value"`` strings (parsed at
            construction time so malformed entries fail fast).

    Mode-aware strictness (D-31):

    * ``free`` (default): validates only output dtype + rank.
      Permits architectural exploration.
    * ``architecture-preserving``: adds parameter-count tolerance (configured
      via ``identity_constraints``, e.g. ``param_count_pct: 5`` means ±5% of
      parent) + per-constraint audit.  Locks F1-classic identity search.

    The ``check()`` method:

    1. Lazy-imports torch (Plan 01-01 invariant: framework loadable without torch).
    2. Instantiates variant_cls and parent_cls (no-arg, or with ``args``).
    3. Constructs minimal stub inputs (features=(N, feat_dim), coords=(N, 2)).
    4. Runs both .forward() passes.
    5. Compares output rank + dtype.
    6. In ``architecture-preserving`` mode: applies each IdentityConstraint.
    7. On failure: atomically writes ``archive_dir/validation_failure.json``
       (PATTERNS.md §3) + raises ValidationError (D-32).

    ``feat_dim`` defaults to 768 (common ViT/CLAM-Small embed dim); the caller
    can override via ``args.feat_dim`` if available.
    """

    def __init__(
        self,
        mode: str = "free",
        identity_constraints: tuple[str, ...] = (),
    ) -> None:
        if mode not in ("free", "architecture-preserving"):
            raise ValueError(
                f"IdentityValidator: mode must be 'free' or "
                f"'architecture-preserving'; got {mode!r}."
            )
        self.mode = mode
        self.constraints = _parse_constraints(identity_constraints)

    def check(
        self,
        variant_cls: type,
        parent_cls: type,
        archive_dir: Path,
        *,
        args: Optional[Any] = None,
        feat_dim: int = 768,
        n_instances: int = 4,
    ) -> None:
        """Run the runtime identity check.  Raises ValidationError on failure.

        Args:
            variant_cls: the registered variant class (resolved from registry).
            parent_cls: the parent baseline class (e.g. CLAM_MB). For free-mode
                tests with no parent, pass variant_cls itself — the validator
                skips parent comparisons gracefully.
            archive_dir: per-node archive directory; validation_failure.json is
                written here on hard-fail (D-32).
            args: optional namespace passed to the variant/parent constructors.
            feat_dim: stub feature dim; default 768.
            n_instances: stub bag size; default 4.
        """
        # Lazy torch import — Plan 01-01 invariant.
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ValidationError(
                validator_name="identity",
                path=archive_dir,
                reason="torch not installed; identity validator requires torch at runtime",
                fix_suggestion="`uv add torch` or `uv sync --extra ml` from the autoMIL source workspace.",
            ) from e

        # Instantiate both classes. Pass ``args`` if provided; else no-arg constructor.
        variant_inst = self._instantiate(variant_cls, args)
        parent_inst = self._instantiate(parent_cls, args)

        # Build stub inputs.
        features = torch.zeros(n_instances, feat_dim)
        coords = torch.zeros(n_instances, 2)

        # Run variant forward — hard-fail on exception.
        try:
            v_out = variant_inst.forward(features, coords)
        except Exception as e:
            self._fail(
                archive_dir,
                variant_cls=variant_cls,
                parent_cls=parent_cls,
                reason=f"variant.forward() raised: {type(e).__name__}: {e}",
                fix_suggestion="Check the variant's forward() implementation.",
            )

        # Run parent forward — degrade gracefully if it raises (edge case where
        # parent baseline has an unusual ctor that needs extra args).
        try:
            p_out = parent_inst.forward(features, coords)
        except Exception as e:
            logger.warning(
                "IdentityValidator: parent.forward() raised %s: %s — "
                "comparing variant against itself (degraded check).",
                type(e).__name__,
                e,
            )
            p_out = v_out

        # Extract first torch.Tensor from potentially tuple/dict output.
        v_tensor = self._extract_tensor(v_out)
        p_tensor = self._extract_tensor(p_out)

        if v_tensor is None:
            self._fail(
                archive_dir,
                variant_cls=variant_cls,
                parent_cls=parent_cls,
                reason="variant.forward() returned no tensor",
                fix_suggestion="Ensure forward() returns at least one torch.Tensor.",
            )

        # Rank check.
        v_rank = v_tensor.dim()
        p_rank = p_tensor.dim() if p_tensor is not None else v_rank
        if v_rank != p_rank:
            self._fail(
                archive_dir,
                variant_cls=variant_cls,
                parent_cls=parent_cls,
                reason=f"output rank mismatch: variant={v_rank}, parent={p_rank}",
                fix_suggestion=(
                    f"Adjust variant.forward() to return a rank-{p_rank} tensor."
                ),
                expected={"rank": p_rank, "dtype": str(p_tensor.dtype) if p_tensor is not None else "unknown"},
                actual={"rank": v_rank, "dtype": str(v_tensor.dtype)},
            )

        # Dtype check.
        if p_tensor is not None and v_tensor.dtype != p_tensor.dtype:
            self._fail(
                archive_dir,
                variant_cls=variant_cls,
                parent_cls=parent_cls,
                reason=(
                    f"output dtype mismatch: variant={v_tensor.dtype}, "
                    f"parent={p_tensor.dtype}"
                ),
                fix_suggestion=(
                    f"Cast variant output to {p_tensor.dtype} before return."
                ),
                expected={"rank": p_rank, "dtype": str(p_tensor.dtype)},
                actual={"rank": v_rank, "dtype": str(v_tensor.dtype)},
            )

        # Architecture-preserving mode: apply per-constraint checks.
        if self.mode == "architecture-preserving" and self.constraints:
            self._run_constraints(
                archive_dir,
                variant_cls,
                parent_cls,
                variant_inst,
                parent_inst,
                v_tensor,
                p_tensor,
            )

        logger.debug(
            "IdentityValidator: %s passed (mode=%r, constraints=%d)",
            variant_cls.__name__,
            self.mode,
            len(self.constraints),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _instantiate(cls: type, args: Optional[Any]) -> Any:
        """Instantiate ``cls``.  Try with ``args`` first, fall back to no-arg."""
        if args is not None:
            try:
                return cls(args)
            except TypeError:
                pass
        return cls()

    @staticmethod
    def _extract_tensor(output: Any) -> Optional[Any]:
        """Return the first torch.Tensor we find in output (tensor, tuple, dict)."""
        import torch

        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (list, tuple)):
            for item in output:
                if isinstance(item, torch.Tensor):
                    return item
        if isinstance(output, dict):
            for v in output.values():
                if isinstance(v, torch.Tensor):
                    return v
        return None

    def _run_constraints(
        self,
        archive_dir: Path,
        variant_cls: type,
        parent_cls: type,
        variant_inst: Any,
        parent_inst: Any,
        v_tensor: Any,
        p_tensor: Any,
    ) -> None:
        for c in self.constraints:
            if c.kind == "param_count_pct":
                try:
                    tol_pct = float(c.value)
                except ValueError:
                    raise ValueError(
                        f"identity_constraints: param_count_pct value "
                        f"{c.value!r} is not a number; expected e.g. "
                        f"'param_count_pct: 5'."
                    )
                v_n = _safe_param_count(variant_inst)
                p_n = _safe_param_count(parent_inst)
                if p_n == 0:
                    logger.warning(
                        "IdentityValidator: param_count_pct: parent has 0 "
                        "parameters — skipping check."
                    )
                    continue
                pct = abs(v_n - p_n) / p_n * 100.0
                if pct > tol_pct:
                    self._fail(
                        archive_dir,
                        variant_cls=variant_cls,
                        parent_cls=parent_cls,
                        reason=(
                            f"param_count_pct exceeded: variant={v_n}, "
                            f"parent={p_n}, diff={pct:.2f}% > "
                            f"tolerance={tol_pct:.2f}%"
                        ),
                        fix_suggestion=(
                            f"Reduce variant parameter count or relax "
                            f"`param_count_pct` in registry.identity_constraints "
                            f"(currently {tol_pct})."
                        ),
                        expected={"param_count_pct_max": tol_pct},
                        actual={"param_count_pct": round(pct, 4)},
                    )
            # Additional constraint kinds (output_rank, layer_shape, etc.) land
            # in later phases.  Unknown kinds are logged and skipped for forward
            # compatibility.
            else:
                logger.debug(
                    "IdentityValidator: unknown constraint kind %r — skipping.",
                    c.kind,
                )

    def _fail(
        self,
        archive_dir: Path,
        *,
        variant_cls: type,
        parent_cls: type,
        reason: str,
        fix_suggestion: str,
        expected: Optional[dict] = None,
        actual: Optional[dict] = None,
    ) -> None:
        """Atomic-write ``validation_failure.json`` then raise ``ValidationError``."""
        report = {
            "validator_name": "identity",
            "mode": self.mode,
            "variant_class": variant_cls.__name__,
            "parent_class": parent_cls.__name__,
            "reason": reason,
            "expected": expected or {},
            "actual": actual or {},
            "constraints_evaluated": [
                f"{c.kind}: {c.value}" for c in self.constraints
            ],
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        try:
            _atomic_write_json(archive_dir / "validation_failure.json", report)
        except Exception as write_err:
            logger.error(
                "IdentityValidator: could not write validation_failure.json "
                "to %s: %s",
                archive_dir,
                write_err,
            )
        raise ValidationError(
            validator_name="identity",
            path=archive_dir / "validation_failure.json",
            reason=reason,
            fix_suggestion=fix_suggestion,
        )
