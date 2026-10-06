"""Consumer-side dispatch for autoMIL's train-only ``PolicyVariant`` seam.

The autoMIL framework only records and transports a selected policy.  This
module is the benchmark consumer's single adapter: it resolves that selection,
instantiates the registered policy, and exposes its guarded operations to all
five MIL arms: optimizer and scheduler wrapping, stopping, the training-bag
transform and the pre-validation hook.  Model construction, forward paths,
defining losses, validation, and result writing remain in protected trainers
and are never passed in.
"""
from __future__ import annotations

import hashlib
import json
import os
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from autobench.pipeline.selection import smoothed_selection_value


def _clone_function(
    function: types.FunctionType,
    globals_dict: dict[str, Any],
) -> types.FunctionType:
    """Clone a function onto a fold-local module-global namespace."""
    cloned = types.FunctionType(
        function.__code__,
        globals_dict,
        name=function.__name__,
        argdefs=function.__defaults__,
        closure=function.__closure__,
    )
    cloned.__kwdefaults__ = (
        dict(function.__kwdefaults__) if function.__kwdefaults__ else None
    )
    cloned.__annotations__ = dict(function.__annotations__)
    cloned.__dict__.update(function.__dict__)
    cloned.__doc__ = function.__doc__
    cloned.__module__ = function.__module__
    cloned.__qualname__ = function.__qualname__
    return cloned


def _fold_local_class(
    source: type[Any],
    globals_dict: dict[str, Any],
    *,
    suffix: str,
) -> type[Any]:
    """Subclass one module-local class with cloned methods/globals."""
    namespace: dict[str, Any] = {"__module__": source.__module__}
    for name, value in source.__dict__.items():
        if isinstance(value, types.FunctionType):
            namespace[name] = _clone_function(value, globals_dict)
        elif isinstance(value, staticmethod):
            namespace[name] = staticmethod(_clone_function(value.__func__, globals_dict))
        elif isinstance(value, classmethod):
            namespace[name] = classmethod(_clone_function(value.__func__, globals_dict))
        elif isinstance(value, property):
            namespace[name] = property(
                _clone_function(value.fget, globals_dict) if value.fget else None,
                _clone_function(value.fset, globals_dict) if value.fset else None,
                _clone_function(value.fdel, globals_dict) if value.fdel else None,
                value.__doc__,
            )
    return type(f"{source.__name__}{suffix}", (source,), namespace)


def _fold_local_policy_type(factory: type[Any]) -> type[Any]:
    """Clone a policy's module-local function/class namespace for one fold."""
    # Local test/programmatic classes have closure state rather than a variant
    # module namespace. A unique subclass still isolates their class rebinding.
    if "<locals>" in factory.__qualname__:
        return type(
            f"{factory.__name__}FoldLocal",
            (factory,),
            {"__module__": factory.__module__},
        )

    source_globals = next((
        value.__globals__
        for value in factory.__dict__.values()
        if isinstance(value, types.FunctionType)
    ), None)
    if source_globals is None:
        source_globals = {"__builtins__": __builtins__}
    isolated_globals = dict(source_globals)

    # Clone module-local helpers first. Their functions all share the same new
    # globals dict, so aliases resolve to the fold-local helper objects below.
    for name, value in list(source_globals.items()):
        if (
            isinstance(value, types.FunctionType)
            and value.__module__ == factory.__module__
        ):
            isolated_globals[name] = _clone_function(value, isolated_globals)
    for name, value in list(source_globals.items()):
        if (
            isinstance(value, type)
            and value is not factory
            and value.__module__ == factory.__module__
            and "<locals>" not in value.__qualname__
        ):
            isolated_globals[name] = _fold_local_class(
                value, isolated_globals, suffix="FoldLocal",
            )

    policy_type = _fold_local_class(
        factory, isolated_globals, suffix="FoldLocal",
    )
    isolated_globals[factory.__name__] = policy_type
    return policy_type


def _scalar_repr(value: Any) -> str:
    """Full-precision canonical rendering for the per-epoch line.

    ``repr(float(...))`` is the shortest round-trip form, and normalizes numpy
    scalars (whose repr grew a type prefix in numpy 2) so the log format does
    not depend on which library produced the metric. Non-numeric values fall
    back to plain repr rather than raising — this line is logging, and must
    never take a training run down.
    """
    try:
        return repr(float(value))
    except (TypeError, ValueError):
        return repr(value)


def _is_real_scalar(value: Any) -> bool:
    """A number the history can hold: numpy scalars yes, bools and strings no."""
    if isinstance(value, bool):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return not isinstance(value, (str, bytes))


def runtime_automil_dir() -> Path:
    """Return this worktree's configured autoMIL directory.

    The orchestrator launches from the git-worktree root even when the consumer
    project is nested.  ``AUTOMIL_DIR_REL`` is therefore the only unambiguous
    runtime locator; the root-level fallback preserves standalone projects.
    """
    raw = os.environ.get("AUTOMIL_DIR_REL", "automil")
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"invalid AUTOMIL_DIR_REL {raw!r}")
    return path


def _policy_from_applied_variant(automil_dir: Path) -> str | None:
    path = automil_dir / "applied_variant.json"
    if not path.exists():
        return None
    try:
        selection = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(selection, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    section = selection.get("policy") or {}
    if not isinstance(section, Mapping):
        raise ValueError(f"{path}: policy must be a JSON object")
    name = section.get("variant")
    if name is None:
        return None
    if not isinstance(name, str) or not name.strip() or Path(name).name != name:
        raise ValueError(f"{path}: invalid policy variant name {name!r}")
    return name


def resolve_policy_name(exp_cfg: Any, automil_dir: Path | None = None) -> str | None:
    """Resolve explicit and archived selection, rejecting disagreement."""
    automil_dir = automil_dir or runtime_automil_dir()
    explicit = getattr(exp_cfg, "policy_variant", None)
    archived = _policy_from_applied_variant(automil_dir)
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip() or Path(explicit).name != explicit:
            raise ValueError(f"invalid policy variant name {explicit!r}")
    if explicit and archived and explicit != archived:
        raise ValueError(
            "policy selection disagrees between ExperimentConfig and "
            f"applied_variant.json ({explicit!r} != {archived!r})"
        )
    return explicit or archived


def defines_seam(policy_cls: type, seam: str) -> bool:
    """True when ``policy_cls`` defines ``seam`` itself rather than inheriting
    the no-op from ``PolicyVariant``."""
    from automil.registry.variants.policy import PolicyVariant

    method = getattr(policy_cls, seam, None)
    return method is not None and method is not getattr(PolicyVariant, seam)


@dataclass
class PolicyRuntime:
    """Fail-loud, fold-local wrapper around one optional ``PolicyVariant``.

    Selected policy classes are resolved once but instantiated lazily.  Every
    trainer seeds its fold before the first guarded policy operation, so even a
    randomized policy constructor is isolated from the preceding fold's RNG
    state.
    """

    name: str | None = None
    policy: Any | None = None
    policy_factory: type[Any] | None = None
    # The fold's training seed and index: they seed each overridden seam's
    # private generator, so a transform is reproducible and draws nothing from
    # the global RNGs that drive shuffling, dropout and sampling.
    seed: int | None = None
    fold: int | None = None
    # Every evaluated epoch's scalar metrics, in call order, for this fold only.
    history: list[tuple[int, dict[str, float]]] = field(default_factory=list)
    _generators: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_experiment(
        cls,
        exp_cfg: Any,
        automil_dir: Path | None = None,
    ) -> "PolicyRuntime":
        automil_dir = automil_dir or runtime_automil_dir()
        name = resolve_policy_name(exp_cfg, automil_dir)
        if name is None:
            return cls()

        from automil.registry import resolve_policy
        from automil.registry.scanner import scan_variants

        variants_root = automil_dir / "variants"
        scan_result = scan_variants(variants_root)
        try:
            policy_cls = resolve_policy(name)
        except KeyError as exc:
            failures = "; ".join(
                f"{path.name}: {reason}" for path, reason in scan_result.failed
            )
            detail = f" Import failures: {failures}" if failures else ""
            raise ValueError(
                f"selected policy variant {name!r} was not registered from "
                f"{variants_root}.{detail}"
            ) from exc
        return cls(name=name, policy_factory=policy_cls)

    def for_fold(
        self, *, seed: int | None = None, fold: int | None = None,
    ) -> "PolicyRuntime":
        """Return a fresh runtime: no policy instance or epoch history crosses folds."""
        factory = self.policy_factory
        if factory is None and self.policy is not None:
            factory = type(self.policy)
        return type(self)(name=self.name, policy_factory=factory, seed=seed, fold=fold)

    def _generator(self, seam: str) -> Any:
        """The seam's private CPU generator for this fold, built on first use."""
        if seam not in self._generators:
            if self.seed is None or self.fold is None:
                raise RuntimeError(
                    f"policy {self.name!r} overrides {seam}() but the runtime was "
                    "not given the fold's seed and index (for_fold(seed=, fold=))"
                )
            import torch

            digest = hashlib.sha256(
                f"{int(self.seed)}:{int(self.fold)}:{seam}".encode()
            ).digest()
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int.from_bytes(digest[:8], "big") >> 1)
            self._generators[seam] = generator
        return self._generators[seam]

    def smoothed(self, selected_epoch: int, key: str) -> float | None:
        """This fold's selection score for ``key`` around ``selected_epoch``."""
        curve = {epoch: metrics[key] for epoch, metrics in self.history if key in metrics}
        return smoothed_selection_value(curve, selected_epoch)

    def _resolved_policy(self) -> Any | None:
        """Instantiate once, at first use inside an already-seeded trainer."""
        if self.policy is None and self.policy_factory is not None:
            self.policy = _fold_local_policy_type(self.policy_factory)()
        return self.policy

    def wrap_optimizer(self, optimizer: Any, *, role: str = "main") -> Any:
        policy = self._resolved_policy()
        if policy is None:
            return optimizer
        wrapped = policy.wrap_optimizer_for(optimizer, role=role)
        if wrapped is None:
            raise TypeError(f"policy {self.name!r} returned None for optimizer role {role!r}")
        for method in ("zero_grad", "step"):
            if not callable(getattr(wrapped, method, None)):
                raise TypeError(
                    f"policy {self.name!r} returned an invalid optimizer for role "
                    f"{role!r}: missing callable {method}()"
                )
        return wrapped

    def scheduler_target(
        self, wrapped: Any, original: Any, *, role: str = "main",
    ) -> Any:
        """The optimizer a ``torch`` LR scheduler must attach to.

        ``PolicyVariant.wrap_optimizer`` is documented to "wrap (or replace)"
        the optimizer, and wrapping is the point of the seam — Lookahead,
        gradient clipping, a custom schedule inside ``step()``. But
        ``LRScheduler.__init__`` does a strict ``isinstance(..., Optimizer)``
        check, so handing it a duck-typed wrapper raises
        ``TypeError: X is not an Optimizer`` from deep inside torch, naming
        nothing useful.

        Both intents are legitimate and need different answers:

        * **replaced** with a different real ``Optimizer`` — schedule that one.
        * **wrapped** around the original — schedule the *original*. A wrapper
          delegates to the optimizer it holds and shares its ``param_groups``,
          which is the only state a scheduler mutates, so the schedule still
          reaches the wrapper. This is how torch's own ecosystem attaches
          schedulers to wrapped optimizers.

        Raises:
            TypeError: neither object is a real ``Optimizer``, so no scheduler
                can be built — reported here, naming the policy and role,
                rather than as an opaque failure inside torch.
        """
        import torch

        if isinstance(wrapped, torch.optim.Optimizer):
            return wrapped
        if isinstance(original, torch.optim.Optimizer):
            return original
        raise TypeError(
            f"policy {self.name!r} returned an optimizer for role {role!r} that "
            f"no LR scheduler can attach to: neither the wrapped object "
            f"({type(wrapped).__name__}) nor the original "
            f"({type(original).__name__}) is a torch.optim.Optimizer."
        )

    def wrap_scheduler(self, scheduler: Any, *, role: str = "main") -> Any:
        policy = self._resolved_policy()
        if policy is None:
            return scheduler
        wrapped = policy.wrap_scheduler_for(scheduler, role=role)
        if wrapped is None or not callable(getattr(wrapped, "step", None)):
            raise TypeError(
                f"policy {self.name!r} returned an invalid scheduler for role {role!r}"
            )
        return wrapped

    def should_stop(
        self,
        default: bool,
        *,
        epoch: int,
        metrics: Mapping[str, float] | None = None,
    ) -> bool:
        # A3: one measurement-neutral trajectory line per epoch, policy or not.
        # This dispatch is the only per-epoch seam shared by the five arms, so
        # printing here makes every run's trajectory readable from run.log
        # without a charged probe. EXACTLY the metrics the trainer already
        # passed — nothing is evaluated here; full-precision floats so equal
        # runs stay bit-comparable. Flushed: these lines are the forensic
        # record a timeout SIGKILL must not swallow.
        rendered = " ".join(
            f"{key}={_scalar_repr(value)}" for key, value in (metrics or {}).items()
        )
        print(f"[epoch {int(epoch)}]" + (f" {rendered}" if rendered else ""), flush=True)
        self.history.append((int(epoch), {
            str(key): float(value)
            for key, value in (metrics or {}).items()
            if _is_real_scalar(value)
        }))
        policy = self._resolved_policy()
        if policy is None:
            return bool(default)
        safe_metrics: dict[str, float] = {}
        for key, value in (metrics or {}).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"policy metric {key!r} must be a real scalar")
            safe_metrics[str(key)] = float(value)
        decision = policy.should_stop(
            default=bool(default), epoch=int(epoch), metrics=safe_metrics,
        )
        if type(decision) is not bool:
            raise TypeError(f"policy {self.name!r} should_stop() must return bool")
        return decision

    def transform_bag(
        self,
        features: Any,
        *,
        label: Any,
        epoch: int,
        min_instances: int = 1,
        keep_shape: bool = False,
    ) -> Any:
        """The training bag the model sees: ``features`` itself unless a policy transforms it.

        ``features`` is one slide's ``[N, D]`` tensor. The arm states what its
        model needs: ``min_instances`` (CLAM samples eight instances per class
        branch) and ``keep_shape`` (a slide-embedding arm has one vector per
        slide). Without an overriding policy this returns the same object and
        draws no random number, so a native run is unchanged.
        """
        policy = self._resolved_policy()
        if policy is None or not defines_seam(type(policy), "transform_bag"):
            return features
        import torch

        version = features._version
        out = policy.transform_bag(
            features, label=label, epoch=int(epoch),
            generator=self._generator("transform_bag"),
        )
        if features._version != version:
            raise RuntimeError(f"policy {self.name!r} modified a training bag in place")
        if (
            not isinstance(out, torch.Tensor)
            or out.dtype != features.dtype
            or out.device != features.device
            or out.ndim != features.ndim
            or out.shape[1:] != features.shape[1:]
        ):
            raise TypeError(
                f"policy {self.name!r} transform_bag() must return a tensor with the "
                f"bag's dtype, device and feature shape {tuple(features.shape[1:])}"
            )
        if keep_shape and out.shape != features.shape:
            raise TypeError(
                f"policy {self.name!r} transform_bag() must keep this arm's bag shape "
                f"{tuple(features.shape)}"
            )
        if out.shape[0] < min_instances:
            raise ValueError(
                f"policy {self.name!r} transform_bag() left {out.shape[0]} instances; "
                f"this arm needs at least {min_instances}"
            )
        return out

    def before_validation(self, *, epoch: int) -> None:
        """Run the policy's pre-validation hook, if it defines one."""
        policy = self._resolved_policy()
        if policy is None or not defines_seam(type(policy), "before_validation"):
            return
        if policy.before_validation(epoch=int(epoch)) is not None:
            raise TypeError(f"policy {self.name!r} before_validation() must return None")
