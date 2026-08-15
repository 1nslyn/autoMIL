"""Candidate admissibility as one fail-closed framework interface.

The interface is intentionally small: load one :class:`CandidatePolicy`, call
``classify`` at submit time, persist the returned verdict, and call
``revalidate`` immediately before launch. Path matching, policy hashing, class
assignment, and rejection reasons therefore have one implementation.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shlex
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping

import yaml


class AdmissibilityError(ValueError):
    """A candidate cannot be proven admissible under the active policy."""


class CandidateClass(str, Enum):
    """Machine-readable relation between a candidate and the search contract."""

    CONFIG_ONLY = "config-only"
    TRAIN_ONLY_SOURCE = "train-only-source"
    FREE_SOURCE = "free-source"
    PROTECTED_SURFACE_VIOLATION = "protected-surface-violation"
    INVALID = "invalid"


def matches_scope(path: str, patterns: Iterable[str]) -> bool:
    """Match exact paths, directory prefixes, and glob patterns."""
    rel_path = Path(path).as_posix()
    for raw_pattern in patterns:
        pattern = str(raw_pattern).strip().replace("\\", "/")
        if not pattern:
            continue
        if pattern.endswith("/"):
            if rel_path.startswith(pattern):
                return True
            continue
        if fnmatch.fnmatch(rel_path, pattern):
            return True
    return False


@dataclass(frozen=True)
class CandidateVerdict:
    """Complete persisted outcome of evaluating one candidate."""

    candidate_class: CandidateClass
    accepted: bool
    files: tuple[str, ...]
    reason: str
    matched_patterns: tuple[str, ...]
    variant_kinds: tuple[str, ...]
    variant_selection_hash: str
    override_hash: str
    policy_hash: str

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_class": self.candidate_class.value,
            "accepted": self.accepted,
            "files": list(self.files),
            "reason": self.reason,
            "matched_patterns": list(self.matched_patterns),
            "variant_kinds": list(self.variant_kinds),
            "variant_selection_hash": self.variant_selection_hash,
            "override_hash": self.override_hash,
            "policy_hash": self.policy_hash,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "CandidateVerdict":
        try:
            accepted = raw["accepted"]
            files = raw["files"]
            matched = raw.get("matched_patterns", [])
            kinds = raw.get("variant_kinds", [])
            if type(accepted) is not bool:
                raise TypeError("accepted must be a JSON boolean")
            if not isinstance(files, list) or not all(isinstance(p, str) for p in files):
                raise TypeError("files must be a list of strings")
            if not isinstance(matched, list) or not all(isinstance(p, str) for p in matched):
                raise TypeError("matched_patterns must be a list of strings")
            if not isinstance(kinds, list) or not all(isinstance(p, str) for p in kinds):
                raise TypeError("variant_kinds must be a list of strings")
            return cls(
                candidate_class=CandidateClass(str(raw["candidate_class"])),
                accepted=accepted,
                files=tuple(files),
                reason=str(raw.get("reason", "")),
                matched_patterns=tuple(matched),
                variant_kinds=tuple(kinds),
                variant_selection_hash=str(raw.get("variant_selection_hash", "")),
                override_hash=str(raw.get("override_hash", "")),
                policy_hash=str(raw["policy_hash"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AdmissibilityError(
                f"invalid recorded admissibility verdict: {exc}"
            ) from exc


@dataclass(frozen=True)
class CandidatePolicy:
    """Classify and later revalidate candidates under one search mode."""

    mode: str = "free"
    editable: tuple[str, ...] = ()
    protected: tuple[str, ...] = ()
    allowed_override_options: tuple[str, ...] = ()
    allowed_variant_kinds: tuple[str, ...] = ()
    identity_locked_hparams: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in {"free", "architecture-preserving"}:
            raise ValueError(f"unknown candidate-policy mode {self.mode!r}")

    @property
    def policy_hash(self) -> str:
        payload = {
            "mode": self.mode,
            "editable": sorted(self.editable),
            "protected": sorted(self.protected),
            "allowed_override_options": sorted(self.allowed_override_options),
            "allowed_variant_kinds": sorted(self.allowed_variant_kinds),
            "identity_locked_hparams": sorted(self.identity_locked_hparams),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _verdict(
        self,
        candidate_class: CandidateClass,
        accepted: bool,
        files: tuple[str, ...],
        reason: str = "",
        matched_patterns: tuple[str, ...] = (),
        variant_kinds: tuple[str, ...] = (),
        variant_selection_hash: str = "",
        override_hash: str = "",
    ) -> CandidateVerdict:
        return CandidateVerdict(
            candidate_class=candidate_class,
            accepted=accepted,
            files=files,
            reason=reason,
            matched_patterns=matched_patterns,
            variant_kinds=variant_kinds,
            variant_selection_hash=variant_selection_hash,
            override_hash=override_hash,
            policy_hash=self.policy_hash,
        )

    def _classify_variant_selection(
        self,
        selection: Mapping[str, object] | None,
        files: tuple[str, ...],
    ) -> tuple[CandidateVerdict | None, tuple[str, ...], str]:
        if not selection:
            return None, (), ""
        selected: dict[str, Mapping[str, object]] = {}
        for kind in ("model", "loss", "policy"):
            section = selection.get(kind)
            if section is None:
                continue
            if not isinstance(section, Mapping):
                return (
                    self._verdict(
                        CandidateClass.INVALID,
                        False,
                        files,
                        f"variant selection section {kind!r} is not a mapping",
                    ),
                    (),
                    "",
                )
            variant = section.get("variant")
            if variant is None:
                continue
            if not isinstance(variant, str) or not variant.strip():
                return (
                    self._verdict(
                        CandidateClass.INVALID,
                        False,
                        files,
                        f"variant selection {kind}.variant must be a non-empty string",
                    ),
                    (),
                    "",
                )
            selected[kind] = dict(section)
        if not selected:
            return None, (), ""

        kinds = tuple(sorted(selected))
        encoded = json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
        selection_hash = hashlib.sha256(encoded).hexdigest()
        if self.mode == "architecture-preserving":
            denied = tuple(kind for kind in kinds if kind not in self.allowed_variant_kinds)
            if denied:
                return (
                    self._verdict(
                        CandidateClass.PROTECTED_SURFACE_VIOLATION,
                        False,
                        files,
                        f"variant kind(s) {list(denied)} are outside "
                        "registry.allowed_variant_kinds "
                        f"{list(self.allowed_variant_kinds)}",
                        variant_kinds=kinds,
                        variant_selection_hash=selection_hash,
                    ),
                    kinds,
                    selection_hash,
                )
        return None, kinds, selection_hash

    def _classify_override(
        self, override: str | None, files: tuple[str, ...],
    ) -> CandidateVerdict | None:
        if override is None:
            return None
        try:
            tokens = shlex.split(override)
        except ValueError as exc:
            return self._verdict(
                CandidateClass.INVALID,
                False,
                files,
                f"cannot parse run-command override: {exc}",
            )
        if not tokens:
            return self._verdict(
                CandidateClass.INVALID,
                False,
                files,
                "run-command override is empty",
            )
        if self.mode == "free":
            return None
        try:
            options = self._parse_override_options(override)
        except ValueError as exc:
            return self._verdict(
                CandidateClass.INVALID,
                False,
                files,
                f"cannot parse run-command override: {exc}",
            )

        if not options:
            return self._verdict(
                CandidateClass.INVALID,
                False,
                files,
                "architecture-preserving override contains no named option",
            )
        denied = tuple(
            option for option in options
            if option not in self.allowed_override_options
        )
        if denied:
            return self._verdict(
                CandidateClass.PROTECTED_SURFACE_VIOLATION,
                False,
                files,
                "run-command option(s) "
                f"{list(denied)} are outside registry.allowed_override_options "
                f"{list(self.allowed_override_options)}",
            )

        # ``--hparams`` is intentionally an opaque arm-specific channel, but it
        # is not allowed to become an identity escape hatch.  Consumers name the
        # few scalar keys that can erase a defining mechanism (for example,
        # disabling CLAM instance clustering).  Parse the exact command that will
        # be launched and reject those keys before a node consumes GPU time.
        hparams_value = options.get("--hparams")
        if "--hparams" in options and hparams_value is None:
            return self._verdict(
                CandidateClass.INVALID,
                False,
                files,
                "--hparams requires a quoted JSON-object value",
            )
        if "--policy-variant" in options and options["--policy-variant"] is None:
            return self._verdict(
                CandidateClass.INVALID,
                False,
                files,
                "--policy-variant requires a value",
            )
        if hparams_value is not None:
            try:
                hparams = json.loads(hparams_value)
            except json.JSONDecodeError as exc:
                return self._verdict(
                    CandidateClass.INVALID,
                    False,
                    files,
                    "--hparams must be a quoted JSON object: " + str(exc),
                )
            if not isinstance(hparams, dict):
                return self._verdict(
                    CandidateClass.INVALID,
                    False,
                    files,
                    "--hparams must decode to a JSON object",
                )
            bad_values = sorted(
                str(key) for key, value in hparams.items()
                if not isinstance(key, str) or isinstance(value, (dict, list))
            )
            if bad_values:
                return self._verdict(
                    CandidateClass.INVALID,
                    False,
                    files,
                    "--hparams must map string keys to scalar values; invalid "
                    f"key(s) {bad_values}",
                )
            locked = tuple(
                sorted(set(hparams).intersection(self.identity_locked_hparams))
            )
            if locked:
                return self._verdict(
                    CandidateClass.PROTECTED_SURFACE_VIOLATION,
                    False,
                    files,
                    "hyperparameter key(s) "
                    f"{list(locked)} are locked by registry.identity_locked_hparams "
                    "because changing them can erase the arm's defining mechanism",
                )
        return None

    @staticmethod
    def _parse_override_options(override: str) -> dict[str, str | None]:
        """Parse one option-only override into a deterministic semantic map."""
        tokens = shlex.split(override)
        if not tokens:
            raise ValueError("run-command override is empty")
        parsed: dict[str, str | None] = {}
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if not token.startswith("-") or token == "-":
                raise ValueError(f"unnamed positional token {token!r} is not allowed")
            if "=" in token:
                option, value = token.split("=", 1)
            else:
                option = token
                value = None
                if index + 1 < len(tokens) and not tokens[index + 1].startswith("-"):
                    value = tokens[index + 1]
                    index += 1
            if option in parsed:
                raise ValueError(f"option {option!r} appears more than once")
            parsed[option] = value
            index += 1
        return parsed

    def _override_hash(self, override: str) -> str:
        """Hash the option/value semantics, independent of CLI spelling/order."""
        if self.mode == "free":
            return hashlib.sha256(override.encode()).hexdigest()
        canonical: dict[str, object] = self._parse_override_options(override)
        if canonical.get("--hparams") is not None:
            canonical["--hparams"] = json.loads(str(canonical["--hparams"]))
        encoded = json.dumps(
            canonical, sort_keys=True, separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def classify(
        self,
        paths: Iterable[str],
        *,
        override: str | None = None,
        variant_selection: Mapping[str, object] | None = None,
    ) -> CandidateVerdict:
        """Return a complete verdict without mutating filesystem or graph state."""
        raw_files = tuple(str(path) for path in paths)
        normalized = tuple(sorted({Path(path).as_posix() if path else "" for path in raw_files}))
        override_hash = ""
        if len(normalized) != len(raw_files):
            return self._verdict(
                CandidateClass.INVALID,
                False,
                normalized,
                "candidate file list contains duplicates",
            )

        # Protected rules intentionally take precedence over generic path errors,
        # preserving the named rule when an operator submits a matching absolute
        # path such as /etc/**.
        for path in normalized:
            matched = tuple(
                pattern for pattern in self.protected
                if matches_scope(path, (pattern,))
            )
            if matched:
                return self._verdict(
                    CandidateClass.PROTECTED_SURFACE_VIOLATION,
                    False,
                    normalized,
                    f"file {path!r} matches registry.protected pattern(s) {list(matched)}",
                    matched,
                )

        for path in normalized:
            if not path or os.path.isabs(path) or ".." in Path(path).parts:
                return self._verdict(
                    CandidateClass.INVALID,
                    False,
                    normalized,
                    f"Invalid path {path!r}: paths must be non-empty, relative, and contain no '..'",
                )

        override_rejection = self._classify_override(override, normalized)
        if override_rejection is not None:
            return override_rejection
        if override is not None:
            override_hash = self._override_hash(override)
        variant_rejection, variant_kinds, variant_selection_hash = (
            self._classify_variant_selection(variant_selection, normalized)
        )
        if variant_rejection is not None:
            return variant_rejection

        if self.mode == "architecture-preserving":
            outside = tuple(
                path for path in normalized
                if not matches_scope(path, self.editable)
            )
            if outside:
                return self._verdict(
                    CandidateClass.PROTECTED_SURFACE_VIOLATION,
                    False,
                    normalized,
                    f"file(s) {list(outside)} are outside the hard files.editable "
                    f"allowlist {list(self.editable)}",
                )

        if normalized:
            candidate_class = (
                CandidateClass.TRAIN_ONLY_SOURCE
                if self.mode == "architecture-preserving"
                else CandidateClass.FREE_SOURCE
            )
            return self._verdict(
                candidate_class,
                True,
                normalized,
                variant_kinds=variant_kinds,
                variant_selection_hash=variant_selection_hash,
                override_hash=override_hash,
            )
        if variant_kinds:
            return self._verdict(
                (
                    CandidateClass.TRAIN_ONLY_SOURCE
                    if self.mode == "architecture-preserving"
                    else CandidateClass.FREE_SOURCE
                ),
                True,
                normalized,
                variant_kinds=variant_kinds,
                variant_selection_hash=variant_selection_hash,
                override_hash=override_hash,
            )
        if override is not None:
            return self._verdict(
                CandidateClass.CONFIG_ONLY,
                True,
                normalized,
                variant_kinds=variant_kinds,
                variant_selection_hash=variant_selection_hash,
                override_hash=override_hash,
            )
        return self._verdict(
            CandidateClass.INVALID,
            False,
            normalized,
            "candidate contains neither source edits nor a configuration override",
        )

    def revalidate(
        self,
        paths: Iterable[str],
        *,
        override: str | None = None,
        variant_selection: Mapping[str, object] | None = None,
        recorded: Mapping[str, object] | None,
    ) -> CandidateVerdict:
        """Recompute a verdict and prove it matches the submit-time record."""
        if recorded is None:
            live = self.classify(
                paths, override=override, variant_selection=variant_selection,
            )
            if not live.accepted:
                raise AdmissibilityError(live.reason)
            if self.mode == "architecture-preserving":
                raise AdmissibilityError(
                    "missing submit-time admissibility verdict in architecture-preserving mode"
                )
            return live

        prior = CandidateVerdict.from_dict(recorded)
        if prior.policy_hash != self.policy_hash:
            raise AdmissibilityError(
                "candidate policy changed between submit and launch "
                f"({prior.policy_hash[:12]} != {self.policy_hash[:12]})"
            )
        live = self.classify(
            paths, override=override, variant_selection=variant_selection,
        )
        if not live.accepted:
            raise AdmissibilityError(live.reason)
        if prior != live:
            raise AdmissibilityError(
                "recorded admissibility verdict does not match the live candidate"
            )
        return live


def enforce_attempt_timeout_cap(
    spec_timeout_min: float | None,
    default_timeout_min: float | None,
) -> None:
    """Refuse a per-spec timeout ABOVE the cell's declared default.

    The campaign's attempt timeout is a hash-audited failure-containment
    constant, not a search budget — a per-spec ``--timeout`` above the cell
    default would silently unbind it (the runtime-canary agent raised 360→600
    unchallenged). Lowering stays free: a cheap probe releasing its slot
    early is exactly what containment wants.

    A standalone, grep-able check with two explicit enforcing callers —
    ``automil submit`` (agent-facing refusal) and the daemon's launch-time
    revalidation (queue specs are agent-editable JSON, so submit-only
    enforcement is bypassable). BOTH callers pass the RAW
    ``orchestrator.default_timeout_min`` config value as the reference: when
    the key is absent the check is skipped symmetrically at both gates, never
    accepted at one and refused (post-billing) at the other against a
    framework fallback the config never declared.
    """
    if spec_timeout_min is None or default_timeout_min is None:
        return
    if float(spec_timeout_min) > float(default_timeout_min):
        raise AdmissibilityError(
            f"campaign attempt timeout is failure containment, not a search "
            f"budget: --timeout {float(spec_timeout_min):g}min exceeds the "
            f"cell's audited default of {float(default_timeout_min):g}min "
            f"(lowering is allowed; raising is not)"
        )


def validate_campaign_binding(
    manifest_path: Path,
    campaign: Mapping[str, object],
    *,
    base_run_command: str | None,
    budget_cell_id: str,
) -> dict[str, object]:
    """Prove command, budget, and cell metadata share one manifest record.

    The framework does not interpret the consumer's dataset/model axes.  It
    only enforces a generic campaign envelope: one unique cell, a canonical
    per-cell hash, one budget identity, and a command selected by stage.
    """
    required = (
        "campaign_id", "manifest", "manifest_sha256", "cell_id",
        "cell_sha256", "budget_cell_id", "stage",
    )
    missing = [
        key for key in required
        if not isinstance(campaign.get(key), str) or not campaign.get(key)
    ]
    if missing:
        raise AdmissibilityError(
            f"campaign binding is missing non-empty string field(s) {missing}"
        )
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AdmissibilityError(f"cannot parse campaign manifest: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("cells"), list):
        raise AdmissibilityError("campaign manifest must contain a cells list")
    if manifest.get("campaign_id") != campaign["campaign_id"]:
        raise AdmissibilityError("campaign_id differs between config and manifest")

    matches = [
        cell for cell in manifest["cells"]
        if isinstance(cell, dict) and cell.get("cell_id") == campaign["cell_id"]
    ]
    if len(matches) != 1:
        raise AdmissibilityError(
            "campaign cell_id must identify exactly one manifest cell "
            f"(found {len(matches)})"
        )
    cell = dict(matches[0])
    identity = cell.get("identity")
    declared_protocol = campaign.get("protocol_version")
    manifest_schema = manifest.get("schema_version")
    if (
        not isinstance(manifest_schema, int)
        or isinstance(manifest_schema, bool)
        or manifest_schema < 1
    ):
        raise AdmissibilityError(
            "campaign manifest schema_version must be a positive integer"
        )
    is_v4_or_later = manifest_schema >= 4
    identity_keys = {
        "dataset", "task", "encoder", "arm", "seed", "protocol_version",
    }
    if not is_v4_or_later and identity is None and declared_protocol is None:
        # Legacy/third-party campaigns predate the optional protocol binding.
        # Preserve their generic envelope without weakening v4 manifests: once
        # either side declares identity metadata, both sides must agree.
        pass
    elif (
        not isinstance(identity, dict)
        or (is_v4_or_later and set(identity) != identity_keys)
        or not isinstance(declared_protocol, str)
        or not declared_protocol
        or identity.get("protocol_version") != declared_protocol
    ):
        raise AdmissibilityError(
            "campaign protocol_version differs from the manifest cell identity"
        )
    recorded_cell_hash = cell.pop("cell_sha256", None)
    canonical = json.dumps(
        cell, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    actual_cell_hash = hashlib.sha256(canonical).hexdigest()
    if (
        recorded_cell_hash != actual_cell_hash
        or campaign["cell_sha256"] != actual_cell_hash
    ):
        raise AdmissibilityError(
            "campaign cell hash differs from the manifest record"
        )

    budget = cell.get("budget_identity")
    manifest_budget = budget.get("cell_id") if isinstance(budget, dict) else None
    if manifest_budget != campaign["budget_cell_id"] or manifest_budget != budget_cell_id:
        raise AdmissibilityError(
            "campaign budget identity differs from the manifest record"
        )

    commands = cell.get("commands")
    stage = campaign["stage"]
    manifest_command = commands.get(stage) if isinstance(commands, dict) else None
    if not isinstance(manifest_command, str) or manifest_command != base_run_command:
        raise AdmissibilityError(
            "base run command differs from the manifest stage command"
        )
    return dict(campaign)


def _string_tuple(raw: object, key: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise TypeError(
            f"automil/config.yaml: {key!r} must be a list of strings; "
            f"got {type(raw).__name__}"
        )
    if not all(isinstance(item, str) for item in raw):
        raise TypeError(f"automil/config.yaml: {key!r} must contain only strings")
    return tuple(raw)


def load_candidate_policy(automil_dir: Path) -> CandidatePolicy:
    """Load the complete candidate policy from one consumer config."""
    from automil.registry.config import load_registry_config

    config_path = automil_dir / "config.yaml"
    raw = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    raw = raw or {}
    if not isinstance(raw, dict):
        raise TypeError("automil/config.yaml must contain a YAML mapping")
    files = raw.get("files") or {}
    if not isinstance(files, dict):
        raise TypeError(
            "automil/config.yaml: 'files' must be a mapping with an editable list"
        )
    registry = load_registry_config(automil_dir)
    return CandidatePolicy(
        mode=registry.mode,
        editable=_string_tuple(files.get("editable"), "files.editable"),
        protected=registry.protected,
        allowed_override_options=registry.allowed_override_options,
        allowed_variant_kinds=registry.allowed_variant_kinds,
        identity_locked_hparams=registry.identity_locked_hparams,
    )


_ARCHIVE_METADATA = frozenset({"spec.json", "run.log", "result.json"})
_FRAMEWORK_OVERLAY = frozenset({"applied_variant.json"})


def revalidate_candidate_spec(
    policy: CandidatePolicy,
    spec: Mapping[str, object],
    overlay_dir: Path,
) -> CandidateVerdict:
    """Revalidate one persisted spec and its exact archived overlay.

    Architecture-preserving mode additionally requires the archive's copyable
    file set to equal the digest manifest. This closes the legacy runner
    behaviour that copies unmanifested files after verifying only claimed ones.
    """
    manifest_raw = spec.get("overlay_manifest") or {}
    if not isinstance(manifest_raw, Mapping):
        raise AdmissibilityError("overlay_manifest must be a mapping")
    manifest_files = tuple(str(path) for path in manifest_raw)

    framework_raw = spec.get("framework_overlay_files") or []
    if not isinstance(framework_raw, list) or not all(
        isinstance(path, str) for path in framework_raw
    ):
        raise AdmissibilityError("framework_overlay_files must be a list of strings")
    framework_overlay = set(framework_raw) | set(_FRAMEWORK_OVERLAY)
    if not set(framework_raw).issubset(set(manifest_files)):
        raise AdmissibilityError(
            "framework_overlay_files contains a path absent from overlay_manifest"
        )

    deletions_raw = spec.get("deletions") or []
    if not isinstance(deletions_raw, list):
        raise AdmissibilityError("deletions must be a list")
    candidate_paths = tuple(
        sorted(
            (set(manifest_files) - framework_overlay)
            | {str(path) for path in deletions_raw}
        )
    )

    selection: Mapping[str, object] | None = None
    applied_candidates = [
        overlay_dir / path
        for path in sorted(framework_overlay)
        if (overlay_dir / path).exists()
    ]
    if len(applied_candidates) > 1:
        raise AdmissibilityError("multiple archived applied_variant.json files")
    if applied_candidates:
        applied_variant = applied_candidates[0]
        try:
            raw_selection = json.loads(applied_variant.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise AdmissibilityError(
                f"cannot read archived applied_variant.json: {exc}"
            ) from exc
        if not isinstance(raw_selection, Mapping):
            raise AdmissibilityError(
                "archived applied_variant.json must contain a JSON object"
            )
        selection = raw_selection

    if policy.mode == "architecture-preserving":
        actual: set[str] = set()
        if overlay_dir.exists():
            for path in overlay_dir.rglob("*"):
                if not path.is_file():
                    continue
                rel = path.relative_to(overlay_dir).as_posix()
                if rel in _ARCHIVE_METADATA:
                    continue
                if rel.startswith("certify/"):
                    continue
                actual.add(rel)
        claimed = set(manifest_files)
        if actual != claimed:
            raise AdmissibilityError(
                "archived overlay does not equal overlay_manifest "
                f"(unclaimed={sorted(actual - claimed)}, "
                f"missing={sorted(claimed - actual)})"
            )

    recorded = spec.get("admissibility")
    if recorded is not None and not isinstance(recorded, Mapping):
        raise AdmissibilityError("admissibility record must be a mapping")
    if (
        recorded is None
        and policy.mode == "free"
        and not candidate_paths
        and spec.get("run_command_override") is None
        and selection is None
    ):
        # Pre-admissibility queue specs did not persist a candidate record and
        # some carried no overlay (e.g. baseline/cap-accounting launches). Keep
        # that compatibility only in free mode; preserving mode remains
        # fail-closed and always requires a submit-time verdict.
        return CandidateVerdict(
            candidate_class=CandidateClass.FREE_SOURCE,
            accepted=True,
            files=(),
            reason="legacy free-mode spec without a persisted candidate",
            matched_patterns=(),
            variant_kinds=(),
            variant_selection_hash="",
            override_hash="",
            policy_hash=policy.policy_hash,
        )
    return policy.revalidate(
        candidate_paths,
        override=(
            str(spec["run_command_override"])
            if spec.get("run_command_override") is not None
            else None
        ),
        variant_selection=selection,
        recorded=recorded,
    )
