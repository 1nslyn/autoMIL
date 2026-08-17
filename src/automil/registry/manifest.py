"""Sibling JSON manifest for registered variants (REG-08 / D-44).

Each variant module `<name>.py` has a sibling `<name>.json` that records the
VariantSpec (provenance) plus port-time metadata.  At registry-refresh time
the manifest is cross-checked against the variant module's docstring header;
a mismatch causes `automil check` to report a consistency failure (D-44).
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from automil.registry.spec import VariantSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Manifest:
    """Sibling JSON manifest for a registered variant (D-44).

    File location: ``<variants_root>/<kind_dir>/<name>.json``
    Purpose: provenance + cross-check against the variant module's docstring.
    """
    spec: VariantSpec
    source_node: str                       # e.g., "node_0176"
    source_overlay_files: tuple[str, ...]  # relative paths the dirty diff touched
    ported_at: str                         # ISO-8601 UTC
    tool_version: str                      # e.g., "automil 0.1.0"

    # ------------------------------------------------------------------ #
    # Write                                                                #
    # ------------------------------------------------------------------ #

    def write(self, path: Path) -> None:
        """Atomic-write the manifest as JSON to ``path``. PATTERNS.md §3."""
        payload: dict = {
            "spec": {
                "name": self.spec.name,
                "kind": self.spec.kind,
                "parent": self.spec.parent,
                "base_commit": self.spec.base_commit,
                "primary_value": self.spec.primary_value,
                "node_id": self.spec.node_id,
                "created_at": self.spec.created_at,
                "mutations": list(self.spec.mutations),
            },
            "source_node": self.source_node,
            "source_overlay_files": list(self.source_overlay_files),
            "ported_at": self.ported_at,
            "tool_version": self.tool_version,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(payload, f, indent=2)
                f.write("\n")
            os.rename(tmp_path, str(path))
            os.utime(str(path))
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        logger.info("Wrote manifest %s (node=%s)", path.name, self.source_node)

    # ------------------------------------------------------------------ #
    # Read                                                                 #
    # ------------------------------------------------------------------ #

    @classmethod
    def read(cls, path: Path) -> "Manifest":
        """Load + validate a manifest from disk.

        Raises:
            FileNotFoundError: if ``path`` does not exist.
            ValueError: on JSON parse error or missing required keys.
        """
        if not path.exists():
            raise FileNotFoundError(f"manifest not found: {path}")
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"manifest at {path} is not valid JSON: {exc.msg} (line {exc.lineno})"
            ) from exc

        for key in ("spec", "source_node", "source_overlay_files", "ported_at", "tool_version"):
            if key not in payload:
                raise ValueError(
                    f"manifest at {path} missing required key: {key!r}"
                )

        spec_data = payload["spec"]
        for k in ("name", "kind", "parent", "base_commit", "primary_value", "node_id", "created_at"):
            if k not in spec_data:
                raise ValueError(
                    f"manifest at {path}: spec section missing required key {k!r}"
                )

        spec = VariantSpec(
            name=spec_data["name"],
            kind=spec_data["kind"],
            parent=spec_data["parent"],
            base_commit=spec_data["base_commit"],
            primary_value=float(spec_data["primary_value"]),
            node_id=spec_data["node_id"],
            created_at=spec_data["created_at"],
            mutations=tuple(spec_data.get("mutations") or []),
        )

        return cls(
            spec=spec,
            source_node=payload["source_node"],
            source_overlay_files=tuple(payload["source_overlay_files"]),
            ported_at=payload["ported_at"],
            tool_version=payload["tool_version"],
        )

    # ------------------------------------------------------------------ #
    # Cross-check                                                          #
    # ------------------------------------------------------------------ #

    def cross_check_with_module(self, module_path: Path) -> tuple[bool, str]:
        """Verify the manifest's spec matches the docstring header in ``module_path``.

        Returns ``(True, "")`` on success; ``(False, reason)`` on mismatch.
        Plan 01-08 (``automil check``) calls this for every registered variant.

        Cross-checked fields (D-44 "cross-checked against the variant module's
        docstring"):
          - first word of the docstring == spec.name
          - ``Parent:`` line (for kind=model) == spec.parent
          - ``Primary_value:`` line (exact float) == spec.primary_value
          - ``Node ID:`` line == spec.node_id
        """
        if not module_path.exists():
            return False, f"module not found: {module_path}"

        text = module_path.read_text()
        doc = self._extract_docstring(text)
        if doc is None:
            return False, "no module docstring found"

        stripped = doc.strip()
        first_line = stripped.splitlines()[0] if stripped else ""

        # The first token of the first line should be the variant name.
        # Handle forms like "clam_mb_v0176 variant." or "clam_mb_v0176: description"
        name_match = re.match(r"^(\S+)", first_line)
        if name_match:
            doc_name = name_match.group(1).rstrip(":,.")
            if doc_name != self.spec.name:
                return False, (
                    f"docstring name {doc_name!r} != manifest name {self.spec.name!r}"
                )

        # Parse structured field lines: "Key: value"
        fields = self._parse_doc_fields(doc)

        # Parent check (model variants only).
        if self.spec.kind == "model" and self.spec.parent is not None:
            doc_parent = fields.get("parent")
            if doc_parent is not None and doc_parent != self.spec.parent:
                return False, (
                    f"docstring parent {doc_parent!r} != manifest parent {self.spec.parent!r}"
                )

        # Primary_value check (exact float comparison — D-44 "no tolerance").
        # A pre-rename module carrying the legacy `Composite:` label must fail
        # CLOSED here, not skip the check: silently accepting it turns the
        # D-44 cross-check into a no-op on exactly the modules most likely to
        # have drifted.
        doc_primary_value_str = fields.get("primary_value")
        if doc_primary_value_str is None and fields.get("composite") is not None:
            return False, (
                "docstring carries the retired 'Composite:' label; rename it "
                "to 'Primary_value:' (the composite concept was retired — "
                "single-metric optimization)"
            )
        if doc_primary_value_str is not None:
            try:
                doc_c = float(doc_primary_value_str)
            except ValueError:
                return False, f"docstring primary_value is not a float: {doc_primary_value_str!r}"
            if doc_c != self.spec.primary_value:
                return False, (
                    f"docstring primary_value {doc_c} != manifest primary_value {self.spec.primary_value}"
                )

        # Node ID check.
        doc_node = fields.get("node id") or fields.get("node_id")
        if doc_node is not None and doc_node != self.spec.node_id:
            return False, (
                f"docstring node_id {doc_node!r} != manifest node_id {self.spec.node_id!r}"
            )

        return True, ""

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_docstring(text: str) -> str | None:
        """Return the module-level docstring using AST, or None if absent."""
        import ast
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return None
        return ast.get_docstring(tree)

    @staticmethod
    def _parse_doc_fields(doc: str) -> dict[str, str]:
        """Parse ``Key: value`` lines from the docstring into a lowercase-key dict."""
        fields: dict[str, str] = {}
        for line in doc.splitlines():
            if ":" not in line:
                continue
            k, _, v = line.partition(":")
            fields[k.strip().lower()] = v.strip()
        return fields
