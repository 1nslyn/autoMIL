#!/usr/bin/env python3
"""Mirror registered baseline archives into durable project storage.

One-way scratch -> project mirror for the preprint campaign. The runtime
cell roots live in purge-eligible scratch; this exporter copies each
registered cell's baseline evidence to the export root (project storage)
in the team-facing hierarchy:

    <export_root>/<cohort>/<framework>/<task>/<encoder>/
        campaign_state.json     ledger snapshot, refreshed each export
        EXPORT_OK               completion marker, written last after
                                destination hashes verify against the ledger
        baseline/               baseline-execution/archive/ minus certify/

Sealed held-out evidence is quarantined under a separate owner-only tree so
the team-browsable mirror never carries test metrics during the search:

    <export_root>/sealed/<cohort>/<framework>/<task>/<encoder>/certify/

This file is deliberately standalone (stdlib only): it is delivered to the
cluster mid-campaign, where the checkout is frozen because running workers
import framework modules per cell. It must never import autobench/automil.

Usage:
    campaign_export.py --cell <cell_id>       # one cell (launcher post-run)
    campaign_export.py --all-registered       # catch-up / one-time mirror

The export root comes from AUTOBENCH_EXPORT_ROOT (benchmarks/.env) unless
--export-root overrides it. A missing EXPORT_OK marks a partial mirror; the
next invocation repairs it. Re-exports only when the ledger's result hash
differs from the marker, so repeated catch-ups are cheap.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

CAMPAIGN_REL = Path("benchmarks/campaigns/preprint_130")
CELL_ID_FIELDS = 6
# Frozen campaign identity artifacts seeded into <export_root>/campaign/.
CAMPAIGN_FILES = (
    "manifest.json",
    "manifest.json.sha256",
    "active_roster.json",
    "guard_margins.json",
    "analysis_plan.json",
)
OPTIONAL_CAMPAIGN_FILES = ("reproduction_policy.json",)

CAMPAIGN_README = """\
# autoMIL preprint campaign mirror

One-way mirror of registered campaign results from the execution runtime
(cluster scratch) into project storage. Layout:

    campaign/                                   frozen identity artifacts
    <cohort>/<framework>/<task>/<encoder>/      one leaf per campaign cell
        campaign_state.json                     cell ledger snapshot
        EXPORT_OK                               present = mirror verified
        baseline/                               validation-only baseline archive
        search/                                 agentic search results (added
                                                when discovery starts)
    sealed/...                                  held-out evidence, owner-only
                                                (0700); opened exclusively by
                                                `automil certify` at the end

Nothing in the framework reads this tree; do not edit it in place. A leaf
without EXPORT_OK is a partial copy — rerun the exporter to repair it.
"""


class ExportError(RuntimeError):
    """A cell failed to mirror; the message carries the cell context."""


def _fail(message: str) -> "SystemExit":
    return SystemExit(f"campaign_export: {message}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, what: str) -> dict:
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ExportError(f"cannot read {what} at {path}: {exc!r}") from exc
    if not isinstance(payload, dict):
        raise ExportError(f"{what} at {path} is not a JSON object")
    return payload


def _leaf_parts(cell_id: str) -> tuple[str, str, str, str]:
    """cell_id = dataset__task__encoder__framework__s<seed>__<protocol>."""
    parts = cell_id.split("__")
    if len(parts) != CELL_ID_FIELDS or not all(parts):
        raise ExportError(f"malformed cell_id {cell_id!r}")
    dataset, task, encoder, framework = parts[0], parts[1], parts[2], parts[3]
    return dataset, framework, task, encoder


def leaf_dir(export_root: Path, cell_id: str) -> Path:
    return export_root.joinpath(*_leaf_parts(cell_id))


def sealed_dir(export_root: Path, cell_id: str) -> Path:
    return export_root.joinpath("sealed", *_leaf_parts(cell_id))


def _manifest_cells(repo_root: Path) -> list[dict]:
    manifest = _load_json(repo_root / CAMPAIGN_REL / "manifest.json", "manifest")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ExportError("manifest has no cells")
    return cells


def roster_cell_ids(repo_root: Path) -> list[str]:
    """Roster-validated cell ids, ported from the launcher's scan contract."""
    roster_path = repo_root / CAMPAIGN_REL / "active_roster.json"
    roster = _load_json(roster_path, "active roster")
    try:
        cohorts = list(roster["cohorts"])
        expected = int(roster["cells"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ExportError(f"invalid active roster {roster_path}: {exc!r}") from exc
    if not cohorts:
        raise ExportError(f"{roster_path}: empty cohort list")
    records = _manifest_cells(repo_root)
    unknown = sorted(set(cohorts) - {cell["dataset"] for cell in records})
    if unknown:
        raise ExportError(f"roster cohorts absent from the manifest: {unknown}")
    cells = sorted(
        cell["cell_id"] for cell in records if cell["dataset"] in cohorts
    )
    if len(cells) != expected:
        raise ExportError(
            f"roster census mismatch: {len(cells)} manifest cells for "
            f"{sorted(cohorts)}, declared cells={expected}"
        )
    return cells


def _registered_baseline(state: dict, cell_id: str) -> dict:
    if state.get("cell_id") != cell_id:
        raise ExportError(
            f"{cell_id}: state carries cell_id {state.get('cell_id')!r}"
        )
    baseline = state.get("baseline")
    if baseline is None:
        raise ExportError(f"{cell_id}: no registered baseline")
    if not isinstance(baseline, dict):
        raise ExportError(
            f"{cell_id}: registered baseline state is invalid "
            f"({type(baseline).__name__})"
        )
    return baseline


def _rsync(source: Path, dest: Path, *extra: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["rsync", "-a", *extra, f"{source}/", f"{dest}/"],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ExportError(
            f"rsync {source} -> {dest} failed rc={completed.returncode}: "
            f"{completed.stderr.strip()[:500]}"
        )


def _atomic_write(path: Path, payload: dict) -> None:
    handle = tempfile.NamedTemporaryFile(
        "w", dir=str(path.parent), prefix=f".{path.name}.", delete=False
    )
    try:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.close()
        os.replace(handle.name, path)
    except OSError:
        handle.close()
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise


def _atomic_copy(source: Path, dest: Path) -> None:
    handle = tempfile.NamedTemporaryFile(
        "wb", dir=str(dest.parent), prefix=f".{dest.name}.", delete=False
    )
    try:
        handle.write(source.read_bytes())
        handle.close()
        os.replace(handle.name, dest)
    except OSError:
        handle.close()
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise


def _marker_current(leaf: Path, baseline: dict) -> bool:
    marker_path = leaf / "EXPORT_OK"
    if not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text())
    except (OSError, ValueError):
        return False
    return (
        isinstance(marker, dict)
        and marker.get("result_sha256") == baseline.get("result_sha256")
        and marker.get("attestation_sha256") == baseline.get("attestation_sha256")
    )


def _verify_destination(leaf: Path, sealed_leaf: Path, baseline: dict) -> None:
    """Destination bytes must match the ledger before the marker is written."""
    result_path = leaf / "baseline" / "result.json"
    if not result_path.is_file():
        raise ExportError(f"mirrored result missing: {result_path}")
    observed = _sha256_file(result_path)
    if observed != baseline.get("result_sha256"):
        raise ExportError(
            f"mirrored result hash mismatch: {observed} != "
            f"{baseline.get('result_sha256')}"
        )
    attestation = _load_json(
        leaf / "baseline" / "baseline_attestation.json", "mirrored attestation"
    )
    if attestation.get("attestation_sha256") != baseline.get("attestation_sha256"):
        raise ExportError("mirrored attestation does not match the ledger")
    sealed_hashes = baseline.get("sealed_fold_sha256")
    if not isinstance(sealed_hashes, dict) or not sealed_hashes:
        raise ExportError("ledger has no sealed fold hashes")
    for name, expected in sorted(sealed_hashes.items()):
        fold_path = sealed_leaf / "certify" / name
        if not fold_path.is_file():
            raise ExportError(f"sealed fold missing from mirror: {fold_path}")
        observed = _sha256_file(fold_path)
        if observed != expected:
            raise ExportError(
                f"sealed fold hash mismatch for {name}: {observed} != {expected}"
            )


def _ensure_sealed_root(export_root: Path) -> None:
    sealed_root = export_root / "sealed"
    sealed_root.mkdir(parents=True, exist_ok=True)
    # Owner-only at the tree root is the single enforcement point: the group
    # cannot traverse below it regardless of copied file modes.
    os.chmod(sealed_root, 0o700)


# Job-created archives carry the compute node's umask (often 0600); the
# public mirror must be group-readable for the team. Modes are normalized in
# Python (rsync --chmod dialects differ between GNU rsync and openrsync).
# The sealed tree keeps source modes — its 0700 root is the enforcement point.
PUBLIC_DIR_MODE = 0o2750
PUBLIC_FILE_MODE = 0o640


def _apply_public_modes(export_root: Path, leaf: Path, files: list[Path]) -> None:
    relative = leaf.relative_to(export_root)
    current = export_root
    for part in relative.parts:
        current = current / part
        os.chmod(current, PUBLIC_DIR_MODE)
    for path in files:
        os.chmod(path, PUBLIC_FILE_MODE)


def _normalize_public_tree(root: Path) -> None:
    os.chmod(root, PUBLIC_DIR_MODE)
    for directory, _subdirs, filenames in os.walk(root):
        os.chmod(directory, PUBLIC_DIR_MODE)
        for name in filenames:
            os.chmod(Path(directory) / name, PUBLIC_FILE_MODE)


def export_cell(runtime: Path, export_root: Path, cell_id: str) -> str:
    """Mirror one registered cell; returns 'exported' or 'current'."""
    cell_root = runtime / cell_id
    state_path = cell_root / "campaign_state.json"
    if not state_path.is_file():
        raise ExportError(f"{cell_id}: no materialized state under {runtime}")
    state = _load_json(state_path, f"{cell_id} state")
    baseline = _registered_baseline(state, cell_id)
    archive = cell_root / "baseline-execution" / "archive"
    if not (archive / "result.json").is_file():
        raise ExportError(f"{cell_id}: execution archive missing at {archive}")

    leaf = leaf_dir(export_root, cell_id)
    sealed_leaf = sealed_dir(export_root, cell_id)
    leaf.mkdir(parents=True, exist_ok=True)
    with (leaf / ".export.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if _marker_current(leaf, baseline):
            return "current"
        _ensure_sealed_root(export_root)
        _rsync(archive, leaf / "baseline", "--exclude=/certify")
        if (archive / "certify").is_dir():
            _rsync(archive / "certify", sealed_leaf / "certify")
        _atomic_copy(state_path, leaf / "campaign_state.json")
        _verify_destination(leaf, sealed_leaf, baseline)
        _atomic_write(
            leaf / "EXPORT_OK",
            {
                "cell_id": cell_id,
                "result_sha256": baseline["result_sha256"],
                "attestation_sha256": baseline["attestation_sha256"],
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "source": str(cell_root),
            },
        )
        _apply_public_modes(
            export_root,
            leaf,
            [leaf / "campaign_state.json", leaf / "EXPORT_OK"],
        )
        _normalize_public_tree(leaf / "baseline")
    return "exported"


def seed_campaign_dir(repo_root: Path, export_root: Path) -> None:
    target = export_root / "campaign"
    target.mkdir(parents=True, exist_ok=True)
    names = list(CAMPAIGN_FILES) + [
        name
        for name in OPTIONAL_CAMPAIGN_FILES
        if (repo_root / CAMPAIGN_REL / name).is_file()
    ]
    for name in names:
        source = repo_root / CAMPAIGN_REL / name
        if not source.is_file():
            raise ExportError(f"campaign artifact missing: {source}")
        shutil.copy2(source, target / name)
    (target / "README.md").write_text(CAMPAIGN_README)
    _apply_public_modes(
        export_root, target, [target / name for name in names + ["README.md"]]
    )


def _resolve_export_root(value: str | None) -> Path:
    root = value or os.environ.get("AUTOBENCH_EXPORT_ROOT", "")
    if not root:
        raise _fail(
            "export root is not set: pass --export-root or set "
            "AUTOBENCH_EXPORT_ROOT in benchmarks/.env"
        )
    path = Path(root)
    if not path.is_dir():
        raise _fail(f"export root does not exist: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cell", help="export one registered cell by cell_id")
    mode.add_argument(
        "--all-registered",
        action="store_true",
        help="export every registered roster cell and seed campaign/",
    )
    parser.add_argument("--export-root", default=None)
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[2]),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    runtime = repo_root / CAMPAIGN_REL / "runtime"
    if not runtime.is_dir():
        raise _fail(f"runtime not materialized: {runtime}")
    export_root = _resolve_export_root(args.export_root)

    if args.cell:
        known = {cell["cell_id"] for cell in _manifest_cells(repo_root)}
        if args.cell not in known:
            raise _fail(f"{args.cell!r} is not a manifest cell")
        try:
            outcome = export_cell(runtime, export_root, args.cell)
        except ExportError as exc:
            print(f"campaign_export: {exc}", file=sys.stderr)
            return 1
        print(f"{args.cell}: {outcome}")
        return 0

    failures: list[str] = []
    exported = current = pending = 0
    for cell_id in roster_cell_ids(repo_root):
        state_path = runtime / cell_id / "campaign_state.json"
        if not state_path.is_file():
            failures.append(f"{cell_id}: no materialized state")
            continue
        try:
            state = _load_json(state_path, f"{cell_id} state")
            if state.get("baseline") is None:
                pending += 1
                continue
            outcome = export_cell(runtime, export_root, cell_id)
        except ExportError as exc:
            failures.append(str(exc))
            continue
        exported += outcome == "exported"
        current += outcome == "current"
    try:
        seed_campaign_dir(repo_root, export_root)
    except ExportError as exc:
        failures.append(f"campaign dir: {exc}")
    print(
        f"exported={exported} current={current} pending={pending} "
        f"failed={len(failures)}"
    )
    for failure in failures:
        print(f"  FAILED {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
