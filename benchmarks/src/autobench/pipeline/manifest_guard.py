"""Manifest-content fingerprint for the task-CSV / splits cache (M-10).

``prepare_all`` (``prepare.py``) validates a cached task
CSV's SCHEMA (columns match the task's type -- classification vs survival)
and validates cached splits against that task CSV's actual slide_id set.
Neither check can tell that the CSV's SOURCE -- the manifest (``mapping_csv``,
the file ``load_all_slides`` reads) -- was rebuilt with different values
since the CSV was generated: same columns, same slide_id set even, but a
label/status/time value silently corrected upstream (e.g. an OS date fix)
never reaches the cached derived artefacts. The cache looks perfectly valid
by schema and identity checks alone, and is quietly wrong.

**Stamp a sidecar, don't self-heal.** ``prepare_all`` runs once per
EXPERIMENT against the SHARED ``benchmark_dir`` (see
``scripts/run_experiment.py``), so under the agentic loop many processes
execute it concurrently. Self-purging can race across concurrent ``rmtree``
calls and can delete
splits another process was already training from -- the same race
``prepare_all``'s own task-CSV/splits guards (PRELAUNCH_REVIEW B2) and
``results_cache.py`` (CR-5b) both document and refuse to self-heal. This
follows the identical pattern: fail loudly with the exact purge command,
and let the operator run it once, deliberately, outside the concurrent hot
path.

**Content hash, measured, not guessed.** SHA256 over the manifest file
takes ~22ms at 100,000 rows / 46.6MB (measured on this machine) -- and every
roster cohort's manifest (LUAD/LGG/GBM/PDAC/HNSC, single-cohort, slide-
indexed) is hundreds to low thousands of rows, i.e. sub-millisecond. That is
negligible next to the rest of ``prepare_all`` (split generation, H5->PT
conversion), and a content hash catches a same-size, same-mtime rewrite
(e.g. certain sync/checkout tools do not preserve mtime) that mtime+size
would miss. mtime+size remains available for a manifest large enough that
hashing becomes material, but is not the default.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile

__all__ = [
    "FINGERPRINT_FILENAME",
    "StaleManifestError",
    "manifest_fingerprint",
    "check_manifest_fingerprint",
]

#: Sidecar written next to the derived artefacts it describes
#: (``<benchmark_dir>/dataset_csv/``).
FINGERPRINT_FILENAME = "manifest_fingerprint.json"


class StaleManifestError(RuntimeError):
    """A cached task-CSV/splits tree was derived from a DIFFERENT manifest.

    Raised instead of resuming, because resuming would silently keep
    serving derived artefacts built from stale manifest values.
    """


def manifest_fingerprint(mapping_csv: str, *, use_content_hash: bool = True) -> dict:
    """Stable fingerprint of a manifest file's identity.

    Args:
        mapping_csv: path to the manifest CSV (``load_all_slides`` reads it).
        use_content_hash: SHA256 over the file's bytes (default -- see
            module docstring for the measurement backing this default).
            Pass ``False`` to fall back to mtime+size for a manifest large
            enough that hashing becomes material; measure before switching,
            don't assume.
    """
    st = os.stat(mapping_csv)
    if use_content_hash:
        h = hashlib.sha256()
        with open(mapping_csv, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()
    else:
        digest = f"mtime_size:{st.st_mtime_ns}:{st.st_size}"
    return {
        "mapping_csv": os.path.abspath(mapping_csv),
        "digest": digest,
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


def _write_atomic(path: str, payload: dict) -> None:
    """Concurrent experiments share ``benchmark_dir``; never leave a torn file."""
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".manifest_fingerprint-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def check_manifest_fingerprint(
    dataset_csv_dir: str,
    mapping_csv: str,
    *,
    use_content_hash: bool = True,
) -> None:
    """Verify (or, on first encounter, stamp) the manifest fingerprint.

    Args:
        dataset_csv_dir: the directory holding the cached task CSVs
            (``<benchmark_dir>/dataset_csv``) -- the sidecar is written
            next to them, one per ``benchmark_dir`` (every task in a given
            ``prepare_all`` call shares the same ``mapping_csv``).
        mapping_csv: the manifest path this run was invoked with.

    Raises:
        StaleManifestError: a sidecar already exists and does not match
            ``mapping_csv``'s current content -- the cached CSVs/splits in
            this ``benchmark_dir`` were derived from a DIFFERENT manifest.

    No sidecar present is treated as first encounter and stamps the current
    manifest as ground truth -- deliberately, mirroring
    ``results_cache.py``'s identical precedent (CR-5b): this cannot
    retroactively catch a manifest change that happened BEFORE this guard
    shipped, only one from this point forward. That is the same trust
    boundary CR-5b accepted for the results cache; re-litigating it here
    would mean either self-healing (the race this module exists to avoid)
    or blocking on manual verification of every pre-existing benchmark_dir.
    """
    os.makedirs(dataset_csv_dir, exist_ok=True)
    path = os.path.join(dataset_csv_dir, FINGERPRINT_FILENAME)
    current = manifest_fingerprint(mapping_csv, use_content_hash=use_content_hash)

    if os.path.exists(path):
        try:
            with open(path) as f:
                stored = json.load(f)
        except (OSError, json.JSONDecodeError):
            stored = None  # unreadable sidecar: re-stamp rather than block
        if stored is not None:
            if stored.get("digest") == current["digest"]:
                return
            raise StaleManifestError(
                f"{dataset_csv_dir} holds task CSVs/splits derived from a "
                f"DIFFERENT manifest than the one this run was given.\n"
                f"  manifest now:  {current['mapping_csv']} "
                f"(digest {current['digest'][:12]}..., "
                f"{current['size']} bytes)\n"
                f"  manifest then: {stored.get('mapping_csv', '?')} "
                f"(digest {str(stored.get('digest', '?'))[:12]}..., "
                f"{stored.get('size', '?')} bytes)\n"
                "The manifest was rebuilt (e.g. a label/status/time value "
                "corrected upstream) since these derived artefacts were "
                "generated -- reusing them would silently keep serving the "
                "stale values. This is NOT purged automatically: prepare_all "
                "runs concurrently against this SHARED benchmark_dir, and an "
                "earlier self-purging version raced concurrent readers into "
                "FileNotFoundError and deleted splits another process was "
                "training from. Purge and re-run prep explicitly, once, "
                "before launching concurrent work:\n"
                f"  rm -rf {dataset_csv_dir}\n"
                f"  rm -rf {os.path.join(os.path.dirname(dataset_csv_dir), 'splits')}"
            )

    _write_atomic(path, current)
