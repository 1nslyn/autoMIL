"""Phase 2 review regression tests.

Covers two findings from the Phase 2 code review:

- **CR-01**: ``LocalBackend.submit`` previously hardcoded
  ``"overlay_dir": f"archive/{spec.node_id}"`` in the daemon queue spec,
  ignoring ``spec.overlay_dir``. ``automil resubmit`` builds a new
  ``JobSpec(node_id=NEW_ID, overlay_dir=archive/<OLD_ID>)`` — the bug
  caused the daemon to look for the overlay under ``archive/<NEW_ID>``
  (which only contains spec.json) and apply nothing, silently running
  the resubmitted experiment on base-commit code instead of the variant.
  The fix uses ``spec.overlay_dir`` directly (relative to ``orch_dir``
  when possible).

- **WR-01**: ``automil/orchestrator.py``'s ``__getattr__`` shim fired a
  ``DeprecationWarning`` for every Python-internal dunder probe
  (``__path__``, ``__bases__``, ``__test__``, etc.) during pytest
  collection and import-machinery probing, flooding test output with
  spurious warnings. The fix short-circuits dunder names before warning.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from automil.backends.base import JobSpec
from automil.backends.local import LocalBackend


# ---------------------------------------------------------------------------
# CR-01: LocalBackend.submit honours spec.overlay_dir
# ---------------------------------------------------------------------------


def _make_local_backend(tmp_path: Path) -> LocalBackend:
    """Build a LocalBackend rooted at ``tmp_path`` with the on-disk
    layout the daemon expects."""
    adir = tmp_path / "automil"
    adir.mkdir()
    (adir / "orchestrator").mkdir()
    (adir / "orchestrator" / "queue").mkdir()
    (adir / "orchestrator" / "running").mkdir()
    (adir / "orchestrator" / "archive").mkdir()
    return LocalBackend(project_root=tmp_path, automil_dir=adir)


def test_local_backend_submit_uses_spec_overlay_dir(tmp_path):
    """CR-01: queue_spec['overlay_dir'] must reflect spec.overlay_dir,
    not a hardcoded f'archive/{spec.node_id}' string."""
    backend = _make_local_backend(tmp_path)
    orch_dir = tmp_path / "automil" / "orchestrator"
    archive_dir = orch_dir / "archive"

    # Pre-create an OLD node's archive (mimics what `automil submit` writes).
    old_archive = archive_dir / "node_OLD"
    old_archive.mkdir()

    # New node uses the OLD archive as its overlay source — this is the
    # exact pattern resubmit.py uses (resubmit.py:169).
    new_spec = JobSpec(
        node_id="node_NEW",
        base_commit="abc1234",
        overlay_files=("train.py",),
        overlay_dir=old_archive,
        command=("python", "train.py"),
        env=(),
        working_subdir="",
        gpu_estimate_gb=0.5,
        walltime_seconds=60,
    )
    backend.submit(new_spec)

    # Read the queue spec back and assert overlay_dir points at the OLD archive.
    queue_file = orch_dir / "queue" / "node_NEW.json"
    assert queue_file.exists(), "submit did not write queue spec"
    queue_spec = json.loads(queue_file.read_text())

    assert queue_spec["overlay_dir"] == "archive/node_OLD", (
        "CR-01 regression: queue_spec['overlay_dir'] should reflect "
        f"spec.overlay_dir (archive/node_OLD), got {queue_spec['overlay_dir']!r}"
    )
    # Sanity: the queue spec's id IS the new node id (so the bug was real —
    # daemon would otherwise resolve archive/{id} to archive/node_NEW).
    assert queue_spec["id"] == "node_NEW"


def test_local_backend_submit_default_archive_path(tmp_path):
    """The non-resubmit path: spec.overlay_dir = automil/orchestrator/archive/<new_id>
    should still produce queue_spec['overlay_dir'] = 'archive/<new_id>' (relative)."""
    backend = _make_local_backend(tmp_path)
    orch_dir = tmp_path / "automil" / "orchestrator"
    archive_dir = orch_dir / "archive"

    new_archive = archive_dir / "node_FRESH"
    new_archive.mkdir()

    spec = JobSpec(
        node_id="node_FRESH",
        base_commit="def5678",
        overlay_files=(),
        overlay_dir=new_archive,
        command=("echo", "hello"),
        env=(),
        working_subdir="",
        gpu_estimate_gb=0.0,
        walltime_seconds=60,
    )
    backend.submit(spec)

    queue_spec = json.loads((orch_dir / "queue" / "node_FRESH.json").read_text())
    assert queue_spec["overlay_dir"] == "archive/node_FRESH"


def test_local_backend_submit_overlay_outside_orch_dir(tmp_path):
    """If spec.overlay_dir is OUTSIDE orch_dir, fall back to absolute string
    (rather than crashing with ValueError on Path.relative_to)."""
    backend = _make_local_backend(tmp_path)
    orch_dir = tmp_path / "automil" / "orchestrator"

    # Overlay is in a sibling directory, NOT under orch_dir.
    external_overlay = tmp_path / "external_overlay"
    external_overlay.mkdir()

    spec = JobSpec(
        node_id="node_EXT",
        base_commit="abc1234",
        overlay_files=(),
        overlay_dir=external_overlay,
        command=("echo", "hi"),
        env=(),
        working_subdir="",
        gpu_estimate_gb=0.0,
        walltime_seconds=60,
    )
    backend.submit(spec)

    queue_spec = json.loads((orch_dir / "queue" / "node_EXT.json").read_text())
    # Absolute string fallback — must NOT raise ValueError on relative_to.
    assert queue_spec["overlay_dir"] == str(external_overlay)


# ---------------------------------------------------------------------------
# WR-01: orchestrator.py shim does not warn on dunder probes
# ---------------------------------------------------------------------------


def test_orchestrator_shim_silent_on_dunder_probes():
    """WR-01: Python-internal dunder probes (__path__, __test__, __wrapped__,
    etc.) on the orchestrator.py shim must NOT produce DeprecationWarning.

    The import machinery and pytest collection probe these names on every
    module access; warning on each would flood test output with 14+ spurious
    warnings per test run.
    """
    import automil.orchestrator as shim

    # The most common dunders the import machinery probes.
    dunder_names = ["__path__", "__test__", "__wrapped__", "__bases__"]

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")  # capture EVERY warning, ignore filters

        for name in dunder_names:
            try:
                getattr(shim, name)
            except AttributeError:
                pass  # Expected — dunder absent, AttributeError is fine.

        # No DeprecationWarning should have been emitted for the dunder probes.
        deprecation_warnings = [
            w for w in captured if issubclass(w.category, DeprecationWarning)
        ]
        assert deprecation_warnings == [], (
            f"WR-01 regression: orchestrator shim emitted "
            f"{len(deprecation_warnings)} DeprecationWarning(s) for dunder "
            f"probes: {[str(w.message) for w in deprecation_warnings]}"
        )


def test_orchestrator_shim_still_warns_on_renamed_names():
    """The dunder fix must NOT silence legitimate deprecation warnings.

    A non-dunder name that needs the shim's lazy lookup should still produce
    a DeprecationWarning. We use a name that is NOT in the explicit re-export
    list at the top of the shim, so __getattr__ is the only resolution path.
    """
    import automil.orchestrator as shim

    # Pick a name that is in _orchestrator_daemon but NOT in the shim's
    # explicit re-export list. `query_gpus` is one such (re-exported via star
    # import, which populates __dict__, so __getattr__ wouldn't fire). We
    # need a name that is ONLY accessible via __getattr__.
    # `logger` (the daemon's module-level logger) is a good candidate — it
    # IS exported by the star-import, so __getattr__ won't fire on it
    # either. Instead, probe a name that doesn't exist on the daemon module
    # at all; getattr must raise AttributeError after warning.
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(AttributeError):
            _ = shim.this_name_definitely_does_not_exist_xyz123

        deprecation_warnings = [
            w for w in captured if issubclass(w.category, DeprecationWarning)
        ]
        # The shim warns FIRST, then delegates to the daemon module's
        # getattr which raises AttributeError. So we should see exactly one
        # DeprecationWarning before the AttributeError propagates.
        assert len(deprecation_warnings) == 1, (
            f"Expected 1 DeprecationWarning for non-dunder name, got "
            f"{len(deprecation_warnings)}: "
            f"{[str(w.message) for w in deprecation_warnings]}"
        )
        assert "this_name_definitely_does_not_exist_xyz123" in str(
            deprecation_warnings[0].message
        )
