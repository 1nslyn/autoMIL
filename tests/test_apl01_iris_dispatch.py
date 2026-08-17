"""APL-01: iris reference consumer dispatches classifier_v0 when selected.

Tests in this file are RED until Plan 10-02 (applied_variant.json mechanism)
and Plan 10-04 (iris train.py dispatch) are both implemented.

Exception: test_iris_baseline_no_variant is a regression guard that must
PASS from the start (it exercises the pre-existing baseline path).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
IRIS_TRAIN = REPO_ROOT / "examples" / "sklearn-iris" / "train.py"
IRIS_AUTOMIL = REPO_ROOT / "examples" / "sklearn-iris" / "automil"


def _run_iris(cwd: Path, extra_env: dict | None = None) -> dict:
    """Run iris train.py in *cwd* and return parsed result.json."""
    import os

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        [sys.executable, str(IRIS_TRAIN)],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"iris train.py exited {result.returncode}\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    result_path = cwd / "result.json"
    assert result_path.exists(), f"result.json not written to {cwd}"
    return json.loads(result_path.read_text())


# ---------------------------------------------------------------------------
# Test 1: Baseline — no variant → result.json written, primary_value > 0
# Regression guard: MUST PASS from the start (no production changes needed).
# ---------------------------------------------------------------------------


def test_iris_baseline_no_variant(tmp_path):
    """Iris train.py with no model.variant writes result.json with primary_value > 0."""
    # Write a minimal automil/config.yaml with NO model.variant key.
    automil_dir = tmp_path / "automil"
    automil_dir.mkdir()
    config = {"data": {"seed": 42}}
    (automil_dir / "config.yaml").write_text(yaml.safe_dump(config))

    result = _run_iris(tmp_path)

    assert result["status"] == "completed"
    assert result["primary_value"] > 0, f"primary_value should be > 0, got {result['primary_value']}"
    assert "accuracy" in result.get("metrics", {})


# ---------------------------------------------------------------------------
# Test 2: Dispatch — model.variant=classifier_v0 → make_classifier is called
# RED until Plan 10-04 adds dispatch to iris train.py.
# ---------------------------------------------------------------------------


def test_iris_dispatches_classifier_v0_when_variant_set(tmp_path):
    """When model.variant=classifier_v0, iris train.py must dispatch to make_classifier.

    Strategy: write the classifier_v0 variant directory into the tmp automil/
    directory, set model.variant in config.yaml, run iris train.py, and assert
    result.json primary_value > 0 AND that the variant was actually dispatched
    (sentinel file written by the variant module OR environment variable probe).

    This test verifies the dispatch branch exists in train.py — it will fail RED
    until Plan 10-04 adds that branch.
    """
    # Copy the classifier_v0 variant into the tmp automil/ directory.
    import shutil

    automil_dir = tmp_path / "automil"
    automil_dir.mkdir()

    src_variant = IRIS_AUTOMIL / "variants" / "classifier_v0"
    if not src_variant.exists():
        pytest.skip(f"classifier_v0 variant not found at {src_variant}")

    dst_variants = automil_dir / "variants" / "classifier_v0"
    shutil.copytree(str(src_variant), str(dst_variants))

    config = {
        "data": {"seed": 42},
        "model": {"variant": "classifier_v0"},
    }
    (automil_dir / "config.yaml").write_text(yaml.safe_dump(config))

    # Write a sentinel-probe wrapper alongside the real variant: we inject a
    # DISPATCH_SENTINEL env var probe.  The simplest approach is to patch via
    # a side-by-side shim script, but since this is a RED stub test we just
    # verify the dispatch branch exists by running and checking for a dispatch
    # indicator written into result.json (plan 10-04 will add "variant_dispatched"
    # to the result payload).
    result = _run_iris(tmp_path)

    assert result["status"] == "completed"
    # Plan 10-04 must add "variant_dispatched" key to result.json to satisfy this assertion.
    assert result.get("variant_dispatched") == "classifier_v0", (
        "APL-01: iris train.py did not dispatch classifier_v0. "
        "Implement the model.variant dispatch branch in train.py (Plan 10-04)."
    )


# ---------------------------------------------------------------------------
# Test 3: A1 closure — applied_variant.json in worktree dir reaches train.py
# RED until Plan 10-02 (applied_variant.json) AND Plan 10-04 (dispatch) are done.
# ---------------------------------------------------------------------------


def test_iris_applied_variant_reaches_worktree_at_runtime(tmp_path):
    """The variant selection written by 'automil apply' reaches iris train.py at runtime.

    This is the key A1 closure test.  It simulates a worktree: no config.yaml
    initially — only applied_variant.json (the file written by Plan 10-02 into
    automil/archive/<node_id>/ and propagated via apply_overlay into the worktree).

    The test proves the variant selection reaches the consumer's train.py via
    applied_variant.json alone (not via config.yaml), making APL-01 correct even
    when config.yaml is gitignored and absent from the worktree.
    """
    import shutil

    automil_dir = tmp_path / "automil"
    automil_dir.mkdir()

    # Copy the classifier_v0 variant into the tmp automil/ directory.
    src_variant = IRIS_AUTOMIL / "variants" / "classifier_v0"
    if not src_variant.exists():
        pytest.skip(f"classifier_v0 variant not found at {src_variant}")

    dst_variants = automil_dir / "variants" / "classifier_v0"
    shutil.copytree(str(src_variant), str(dst_variants))

    # Write ONLY applied_variant.json — no config.yaml (simulates gitignored-config worktree).
    applied_variant = {
        "model": {"variant": "classifier_v0", "parent": None},
        "loss": {"variant": None},
        "policy": {"variant": None},
    }
    (automil_dir / "applied_variant.json").write_text(json.dumps(applied_variant, indent=2))

    result = _run_iris(tmp_path)

    assert result["status"] == "completed"
    assert result.get("variant_dispatched") == "classifier_v0", (
        "APL-01 A1 closure: iris train.py did not dispatch classifier_v0 from "
        "applied_variant.json. The variant must be read from applied_variant.json "
        "even when config.yaml is absent."
    )
    assert result["primary_value"] > 0
