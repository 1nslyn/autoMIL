"""APL-02 real-run: verify a CLAM variant produces a different primary_value from baseline.

This entire module is workstation-only — marked with pytest.mark.workstation.
It requires:
  - AUTOBENCH_CCRCC_ROOT environment variable pointing to real CCRCC dataset
  - A GPU workstation capable of running a full CLAM experiment

CI skips this module automatically because AUTOBENCH_CCRCC_ROOT is not set.

Status: STUB — the single test raises pytest.fail until Plan 10-03 wires the
CLAM dispatch in run_experiment.py.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.workstation


# ---------------------------------------------------------------------------
# Real-run primary_value delta test (workstation-only)
# RED until Plan 10-03 wires variant dispatch in run_experiment.py.
# ---------------------------------------------------------------------------


def test_real_clam_run_primary_value_differs_from_baseline():
    """A CLAM experiment run with a model variant applied produces a different primary_value.

    Prerequisite: AUTOBENCH_CCRCC_ROOT must be set to a valid CCRCC dataset root.
    Skipped when the env var is absent (CI behavior).

    What this test proves (APL-02 D-04 verification split):
      - Running an experiment with 'automil apply' having set model.variant in
        config.yaml causes run_experiment.py to pass different args to clam_train,
        resulting in a measurably different primary-metric value from the baseline.
      - The primary_value delta is > noise threshold (TBD by Plan 10-03 implementer).

    RED until Plan 10-03 implements variant dispatch wiring in run_experiment.py.
    """
    ccrcc_root = os.environ.get("AUTOBENCH_CCRCC_ROOT")
    if not ccrcc_root:
        pytest.skip("AUTOBENCH_CCRCC_ROOT not set — workstation-only test skipped")

    pytest.fail(
        "APL-02 real-run: not yet implemented — requires CLAM dispatch wired in "
        "run_experiment.py (Plan 10-03). Once implemented, this test should: "
        "(1) run a baseline experiment, (2) apply a model variant, "
        "(3) run the experiment again, (4) assert primary_value differs by > noise."
    )
