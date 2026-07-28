"""M-14 (audit 2026-07-23): the budget-cell identity must include the task.

Without it, ``(GBM, uni_v2, clam_mb)`` on the classification task and on the OS
survival task mapped to the SAME cell — whichever search ran first drained the
shared clock and the second was refused (starved), unevenly across cells.
"""
from __future__ import annotations

from automil.cells.state import make_cell_id


def test_task_changes_cell_identity():
    a = make_cell_id("cptac_gbm", "uni_v2", "clam mb", "tp53")
    b = make_cell_id("cptac_gbm", "uni_v2", "clam mb", "os")
    assert a != b


def test_same_task_is_stable():
    a = make_cell_id("cptac_gbm", "uni_v2", "clam mb", "os")
    b = make_cell_id("cptac_gbm", "uni_v2", "clam mb", "os")
    assert a == b


def test_none_task_reproduces_legacy_identity():
    """Existing cells (created before M-14) must keep their ids."""
    legacy = make_cell_id("ccrcc", "uni-v2", "clam_sb")
    assert make_cell_id("ccrcc", "uni-v2", "clam_sb", None) == legacy
    # An empty string is treated as absent, not as a distinct task.
    assert make_cell_id("ccrcc", "uni-v2", "clam_sb", "") == legacy


def test_id_shape_unchanged():
    cid = make_cell_id("d", "e", "m", "t")
    assert len(cid) == 16
    int(cid, 16)  # hex
