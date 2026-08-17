"""nnMIL reports real sensitivity/specificity, via an add-on — not a vendored edit.

nnMIL's trainer never emitted these two. ``get_eval_metrics``
(lib/nnMIL/utilities/utils.py) returns only acc / bacc / kappa / nw_kappa /
weighted_f1 / loss / auroc, so ``normalize_nnmil_metrics`` fell through to
``setdefault(..., nan)`` on EVERY task, binary included. Every other arm
(CLAM, ABMIL, DTFD, TITAN) routes through ``compute_extended_metrics`` and
reports both, which left a hole in the results table on exactly one arm.

The fix is an add-on: autobench wraps the ``get_eval_metrics`` binding that
nnMIL's trainer module already holds, computes the two metrics from the
targets/preds that call is being handed anyway, and lets them flow out through
the trainer's normal metric dict. nnMIL's METRIC layer
(``benchmarks/lib/nnMIL/utilities/``) is not modified — pinned by
``test_vendored_nnmil_metric_layer_is_not_modified`` below. (The trainer files
do carry sanctioned benchmark instrumentation — the PolicyRuntime dispatch
calls and the A3 ``[epoch]``/``[selected]`` lines — which is exactly why the
pin is on the metric layer, not the whole tree.)

Both arms share ONE implementation (``evaluate.sensitivity_specificity``), so
the numbers are identical by construction rather than merely equivalent — a
stronger guarantee than the documented L-10 AUC asymmetry, where the two paths
agree only when every class is present in every fold.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from autobench.pipeline.evaluate import (
    compute_extended_metrics,
    sensitivity_specificity,
)
from autobench.pipeline.nnmil.metrics_addon import (
    install_sensitivity_specificity,
    with_sensitivity_specificity,
)


# --- the shared formula -----------------------------------------------------

class TestSharedFormula:
    """The extraction must be a pure refactor: identical numbers, both branches."""

    @pytest.mark.parametrize("n_classes", [2, 3, 4])
    def test_helper_matches_compute_extended_metrics(self, n_classes):
        rng = np.random.default_rng(11)
        y_true = rng.integers(0, n_classes, size=30)
        y_pred = rng.integers(0, n_classes, size=30)
        y_probs = np.eye(n_classes)[y_pred]

        full = compute_extended_metrics(y_true, y_probs, y_pred, n_classes)
        computed = sensitivity_specificity(y_true, y_pred, n_classes)

        assert computed  # the shape decides the key names; both must be present
        for name, value in computed.items():
            assert value == full[name]

    def test_binary_is_still_positive_class_only(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        computed = sensitivity_specificity(y_true, y_pred, 2)
        assert computed == {"sensitivity": 1.0, "specificity": 0.5}  # macro: 0.75


# --- the add-on wrapper -----------------------------------------------------

def _trainer_module():
    """Import nnMIL's trainer module the way production does.

    ``_imports`` performs the sys.path setup; without it a bare
    ``importorskip("nnMIL...")`` skips, which would silently drop exactly the
    assertions that prove the add-on seam works.
    """
    from autobench.pipeline.nnmil import _imports  # noqa: F401  (sys.path setup)

    import nnMIL.training.trainers.classification_trainer as trainer_mod

    return trainer_mod


def _vendored_utils():
    from autobench.pipeline.nnmil import _imports  # noqa: F401  (sys.path setup)

    import nnMIL.utilities.utils as utils

    return utils


def _fake_get_eval_metrics(targets_all, preds_all, probs_all=None,
                           unique_classes=None, get_report=True, prefix="",
                           roc_kwargs=None):
    """Stand-in with nnMIL's exact signature and key convention."""
    return {f"{prefix}/acc": 0.75, f"{prefix}/bacc": 0.70}


class TestWrapper:
    def test_adds_both_metrics_without_disturbing_the_originals(self):
        wrapped = with_sensitivity_specificity(_fake_get_eval_metrics)
        out = wrapped(
            targets_all=np.array([0, 0, 1, 1]),
            preds_all=np.array([0, 1, 1, 1]),
            unique_classes=[0, 1],
            prefix="val",
        )

        assert out["val/acc"] == 0.75          # untouched
        assert out["val/bacc"] == 0.70
        assert out["val/sensitivity"] == 1.0
        assert out["val/specificity"] == 0.5

    def test_multiclass_uses_the_macro_branch(self):
        wrapped = with_sensitivity_specificity(_fake_get_eval_metrics)
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 2, 2, 2, 2])
        out = wrapped(
            targets_all=y_true, preds_all=y_pred,
            unique_classes=[0, 1, 2], prefix="test",
        )

        assert out["test/macro_recall"] == pytest.approx(2 / 3)
        assert out["test/macro_specificity_ovr"] == pytest.approx(5 / 6)

    def test_class_count_comes_from_unique_classes(self):
        """The trainer passes list(range(num_classes)) — a fold missing a class
        must still be scored as multi-class, not silently as binary."""
        wrapped = with_sensitivity_specificity(_fake_get_eval_metrics)
        y_true = np.array([0, 0, 1, 1])      # class 2 absent from this fold
        y_pred = np.array([0, 1, 1, 1])

        as_three = wrapped(targets_all=y_true, preds_all=y_pred,
                           unique_classes=[0, 1, 2], prefix="v")
        as_two = wrapped(targets_all=y_true, preds_all=y_pred,
                         unique_classes=[0, 1], prefix="v")

        assert as_three["v/macro_specificity_ovr"] != as_two["v/specificity"]
        assert np.isfinite(as_three["v/macro_recall"])

    def test_a_failure_in_the_addon_cannot_break_evaluation(self):
        """Diagnostics must never take a training run down."""
        wrapped = with_sensitivity_specificity(_fake_get_eval_metrics)
        out = wrapped(targets_all=None, preds_all=None,
                      unique_classes=[0, 1], prefix="val")

        assert out["val/acc"] == 0.75
        assert "val/sensitivity" not in out

    def test_wrapper_does_not_mutate_the_wrapped_result(self):
        returned = {"val/acc": 0.75}

        def _returns_shared(**kwargs):
            return returned

        wrapped = with_sensitivity_specificity(_returns_shared)
        wrapped(targets_all=np.array([0, 1]), preds_all=np.array([0, 1]),
                unique_classes=[0, 1], prefix="val")

        assert returned == {"val/acc": 0.75}


@pytest.fixture
def pristine_binding():
    """Reset the trainer binding to nnMIL's own function, and restore afterwards.

    ``install_sensitivity_specificity`` mutates module state that outlives the
    test. Any earlier test in the session that touches the nnMIL training path
    leaves the binding already wrapped, which made an identity assertion here
    pass in isolation and fail in the full suite. Start from a known state, and
    leave the session exactly as found.
    """
    trainer_mod = _trainer_module()
    saved = trainer_mod.get_eval_metrics
    trainer_mod.get_eval_metrics = _vendored_utils().get_eval_metrics
    try:
        yield trainer_mod
    finally:
        trainer_mod.get_eval_metrics = saved


class TestInstall:
    def test_install_patches_the_trainer_binding(self, pristine_binding):
        trainer_mod = pristine_binding
        unwrapped = trainer_mod.get_eval_metrics

        install_sensitivity_specificity()
        assert trainer_mod.get_eval_metrics is not unwrapped

        out = trainer_mod.get_eval_metrics(
            targets_all=np.array([0, 0, 1, 1]),
            preds_all=np.array([0, 1, 1, 1]),
            unique_classes=[0, 1],
            prefix="val",
        )
        assert out["val/sensitivity"] == 1.0
        assert out["val/specificity"] == 0.5
        assert "val/bacc" in out       # nnMIL's own metrics still there

    def test_install_is_idempotent(self, pristine_binding):
        trainer_mod = pristine_binding

        install_sensitivity_specificity()
        once = trainer_mod.get_eval_metrics
        install_sensitivity_specificity()

        assert trainer_mod.get_eval_metrics is once

    def test_install_on_an_already_wrapped_binding_is_safe(self, pristine_binding):
        """The production call site re-enters this once per fold."""
        trainer_mod = pristine_binding
        for _ in range(3):
            assert install_sensitivity_specificity() is True

        out = trainer_mod.get_eval_metrics(
            targets_all=np.array([0, 0, 1, 1]),
            preds_all=np.array([0, 1, 1, 1]),
            unique_classes=[0, 1],
            prefix="val",
        )
        assert out["val/sensitivity"] == 1.0     # not double-wrapped into nonsense


# --- the constraint: this is an add-on ---------------------------------------

def test_vendored_nnmil_metric_layer_is_not_modified():
    """The metric mechanism stays an add-on: nnMIL's metric layer gains nothing.

    Asks git whether the vendored METRIC layer (``utilities/``, where
    ``get_eval_metrics`` lives) changed, rather than grepping one function body
    for three substrings. The grep version passed against a real violation
    demonstrated in review -- a new helper added to ``utils.py`` outside
    ``get_eval_metrics`` -- and missed any rename (``tpr``/``tnr``).

    Scoped to ``utilities/`` deliberately, not the whole vendored tree: the
    TRAINER files carry sanctioned benchmark instrumentation with in-repo
    precedent (the PolicyRuntime ``should_stop`` dispatch, the A3
    ``[selected] epoch=`` line), landed as reviewed commits. What must never
    move into the vendored tree is the metric computation itself -- that lives
    in ``autobench/pipeline/nnmil/metrics_addon.py`` and
    ``autobench/pipeline/evaluate.py``, shared with every other arm.

    Uncommitted edits are refused for the WHOLE vendored tree: any vendored
    change must be a deliberate, reviewed commit, never a loose working-tree
    edit. Committed edits are checked against the merge-base with the default
    branch, scoped to the metric layer.
    """
    repo = Path(__file__).resolve().parents[2]
    vendored = "benchmarks/lib/nnMIL"
    metric_layer = "benchmarks/lib/nnMIL/utilities"

    def _git(*args) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True,
        )

    if _git("rev-parse", "--git-dir").returncode != 0:
        pytest.skip("not a git checkout")

    dirty = _git("status", "--porcelain", "--", vendored).stdout.strip()
    assert not dirty, (
        f"benchmarks/lib/nnMIL has uncommitted changes:\n{dirty}\n"
        "Vendored edits must be deliberate, reviewed commits -- and the metric "
        "mechanism in particular belongs in "
        "autobench/pipeline/nnmil/metrics_addon.py, not in vendored code."
    )

    # Committed-diff guard covers the FULL vendored tree minus the explicit
    # allowlist of instrumentation-touched files, so models/, losses/, and
    # data/ stay guarded — a loss tweak or architecture edit hidden in the
    # vendored tree would move the arm's numbers for a reason invisible in
    # autobench/ while the suite stayed green.
    allowed_vendored_edits = {
        "benchmarks/lib/nnMIL/training/trainers/classification_trainer.py",
        "benchmarks/lib/nnMIL/training/trainers/survival_trainer.py",
        "benchmarks/lib/nnMIL/training/trainers/survival_porpoise_trainer.py",
        "benchmarks/lib/nnMIL/training/callbacks/early_stopping.py",
    }
    base = _git("merge-base", "HEAD", "origin/main").stdout.strip()
    if base:
        committed = [
            p for p in _git(
                "diff", "--name-only", base, "HEAD", "--", vendored,
            ).stdout.split()
            if p not in allowed_vendored_edits
        ]
        assert not committed, (
            f"benchmarks/lib/nnMIL was edited outside the sanctioned "
            f"instrumentation allowlist on this branch:\n"
            + "\n".join(committed)
            + "\nVendored edits beyond the PolicyRuntime dispatch / A3 "
            "instrumentation files change what the paper compares."
        )
        assert metric_layer  # the metric layer is never allowlisted


def test_vendored_clam_is_not_modified_outside_instrumentation():
    """CLAM twin of the nnMIL pin — it had no guard at all while this branch
    was already editing its ``core_utils.py``."""
    repo = Path(__file__).resolve().parents[2]
    vendored = "benchmarks/lib/CLAM"
    allowed_vendored_edits = {
        "benchmarks/lib/CLAM/utils/core_utils.py",
    }

    def _git(*args) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True,
        )

    if _git("rev-parse", "--git-dir").returncode != 0:
        pytest.skip("not a git checkout")

    dirty = _git("status", "--porcelain", "--", vendored).stdout.strip()
    assert not dirty, (
        f"benchmarks/lib/CLAM has uncommitted changes:\n{dirty}\n"
        "Vendored edits must be deliberate, reviewed commits."
    )

    base = _git("merge-base", "HEAD", "origin/main").stdout.strip()
    if base:
        committed = [
            p for p in _git(
                "diff", "--name-only", base, "HEAD", "--", vendored,
            ).stdout.split()
            if p not in allowed_vendored_edits
        ]
        assert not committed, (
            f"benchmarks/lib/CLAM was edited outside the sanctioned "
            f"instrumentation allowlist on this branch:\n"
            + "\n".join(committed)
        )


# --- end of the chain: the normalized metric dict -----------------------------

class TestNormalization:
    def test_mapped_through_when_the_trainer_reports_them(self):
        from autobench.pipeline.nnmil.evaluate import normalize_nnmil_metrics

        out = normalize_nnmil_metrics(
            {
                "test_test/bacc": 0.70,
                "test_test/auroc": 0.80,
                "test_test/sensitivity": 0.65,
                "test_test/specificity": 0.72,
            },
            split="test",
        )

        assert out["sensitivity"] == 0.65
        assert out["specificity"] == 0.72

    def test_still_nan_when_absent(self):
        """Fallback is preserved: an un-installed path degrades, never crashes."""
        from autobench.pipeline.nnmil.evaluate import normalize_nnmil_metrics

        out = normalize_nnmil_metrics({"test_test/bacc": 0.70}, split="test")

        assert np.isnan(out["sensitivity"])
        assert np.isnan(out["specificity"])


class TestReviewFindings:
    """Regressions for the PR-49 review findings."""

    def test_multiclass_does_not_also_emit_phantom_binary_nans(self):
        """A real macro_recall must not ship beside a NaN `sensitivity`.

        Defaulting unconditionally opened a new cross-arm asymmetry on 3-class
        (no sibling arm emits sensitivity there) and made a NaN indistinguishable
        from "the add-on failed" — a diagnostic lying about its own health.
        """
        from autobench.pipeline.nnmil.evaluate import normalize_nnmil_metrics

        out = normalize_nnmil_metrics(
            {
                "test_test/bacc": 0.70,
                "test_test/macro_recall": 0.61,
                "test_test/macro_specificity_ovr": 0.80,
            },
            split="test",
        )

        assert out["macro_recall"] == 0.61
        assert "sensitivity" not in out
        assert "specificity" not in out

    def test_binary_still_gets_the_nan_fallback_when_the_addon_is_absent(self):
        from autobench.pipeline.nnmil.evaluate import normalize_nnmil_metrics

        out = normalize_nnmil_metrics({"test_test/bacc": 0.70}, split="test")

        assert np.isnan(out["sensitivity"])
        assert np.isnan(out["specificity"])

    def test_positional_call_shapes_still_get_the_metrics(self):
        """nnMIL calls all-keyword today; positional must not silently disable."""
        wrapped = with_sensitivity_specificity(_fake_get_eval_metrics)
        y_true, y_pred = np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])

        out = wrapped(y_true, y_pred, None, [0, 1], True, "val")

        assert out["val/sensitivity"] == 1.0
        assert out["val/specificity"] == 0.5

    def test_a_non_mapping_return_is_contained(self):
        """The merge sits inside the try, so this cannot escape the wrapper."""
        wrapped = with_sensitivity_specificity(lambda **kw: None)
        out = wrapped(targets_all=np.array([0, 1]), preds_all=np.array([0, 1]),
                      unique_classes=[0, 1], prefix="val")
        assert out is None

    def test_the_except_branch_is_actually_exercised(self):
        """Not just the early return — force a raise inside the try."""
        wrapped = with_sensitivity_specificity(_fake_get_eval_metrics)

        out = wrapped(targets_all=np.array([0, 0, 1]), preds_all=np.array([0, 1]),
                      unique_classes=[0, 1], prefix="val")

        assert out == {"val/acc": 0.75, "val/bacc": 0.70}   # nnMIL's own, intact

    def test_wrapper_keeps_the_wrapped_identity(self):
        wrapped = with_sensitivity_specificity(_fake_get_eval_metrics)
        assert wrapped.__name__ == "_fake_get_eval_metrics"
