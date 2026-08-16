"""CR-1b (audit 2026-07-23): the selection composite is derived from the declared
VALIDATION metrics, not trusted verbatim from the agent-written result.json.

Before this, a training script whose ``composite`` came from the sealed test block
produced a schema-valid result, the graph selected on it, and nothing detected the
leak — defeating the val-firewall's central claim.
"""
from __future__ import annotations

import pytest

from automil.graph import ExperimentGraph
from automil.scoring import (
    DEFAULT_FORMULA,
    composite_disagrees,
    fold_composite_entries,
    known_formula,
    recompute_composite,
    recompute_composite_se,
    recompute_refused,
)
from automil.terminal_writer import write_terminal_state


# --- reducer unit tests ---

def test_mean_reproduces_classification_composite():
    # The established formula is (val_auc + val_bacc) / 2 — 'mean' reproduces it.
    m = {"val_auc": 0.90, "val_bacc": 0.80}
    assert recompute_composite(m, "mean") == pytest.approx(0.85)


def test_mean_reproduces_survival_composite():
    # Survival composite is the lone val_c_index — mean of one value is itself.
    assert recompute_composite({"val_c_index": 0.62}, "mean") == pytest.approx(0.62)


def test_opt_out_returns_none():
    assert recompute_composite({"val_auc": 0.9}, "trust_reported") is None


def test_empty_metrics_returns_none():
    assert recompute_composite({}, DEFAULT_FORMULA) is None
    assert recompute_composite({"note": "text"}, DEFAULT_FORMULA) is None


def test_booleans_excluded():
    # bool is an int subclass — must not be averaged in as 1.0/0.0.
    assert recompute_composite({"val_auc": 0.8, "ok": True}, "mean") == pytest.approx(0.8)


def test_unknown_formula_raises():
    with pytest.raises(ValueError):
        recompute_composite({"val_auc": 0.8}, "bogus")


def test_disagreement_tolerates_4dp_rounding():
    # result.json rounds composite and metrics to 4dp; that must not trip the guard.
    assert not composite_disagrees(0.8500, 0.85004)
    assert composite_disagrees(0.8500, 0.9500)


# --- val_* metric selectors (round 2: the composite is the primary metric) ---

def test_selector_is_known_and_returns_the_named_metric():
    assert known_formula("val_auc") and known_formula("val_c_index")
    m = {"val_auc": 0.90, "val_bacc": 0.60}
    # Companions are recorded but do not vote.
    assert recompute_composite(m, "val_auc") == pytest.approx(0.90)


def test_empty_formula_resolves_to_the_default_not_the_opt_out():
    m = {"val_auc": 0.90, "val_bacc": 0.70}
    assert recompute_composite(m, "") == pytest.approx(0.80)
    assert recompute_composite(m, None) == pytest.approx(0.80)


def test_unsupported_selector_refuses_instead_of_trusting():
    """B2 under selectors: a typo'd selector cannot be caught statically, so a
    present metrics block that cannot support the formula must REFUSE (the
    ingest mouths score the payload 0.0) rather than silently fall back to the
    reported scalar."""
    m = {"val_auc": 0.90, "val_bacc": 0.60}
    assert recompute_composite(m, "val_aucc") is None        # typo'd selector
    assert recompute_refused(m, "val_aucc") is True
    assert recompute_refused(m, "val_auc") is False           # healthy
    assert recompute_refused({}, "val_auc") is False          # no evidence ≠ refusal
    assert recompute_refused(None, "val_auc") is False
    assert recompute_refused(m, "trust_reported") is False    # opt-out never refuses
    assert recompute_refused(
        {"val_auc": float("nan"), "val_bacc": 0.6}, "val_auc") is True
    # A mean-reducer block with no finite value refuses too — same principle.
    assert recompute_refused({"val_auc": float("nan")}, "mean") is True


def test_fold_entries_recompute_under_a_selector():
    res = {"validation_folds": [
        {"fold_index": 0, "metrics": {"val_auc": 0.9, "val_bacc": 0.5},
         "composite": 0.7},   # stale mean — the selector value must win
    ]}
    assert fold_composite_entries(res, "val_auc") == [
        {"fold_index": 0, "composite": 0.9},
    ]


def test_fold_entry_that_cannot_support_the_formula_is_dropped():
    res = {"validation_folds": [
        {"fold_index": 0, "metrics": {"val_auc": 0.9, "val_bacc": 0.5},
         "composite": 0.9},
        {"fold_index": 1, "metrics": {"val_auc": None, "val_bacc": 0.6},
         "composite": 0.99},   # fabricated value must NOT survive
    ]}
    assert fold_composite_entries(res, "val_auc") == [
        {"fold_index": 0, "composite": 0.9},
    ]


def test_fold_entry_without_metrics_keeps_the_reported_value():
    # Legacy grace: state artifacts predating the fold-metrics contract.
    res = {"validation_folds": [{"fold_index": 3, "composite": 0.61}]}
    assert fold_composite_entries(res, "val_auc") == [
        {"fold_index": 3, "composite": 0.61},
    ]


def test_companion_lossy_fold_is_not_resurrected_by_a_selector():
    """The projection's fold validity spans the FULL recorded evidence, like
    the trainer's and the campaign validator's: a fold the trainer nulled
    (lost companion, healthy selector key) must not come back one layer up —
    that would re-open the two-views-of-one-fold seam and shift the marginal
    SE off the payload's own n_valid_folds."""
    import math

    res = {"validation_folds": [
        {"fold_index": 0, "metrics": {"val_auc": 0.70, "val_bacc": 0.60},
         "composite": 0.70},
        {"fold_index": 1, "metrics": {"val_auc": 0.72, "val_bacc": None},
         "composite": None},   # trainer: invalid fold
        {"fold_index": 2, "metrics": {"val_auc": 0.68, "val_bacc": 0.60},
         "composite": 0.68},
    ]}
    assert fold_composite_entries(res, "val_auc") == [
        {"fold_index": 0, "composite": 0.70},
        {"fold_index": 2, "composite": 0.68},
    ]
    # And the marginal SE agrees with the trainer's two-fold measurement.
    assert recompute_composite_se(res, "val_auc") == pytest.approx(
        0.02 / math.sqrt(2) / math.sqrt(2))


def test_marginal_se_measures_the_recomputed_projection():
    """The SE must be computed over the same per-fold values the graph stores —
    reported fold composites shaped flat must not zero the noise floor."""
    import math

    res = {"validation_folds": [
        {"fold_index": i, "metrics": {"val_auc": v, "val_bacc": 0.5},
         "composite": 0.7}    # flat reported vector — a lie
        for i, v in enumerate((0.6, 0.7, 0.8))
    ]}
    se = recompute_composite_se(res, "val_auc")
    assert se == pytest.approx(0.1 / math.sqrt(3))


# --- end-to-end through terminal_writer ---

def _run(tmp_path, result, formula=None):
    adir = tmp_path / "automil"
    adir.mkdir()
    graph = ExperimentGraph(path=str(adir / "graph.json"))
    if formula is not None:
        graph.meta.setdefault("scoring", {})["formula"] = formula
    nid = graph.add_proposed("root", "exp", [], kind="hp")
    graph.save()

    completed = adir / "orchestrator" / "completed"
    archive = adir / "orchestrator" / "archive" / nid
    completed.mkdir(parents=True)
    archive.mkdir(parents=True)

    write_terminal_state(
        node_id=nid, result=result, graph=graph,
        completed_dir=completed, archive_dir=archive,
        results_tsv_writer=lambda *a, **k: None,
        spec={}, elapsed_s=1.0, gpu_id=0,
    )
    return nid, ExperimentGraph(path=str(adir / "graph.json"))


def test_test_derived_composite_is_overridden_by_val(tmp_path):
    """The exploit: composite reported from the sealed test block."""
    nid, g = _run(tmp_path, {
        "status": "completed",
        "composite": 0.99,                                  # from test — a lie
        "metrics": {"val_auc": 0.70, "val_bacc": 0.60},     # true val → 0.65
        "held_out": {"test_auc": 0.99},
    })
    node = g.get_node(nid)
    assert node["composite"] == pytest.approx(0.65)
    assert node["metadata"]["composite_disagreement"]["reported"] == pytest.approx(0.99)


def test_honest_composite_passes_through_unflagged(tmp_path):
    nid, g = _run(tmp_path, {
        "status": "completed",
        "composite": 0.85,
        "metrics": {"val_auc": 0.90, "val_bacc": 0.80},
        "held_out": {"test_auc": 0.5},
    })
    node = g.get_node(nid)
    assert node["composite"] == pytest.approx(0.85)
    assert "composite_disagreement" not in (node.get("metadata") or {})


def test_selector_end_to_end_scores_on_the_named_metric(tmp_path):
    """A graph declaring `formula: val_auc` overrides a stale mean composite."""
    nid, g = _run(tmp_path, {
        "status": "completed",
        "composite": 0.75,                                   # (auc+bacc)/2 — stale
        "metrics": {"val_auc": 0.90, "val_bacc": 0.60},
        "held_out": {"test_auc": 0.5},
    }, formula="val_auc")
    node = g.get_node(nid)
    assert node["composite"] == pytest.approx(0.90)
    assert node["metadata"]["composite_disagreement"]["reported"] == pytest.approx(0.75)


def test_selector_miss_fails_the_node_closed(tmp_path):
    """Metrics present but no selector key: never trust the reported scalar."""
    nid, g = _run(tmp_path, {
        "status": "completed",
        "composite": 0.99,                     # would win selection if trusted
        "metrics": {"val_bacc": 0.80},         # selector key stripped
    }, formula="val_auc")
    node = g.get_node(nid)
    assert node["composite"] == 0.0
    assert node["metadata"]["composite_disagreement"]["refused"] is True
