"""CR-1b (audit 2026-07-23): the selection primary_value is derived from the declared
VALIDATION metrics, not trusted verbatim from the agent-written result.json.

Before this, a training script whose ``primary_value`` came from the sealed test block
produced a schema-valid result, the graph selected on it, and nothing detected the
leak — defeating the val-firewall's central claim.
"""
from __future__ import annotations

import pytest

from automil.graph import ExperimentGraph
from automil.scoring import (
    DEFAULT_FORMULA,
    primary_value_disagrees,
    fold_primary_value_entries,
    known_formula,
    recompute_primary_value,
    recompute_primary_se,
    recompute_refused,
)
from automil.terminal_writer import write_terminal_state


# --- reducer unit tests ---

def test_mean_reproduces_classification_primary_value():
    # The established formula is (val_auc + val_bacc) / 2 — 'mean' reproduces it.
    m = {"val_auc": 0.90, "val_bacc": 0.80}
    assert recompute_primary_value(m, "mean") == pytest.approx(0.85)


def test_mean_reproduces_survival_primary_value():
    # Survival primary_value is the lone val_c_index — mean of one value is itself.
    assert recompute_primary_value({"val_c_index": 0.62}, "mean") == pytest.approx(0.62)


def test_opt_out_returns_none():
    assert recompute_primary_value({"val_auc": 0.9}, "trust_reported") is None


def test_empty_metrics_returns_none():
    assert recompute_primary_value({}, DEFAULT_FORMULA) is None
    assert recompute_primary_value({"note": "text"}, DEFAULT_FORMULA) is None


def test_booleans_excluded():
    # bool is an int subclass — must not be averaged in as 1.0/0.0.
    assert recompute_primary_value({"val_auc": 0.8, "ok": True}, "mean") == pytest.approx(0.8)


def test_unknown_formula_raises():
    with pytest.raises(ValueError):
        recompute_primary_value({"val_auc": 0.8}, "bogus")


def test_disagreement_tolerates_4dp_rounding():
    # result.json rounds primary_value and metrics to 4dp; that must not trip the guard.
    assert not primary_value_disagrees(0.8500, 0.85004)
    assert primary_value_disagrees(0.8500, 0.9500)


# --- val_* metric selectors (round 2: the primary_value is the primary metric) ---

def test_selector_is_known_and_returns_the_named_metric():
    assert known_formula("val_auc") and known_formula("val_c_index")
    m = {"val_auc": 0.90, "val_bacc": 0.60}
    # Companions are recorded but do not vote.
    assert recompute_primary_value(m, "val_auc") == pytest.approx(0.90)


def test_empty_formula_resolves_to_the_default_not_the_opt_out():
    m = {"val_auc": 0.90, "val_bacc": 0.70}
    assert recompute_primary_value(m, "") == pytest.approx(0.80)
    assert recompute_primary_value(m, None) == pytest.approx(0.80)


def test_unsupported_selector_refuses_instead_of_trusting():
    """B2 under selectors: a typo'd selector cannot be caught statically, so a
    present metrics block that cannot support the formula must REFUSE (the
    ingest mouths score the payload 0.0) rather than silently fall back to the
    reported scalar."""
    m = {"val_auc": 0.90, "val_bacc": 0.60}
    assert recompute_primary_value(m, "val_aucc") is None        # typo'd selector
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
         "primary_value": 0.7},   # stale mean — the selector value must win
    ]}
    assert fold_primary_value_entries(res, "val_auc") == [
        {"fold_index": 0, "primary_value": 0.9},
    ]


def test_fold_entry_that_cannot_support_the_formula_is_dropped():
    res = {"validation_folds": [
        {"fold_index": 0, "metrics": {"val_auc": 0.9, "val_bacc": 0.5},
         "primary_value": 0.9},
        {"fold_index": 1, "metrics": {"val_auc": None, "val_bacc": 0.6},
         "primary_value": 0.99},   # fabricated value must NOT survive
    ]}
    assert fold_primary_value_entries(res, "val_auc") == [
        {"fold_index": 0, "primary_value": 0.9},
    ]


def test_fold_entry_without_metrics_is_dropped_never_trusted():
    """A bare reported fold value is unverifiable evidence. Trusting it
    hands the paired margin to the agent: fold values fabricated as
    parent + uniform delta (with an honest aggregate, so the mean-identity
    guard passes) drive the paired delta-SE to 0.0 and drop the keep bar
    to the bare δ floor. Verifiable fold metrics or the marginal basis —
    nothing in between."""
    res = {"validation_folds": [{"fold_index": 3, "primary_value": 0.61}]}
    assert fold_primary_value_entries(res, "val_auc") is None


def test_forged_uniform_delta_folds_cannot_zero_the_paired_se():
    """The attack shape end to end: every fold entry carries only a
    reported value (parent fold + constant delta). The projection must
    yield nothing, forcing the marginal-SE fallback."""
    res = {
        "primary_value": 0.75,
        "metrics": {"val_auc": 0.75},
        "validation_folds": [
            {"fold_index": i, "primary_value": 0.70 + 0.01 * i + 0.02}
            for i in range(3)
        ],
    }
    assert fold_primary_value_entries(res, "val_auc") is None


def test_fold_metrics_with_held_out_keys_fail_the_payload_closed():
    """validation_folds[*].metrics is a validation surface: a test_auc there
    feeds the recomputed per-fold values (mean reducer averages it straight
    into the paired margin) and stays in the agent-visible archive."""
    from automil.scoring import ingest_signal

    res = {
        "status": "completed",
        "primary_value": 0.9,
        "metrics": {"val_auc": 0.9},
        "validation_folds": [
            {"fold_index": 0,
             "metrics": {"val_auc": 0.9, "test_auc": 0.95},
             "primary_value": 0.9},
        ],
    }
    leaking, recomputed, se, refused = ingest_signal(res, "mean")
    assert leaking == ("validation_folds[0].metrics.test_auc",)
    assert recomputed is None and se is None and refused is False


def test_unknown_frozen_formula_refuses_instead_of_trusting():
    """One typo'd reducer in a hand-edited graph must not silently disable
    CR-1b for every subsequent result."""
    from automil.scoring import ingest_signal

    m = {"val_auc": 0.90, "val_bacc": 0.60}
    assert recompute_refused(m, "meen") is True
    leaking, recomputed, se, refused = ingest_signal(
        {"status": "completed", "primary_value": 0.99, "metrics": m}, "meen",
    )
    assert leaking == () and recomputed is None and refused is True


def test_companion_lossy_fold_is_not_resurrected_by_a_selector():
    """The projection's fold validity spans the FULL recorded evidence, like
    the trainer's and the campaign validator's: a fold the trainer nulled
    (lost companion, healthy selector key) must not come back one layer up —
    that would re-open the two-views-of-one-fold seam and shift the marginal
    SE off the payload's own n_valid_folds."""
    import math

    res = {"validation_folds": [
        {"fold_index": 0, "metrics": {"val_auc": 0.70, "val_bacc": 0.60},
         "primary_value": 0.70},
        {"fold_index": 1, "metrics": {"val_auc": 0.72, "val_bacc": None},
         "primary_value": None},   # trainer: invalid fold
        {"fold_index": 2, "metrics": {"val_auc": 0.68, "val_bacc": 0.60},
         "primary_value": 0.68},
    ]}
    assert fold_primary_value_entries(res, "val_auc") == [
        {"fold_index": 0, "primary_value": 0.70},
        {"fold_index": 2, "primary_value": 0.68},
    ]
    # And the marginal SE agrees with the trainer's two-fold measurement.
    assert recompute_primary_se(res, "val_auc") == pytest.approx(
        0.02 / math.sqrt(2) / math.sqrt(2))


def test_marginal_se_measures_the_recomputed_projection():
    """The SE must be computed over the same per-fold values the graph stores —
    reported fold primary values shaped flat must not zero the noise floor."""
    import math

    res = {"validation_folds": [
        {"fold_index": i, "metrics": {"val_auc": v, "val_bacc": 0.5},
         "primary_value": 0.7}    # flat reported vector — a lie
        for i, v in enumerate((0.6, 0.7, 0.8))
    ]}
    se = recompute_primary_se(res, "val_auc")
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


def test_test_derived_primary_value_is_overridden_by_val(tmp_path):
    """The exploit: primary_value reported from the sealed test block."""
    nid, g = _run(tmp_path, {
        "status": "completed",
        "primary_value": 0.99,                                  # from test — a lie
        "metrics": {"val_auc": 0.70, "val_bacc": 0.60},     # true val → 0.65
        "held_out": {"test_auc": 0.99},
    })
    node = g.get_node(nid)
    assert node["primary_value"] == pytest.approx(0.65)
    # The stamp must NOT carry the raw reported scalar: the training
    # script (which sees test) wrote it, so republishing it on an
    # agent-facing surface is a one-scalar exfiltration channel.
    assert "reported" not in node["metadata"]["primary_value_disagreement"]


def test_honest_primary_value_passes_through_unflagged(tmp_path):
    nid, g = _run(tmp_path, {
        "status": "completed",
        "primary_value": 0.85,
        "metrics": {"val_auc": 0.90, "val_bacc": 0.80},
        "held_out": {"test_auc": 0.5},
    })
    node = g.get_node(nid)
    assert node["primary_value"] == pytest.approx(0.85)
    assert "primary_value_disagreement" not in (node.get("metadata") or {})


def test_selector_end_to_end_scores_on_the_named_metric(tmp_path):
    """A graph declaring `formula: val_auc` overrides a stale mean primary_value."""
    nid, g = _run(tmp_path, {
        "status": "completed",
        "primary_value": 0.75,                                   # (auc+bacc)/2 — stale
        "metrics": {"val_auc": 0.90, "val_bacc": 0.60},
        "held_out": {"test_auc": 0.5},
    }, formula="val_auc")
    node = g.get_node(nid)
    assert node["primary_value"] == pytest.approx(0.90)
    assert "reported" not in node["metadata"]["primary_value_disagreement"]


def test_selector_miss_fails_the_node_closed(tmp_path):
    """Metrics present but no selector key: never trust the reported scalar."""
    nid, g = _run(tmp_path, {
        "status": "completed",
        "primary_value": 0.99,                     # would win selection if trusted
        "metrics": {"val_bacc": 0.80},         # selector key stripped
    }, formula="val_auc")
    node = g.get_node(nid)
    assert node["primary_value"] == 0.0
    assert node["metadata"]["primary_value_disagreement"]["refused"] is True


def test_selector_miss_refusal_reaches_every_agent_facing_artifact(tmp_path):
    """The refusal must not stop at the graph node: completed/<node>.json,
    archive result.json, and the results.tsv row are what the agent (and
    reconcile) read back — leaving the refused 0.99 there would publish
    exactly the scalar the framework refused, diverging from graph.json."""
    import json

    adir = tmp_path / "automil"
    adir.mkdir()
    graph = ExperimentGraph(path=str(adir / "graph.json"))
    graph.meta.setdefault("scoring", {})["formula"] = "val_auc"
    nid = graph.add_proposed("root", "exp", [], kind="hp")
    graph.save()

    completed = adir / "orchestrator" / "completed"
    archive = adir / "orchestrator" / "archive" / nid
    completed.mkdir(parents=True)
    archive.mkdir(parents=True)

    tsv_rows = []
    write_terminal_state(
        node_id=nid,
        result={
            "status": "completed",
            "primary_value": 0.99,
            "metrics": {"val_bacc": 0.80},     # selector key stripped
        },
        graph=graph,
        completed_dir=completed, archive_dir=archive,
        results_tsv_writer=lambda n, r, **k: tsv_rows.append((n, r)),
        spec={}, elapsed_s=1.0, gpu_id=0,
    )

    completion = json.loads((completed / f"{nid}.json").read_text())
    assert completion["primary_value"] == 0.0

    archived = json.loads((archive / "result.json").read_text())
    assert archived["primary_value"] == 0.0
    assert archived["metadata"]["primary_value_disagreement"]["refused"] is True

    assert len(tsv_rows) == 1
    assert tsv_rows[0][1]["primary_value"] == 0.0


def test_recompute_survives_a_missing_graph_node(tmp_path):
    """The CR-1b sanitation must not be conditional on the graph update
    succeeding: with the node absent from the graph (or the lock failing),
    completed/, the archive result.json and the results.tsv row must still
    carry the val-derived value, never the raw reported scalar."""
    import json

    adir = tmp_path / "automil"
    adir.mkdir()
    graph = ExperimentGraph(path=str(adir / "graph.json"))
    graph.meta.setdefault("scoring", {})["formula"] = "val_auc"
    graph.save()

    completed = adir / "orchestrator" / "completed"
    archive = adir / "orchestrator" / "archive" / "node_9999"
    completed.mkdir(parents=True)
    archive.mkdir(parents=True)

    tsv_rows = []
    write_terminal_state(
        node_id="node_9999",              # NOT in the graph
        result={
            "status": "completed",
            "primary_value": 0.99,            # reported — must not survive
            "metrics": {"val_auc": 0.70},
        },
        graph=graph,
        completed_dir=completed, archive_dir=archive,
        results_tsv_writer=lambda n, r, **k: tsv_rows.append((n, r)),
        spec={}, elapsed_s=1.0, gpu_id=0,
    )

    completion = json.loads((completed / "node_9999.json").read_text())
    assert completion["primary_value"] == pytest.approx(0.70)
    archived = json.loads((archive / "result.json").read_text())
    assert archived["primary_value"] == pytest.approx(0.70)
    assert tsv_rows[0][1]["primary_value"] == pytest.approx(0.70)
