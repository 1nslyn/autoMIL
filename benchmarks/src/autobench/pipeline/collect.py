"""Cross-cohort results collector: ``summary.json`` -> DataFrames for the preprint.

FIG-0 (paper/preprint/READINESS-2026-07-28.md Sec.0, Sec.2.3): no figure in the paper
plan had producing code that read real results. The only code that ever turned
real fold metrics into figures was ``tasks/baseline_summary/scripts/`` -- gitignored,
and structurally broken against the current five-framework / TCGA+CPTAC roster:

  - it globs ``TCGA-*`` only, so both CPTAC cohorts are invisible;
  - it hardcodes ``for fw in ("clam", "nnmil")``, so abmil/dtfd/titan are never walked;
  - it keys on the wrong model names (``ab_mil`` instead of ``abmil``; abmil and
    dtfd_mil are their own frameworks, not nnMIL model types);
  - its ``METRIC_KEYS`` has no ``c_index``, so every survival experiment is invisible;
  - its walk assumes a fixed ``task/encoder/model`` depth with no ``survival_loss``
    level, so cox and nllsurv variants collide.

This module replaces it with a framework/strategy-agnostic walker
(:func:`collect_summaries`) plus two DataFrame builders that both autobench and
``paper/preprint/figures`` can consume: one row per experiment
(:func:`summaries_to_frame`) and one row per per-fold metric
(:func:`per_fold_frame`).

Path shape (``ExperimentConfig.results_subdir``, ``pipeline/config.py``)::

    results/{framework}/{strategy}/{task}/{encoder}/{model}[/{survival_loss}]/s{seed}/summary.json

Note the OPTIONAL ``survival_loss`` level (classification experiments skip it)
and the trailing ``s{seed}`` level -- depth is 6 or 7 directories below
``results/``, never fixed, so this module walks with a recursive glob rather
than assuming a level count.
"""

from __future__ import annotations

import glob
import json
import os

import pandas as pd

from autobench.pipeline.orchestrator import aggregate_results

__all__ = ["collect_summaries", "summaries_to_frame", "per_fold_frame"]


def collect_summaries(roots: list[str] | list[tuple[str, str]]) -> list[dict]:
    """Load every ``summary.json`` under ``results/**/`` for each root.

    Each ``root`` is a ``benchmark_dir`` (e.g. one per dataset cohort) -- this
    lets one call pool cohorts that each have their own ``benchmark_dir``
    (the roster spans TCGA *and* CPTAC; never filter/assume a shared prefix).
    A root may be given either as a plain path or as a ``(label, path)`` pair;
    see the cohort-identity note below for why the pair form is the safe one.

    A single recursive glob (``results/**/summary.json``) is used instead of
    depth-hardcoded patterns, because the ``survival_loss`` level is optional:
    classification experiments sit 6 levels below ``results/``, survival
    experiments sit 7. ``glob(..., recursive=True)`` matches both without
    caring which.

    Unreadable or absent files are skipped rather than raising: a half-written
    ``summary.json`` (mid-write on a live results tree), a corrupt one, or a
    root with no ``results/`` directory at all must not crash a collection run.

    **Cohort identity is stamped from the root, because the runners do not
    record it.** No framework runner writes a ``dataset`` key into
    ``summary.json`` -- verified against the completed phase-2 campaign, where
    0 of 195 real summaries carry one. Since ``aggregate_results`` reads
    ``s.get("dataset", "")``, an unstamped pooled collection gives every
    experiment the same empty cohort, and any per-cohort figure silently
    averages all five cohorts into one row. Each summary therefore gets
    ``dataset`` filled in from its root's label (or, unlabelled, the root
    directory's basename) -- but **only when the summary does not already carry
    a non-empty one**, so a runner that does record it always wins.

    Stamping returns new dicts and never rewrites the file on disk; the results
    tree is read-only to this module.
    """
    summaries: list[dict] = []
    for entry in roots:
        label, root = entry if isinstance(entry, tuple) else (None, entry)
        results_root = os.path.join(root, "results")
        if not os.path.isdir(results_root):
            continue
        fallback = label or os.path.basename(os.path.normpath(root))
        pattern = os.path.join(results_root, "**", "summary.json")
        for path in sorted(glob.glob(pattern, recursive=True)):
            try:
                with open(path) as f:
                    summary = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(summary, dict):
                continue
            if not summary.get("dataset"):
                summary = {**summary, "dataset": fallback}
            summaries.append(summary)
    return summaries


def _task_type(summary: dict) -> str:
    """Derive ``"survival"``/``"classification"`` from the metric actually present.

    This is a derived value, NOT a passthrough of a ``task_type`` field --
    verified by reading the ``exp_summary`` construction in all five framework
    runners (``clam``/``nnmil``/``dtfd``/``abmil``/``titan``): none of them
    write ``task_type`` into ``summary.json``. Survival summaries' ``test``
    block only ever carries ``c_index`` (see ``clam/survival_train.py`` etc.,
    identically across arms); classification summaries never do. That makes
    the presence of ``c_index`` an unambiguous stand-in with no schema change
    and no defaulting.
    """
    return "survival" if "c_index" in (summary.get("test") or {}) else "classification"


def summaries_to_frame(summaries: list[dict]) -> pd.DataFrame:
    """One row per experiment.

    Reuses :func:`autobench.pipeline.orchestrator.aggregate_results` for the
    core columns (dataset/framework/strategy/task/encoder/model_type/
    survival_loss/embed_dim/n_folds/seed plus flattened
    ``test_*``/``val_*`` mean/std/ci_low/ci_high) and adds two columns on top
    that callers of this module need but the shared aggregator does not emit:

      - ``experiment_id`` -- absent from ``aggregate_results``' output; the
        natural join key back to :func:`per_fold_frame`. NOTE it does not
        include ``dataset`` (see ``ExperimentConfig.experiment_id``), so a
        key unique across cohorts is ``(dataset, experiment_id)``, not
        ``experiment_id`` alone.
      - ``task_type`` -- see :func:`_task_type`.

    Positional zip with ``summaries`` is safe here because
    ``aggregate_results`` emits exactly one row per input summary, in order,
    with no filtering.
    """
    frame = aggregate_results(summaries)
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["experiment_id"] = [s.get("experiment_id") for s in summaries]
    frame["task_type"] = [_task_type(s) for s in summaries]
    return frame


#: Explicit column order for :func:`per_fold_frame`, kept stable even when the
#: input list is empty so callers can rely on ``.columns`` without a
#: not-empty guard.
_PER_FOLD_COLUMNS = [
    "dataset", "framework", "strategy", "task", "encoder", "model_type",
    "survival_loss", "seed", "experiment_id", "split", "fold", "metric", "value",
]


def per_fold_frame(summaries: list[dict]) -> pd.DataFrame:
    """Long format: one row per (experiment, split, fold, metric, value).

    ``per_fold_test`` / ``per_fold_val`` are lists of flat ``{metric: value}``
    dicts -- the exact input shape ``compute_confidence_intervals`` expects
    (see ``clam/runner.py``, identically ``nnmil``/``dtfd``/``abmil``/
    ``titan``). They exist in every ``summary.json`` but are dropped by every
    roll-up: ``aggregate_results`` (and ``results.aggregate_cross_framework``)
    only keep the fold-pooled mean/std/ci_low/ci_high. This is the source
    data error bars and any fold-level variance decomposition need.
    """
    rows: list[dict] = []
    for s in summaries:
        base = {
            "dataset": s.get("dataset", ""),
            "framework": s.get("framework", "clam"),
            "strategy": s.get("strategy"),
            "task": s.get("task"),
            "encoder": s.get("encoder"),
            "model_type": s.get("model_type"),
            "survival_loss": s.get("survival_loss"),
            "seed": s.get("seed"),
            "experiment_id": s.get("experiment_id"),
        }
        for split_name, key in (("test", "per_fold_test"), ("val", "per_fold_val")):
            for fold_idx, fold_metrics in enumerate(s.get(key) or []):
                if not isinstance(fold_metrics, dict):
                    continue
                for metric_name, value in fold_metrics.items():
                    rows.append({
                        **base,
                        "split": split_name,
                        "fold": fold_idx,
                        "metric": metric_name,
                        "value": value,
                    })
    return pd.DataFrame(rows, columns=_PER_FOLD_COLUMNS)
