"""Cross-framework result aggregation and comparison tables."""

from __future__ import annotations

import json
import os

import pandas as pd

from autobench.pipeline.config import ExperimentConfig


def aggregate_cross_framework(
    benchmark_dir: str,
    experiments: list[ExperimentConfig],
) -> pd.DataFrame:
    """Build a unified DataFrame from all experiment summary.json files.

    Columns: framework, strategy, task, encoder, model_type,
    test_auc_roc_mean, test_auc_roc_ci_low, test_auc_roc_ci_high, ...
    """
    rows: list[dict] = []

    for exp in experiments:
        summary_path = os.path.join(
            benchmark_dir, "results", exp.results_subdir, "summary.json",
        )
        if not os.path.exists(summary_path):
            continue

        with open(summary_path) as f:
            s = json.load(f)

        row = {
            "framework": s.get("framework", exp.framework.value),
            "strategy": s.get("strategy", exp.strategy),
            "task": s["task"],
            "encoder": s["encoder"],
            "model_type": s["model_type"],
            "survival_loss": s.get("survival_loss", getattr(exp, "survival_loss", None)),
            "embed_dim": s["embed_dim"],
            "n_folds": s["n_folds"],
            "seed": s["seed"],
        }
        for split_name in ("test", "val"):
            if split_name not in s:
                continue
            for metric_name, metric_data in s[split_name].items():
                if isinstance(metric_data, dict):
                    for stat in ("mean", "std", "ci_low", "ci_high"):
                        row[f"{split_name}_{metric_name}_{stat}"] = metric_data.get(stat)
        rows.append(row)

    return pd.DataFrame(rows)


def generate_comparison_tables(
    results_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Generate pivot tables: CLAM vs nnMIL for each (task, strategy).

    Returns a dict keyed by ``"{task}_{strategy}"`` with DataFrames
    showing models as rows and encoders as columns. Values are
    ``test_c_index_mean`` for survival results (when present) else
    ``test_auc_roc_mean``.
    """
    tables: dict[str, pd.DataFrame] = {}

    if results_df.empty:
        return tables
    value_col = (
        "test_c_index_mean"
        if "test_c_index_mean" in results_df.columns
        else "test_auc_roc_mean"
    )
    if value_col not in results_df.columns:
        return tables

    for (task, strategy), group in results_df.groupby(["task", "strategy"]):
        key = f"{task}_{strategy}"
        pivot = group.pivot_table(
            index=["framework", "model_type"],
            columns="encoder",
            values=value_col,
            aggfunc="first",
        )
        if not pivot.empty:
            tables[key] = pivot.round(3)

    return tables
