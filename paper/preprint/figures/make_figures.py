#!/usr/bin/env python3
"""Generate REAL-data figures for the autoMIL preprint (FIG-0).

Reads the collector's output (``autobench.pipeline.collect``, via
``benchmarks/scripts/collect_results.py``) and produces the two figures from
``EXPERIMENT_GRID.md`` Sec.4 that are ready to be built from real data today:

  fig1_leaderboard_heatmap  -- dataset x (aggregator, encoder) test AUC
  fig4_survival_cindex      -- dataset x arm OS c-index, with per-fold error bars

Fig 2 (encoder-vs-aggregator variance) and Fig 3 (recipe-search effect) are
OUT OF SCOPE here: per ``READINESS-2026-07-28.md`` (settled 2026-07-28 by Leo),
the "encoder >> aggregator" claim is DROPPED, and Fig 3 needs the agentic-search
layer this script does not touch. Fig 5 (search trajectory) reads
``graph.json``, not the results CSVs this script consumes. See
``paper/preprint/READINESS-2026-07-28.md`` Sec.2.2 for the per-figure verdicts.

Design constraints from review (PRELAUNCH_REVIEW.md O3), both enforced below:
  - Fig 1 must NOT place binary AUC (2-class cohorts) and macro-OvR AUC
    (3-class cohorts) on one colour scale, and must never print a
    cross-dataset column mean. This script plots a WITHIN-DATASET centred
    delta instead of raw AUC, and never emits a summary row/column.
  - Fig 4 must not "pre-draw TITAN as the slide-level winner": arms are drawn
    in a fixed, score-independent (alphabetical) order with one neutral,
    non-highlighted colour cycle.

If the input data cannot support a figure (a required column is missing, or
no rows of the needed task_type exist), the responsible figure is SKIPPED
with a message naming exactly what is missing -- never drawn from partial
data. ``make_mock_figures.py`` remains the layout-only reference until every
figure in the plan has a real producer.

Usage
-----
python paper/preprint/figures/make_figures.py \
    --results paper/preprint/figures/results.csv \
    --per-fold paper/preprint/figures/per_fold.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import figstyle  # noqa: E402
from roster import RosterError, filter_roster, format_report  # noqa: E402

DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "mock")

# Fig 1 type sizes. Named rather than inline because they are set relative to
# the heatmap's cell geometry (see fig1_leaderboard_heatmap) -- changing one
# without the other is what made the first pass illegible at page width.
FIG1_CELL_PT = 13      # the +/-0.000 delta printed in each cell
FIG1_TICK_PT = 12.5    # (aggregator, encoder) and dataset tick labels
FIG1_LABEL_PT = 12     # colourbar label and its ticks
FIG1_TITLE_PT = 14

# All figure text is Times New Roman -- see figstyle.py.
figstyle.apply(**{
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
})


class FigureDataError(RuntimeError):
    """A figure cannot be produced from the given data.

    Raised with a message naming exactly which column or subset is missing.
    Caught by :func:`main`, which reports it and moves on to the next figure
    rather than emitting a plot from partial data.
    """


def _require_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise FigureDataError(
            f"{context}: missing required column(s) {missing} "
            f"(have: {sorted(df.columns)})"
        )


# ---------------------------------------------------------------------------
# FIG 1 -- classification leaderboard heatmap
# ---------------------------------------------------------------------------


def fig1_leaderboard_heatmap(results_df: pd.DataFrame, out_dir: str) -> str:
    """Dataset x (aggregator, encoder) test AUC, within-dataset centred delta.

    PRELAUNCH_REVIEW O3: binary AUC (2-class cohorts) and macro-OvR AUC
    (3-class cohorts) are different quantities and must not share one colour
    scale, and a cross-dataset column mean would silently average across that
    boundary. Both are respected by never computing a raw cross-dataset
    comparison at all: each dataset (row) is centred on its OWN mean before
    the colour scale is applied, and that row mean is shown in the y-tick
    label (not folded into a "Mean" row/column).

    "aggregator" = ``model_type`` (matches the paper's terminology; nnMIL's
    aggregator in this roster is ``simple_mil``, so aggregator != framework).
    Missing (aggregator, encoder) cells for a dataset (e.g. an unfinished
    grid) are drawn as an explicit "n/a" cell, never defaulted to 0.
    """
    _require_columns(
        results_df,
        ["dataset", "task_type", "model_type", "encoder", "test_auc_roc_mean"],
        "fig1_leaderboard_heatmap",
    )
    cls_df = results_df[results_df["task_type"] == "classification"]
    if cls_df.empty:
        raise FigureDataError(
            "fig1_leaderboard_heatmap: --results has no task_type == "
            "'classification' rows -- nothing to plot"
        )

    # aggfunc="mean": collapses duplicate seeds for the same (dataset,
    # aggregator, encoder) cell into one point estimate. Fig 1 has no error
    # bars (see EXPERIMENT_GRID.md Sec.4); per-fold variance belongs to Fig 4
    # and any future variance-decomposition figure.
    pivot = cls_df.pivot_table(
        index="dataset", columns=["model_type", "encoder"],
        values="test_auc_roc_mean", aggfunc="mean",
    ).sort_index(axis=1)
    if pivot.empty:
        raise FigureDataError(
            "fig1_leaderboard_heatmap: pivot on dataset x (model_type, "
            "encoder) produced no cells"
        )

    row_mean = pivot.mean(axis=1, skipna=True)
    delta = pivot.sub(row_mean, axis=0)
    delta_arr = delta.to_numpy()
    finite = delta_arr[np.isfinite(delta_arr)]
    vmax = max(float(np.max(np.abs(finite))), 0.01) if finite.size else 0.01

    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad(color="#dcdcdc")
    masked = np.ma.masked_invalid(delta_arr)

    n_rows, n_cols = pivot.shape

    # Figure geometry and type sizes are tied together deliberately. The figure
    # is placed at page width in the manuscript, so what matters is the ratio of
    # type size to figure width, not the absolute point size: at the previous
    # 1.05 in/column the 13-column grid spanned ~15.6 in, and scaling that down
    # to a ~7 in text block shrank 8 pt type to an unreadable ~3.6 pt effective.
    # Narrower cells + larger type keeps the same information legible in print.
    col_in, row_in = 0.75, 1.0
    fig, ax = plt.subplots(
        figsize=(max(6.0, col_in * n_cols + 3.0), row_in * n_rows + 2.9)
    )
    im = ax.imshow(masked, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        [f"{agg}\n{enc}" for agg, enc in pivot.columns],
        rotation=45, ha="right", fontsize=FIG1_TICK_PT,
    )
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(
        [f"{ds}\n(row mean {row_mean[ds]:.2f})" for ds in pivot.index],
        fontsize=FIG1_TICK_PT,
    )
    ax.tick_params(length=2, pad=2)

    raw_arr = pivot.to_numpy()
    for i in range(n_rows):
        for j in range(n_cols):
            if not np.isfinite(raw_arr[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center",
                        fontsize=FIG1_CELL_PT, color="#777")
                continue
            d = delta_arr[i, j]
            color = "white" if abs(d) > vmax * 0.6 else "#222"
            ax.text(j, i, f"{d:+.3f}", ha="center", va="center",
                    fontsize=FIG1_CELL_PT, color=color)

    ax.set_title(
        "Classification leaderboard — test AUC, within-dataset centred ΔAUC\n"
        "(binary and macro-OvR AUC are different quantities; never averaged across datasets)",
        fontsize=FIG1_TITLE_PT, pad=12,
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Δ test AUC vs. that dataset's own row mean", fontsize=FIG1_LABEL_PT)
    cb.ax.tick_params(labelsize=FIG1_LABEL_PT)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "fig1_leaderboard_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# FIG 4 -- survival c-index
# ---------------------------------------------------------------------------


def fig4_survival_cindex(results_df: pd.DataFrame, per_fold_df: pd.DataFrame, out_dir: str) -> str:
    """Grouped bars: OS c-index per dataset x arm, error bars from per-fold data.

    "arm" = ``model_type`` x ``survival_loss`` (e.g. ``clam_mb·nllsurv``),
    pooling every constituent per-fold test c-index across encoder AND fold
    into one mean +/- sd -- this collapses the encoder axis the same way Fig 1
    collapses nothing (Fig 1 keeps encoder as its own column); it is a
    documented choice so the chart stays one bar per (model, loss) per
    dataset, not a default -- no per-fold value is dropped, each arm's bar and
    error bar are computed from the SAME pooled set.

    PRELAUNCH_REVIEW O3: "delete the figure-plan line that pre-draws TITAN as
    the slide-level winner". Arms are drawn in a fixed alphabetical order
    (independent of score) with a single neutral colour cycle -- nothing about
    ordering, colour, or annotation implies a winner.
    """
    _require_columns(results_df, ["dataset", "task_type"], "fig4_survival_cindex (--results)")
    if (results_df["task_type"] == "survival").sum() == 0:
        raise FigureDataError(
            "fig4_survival_cindex: --results has no task_type == 'survival' "
            "rows -- nothing to plot"
        )
    _require_columns(
        per_fold_df,
        ["dataset", "model_type", "survival_loss", "split", "metric", "value"],
        "fig4_survival_cindex (--per-fold)",
    )
    cidx = per_fold_df[
        (per_fold_df["metric"] == "c_index") & (per_fold_df["split"] == "test")
    ].copy()
    if cidx.empty:
        raise FigureDataError(
            "fig4_survival_cindex: --per-fold has no (metric=='c_index', "
            "split=='test') rows -- nothing to plot"
        )

    cidx["arm"] = cidx["model_type"].astype(str) + "·" + cidx["survival_loss"].astype(str)
    stats = (
        cidx.groupby(["dataset", "arm"])["value"]
        .agg(mean="mean", std=lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0)
        .reset_index()
    )

    datasets = sorted(stats["dataset"].unique())
    arms = sorted(stats["arm"].unique())  # score-independent order -- see docstring
    cmap = mpl.colormaps["tab10"]

    fig, ax = plt.subplots(figsize=(max(7.0, 1.7 * len(datasets) + 2), 4.4))
    x = np.arange(len(datasets))
    width = 0.8 / max(len(arms), 1)

    for k, arm in enumerate(arms):
        means, errs = [], []
        for d in datasets:
            row = stats[(stats["dataset"] == d) & (stats["arm"] == arm)]
            if row.empty:
                means.append(np.nan)
                errs.append(np.nan)
            else:
                means.append(float(row["mean"].iloc[0]))
                errs.append(float(row["std"].iloc[0]))
        offset = (k - (len(arms) - 1) / 2) * width
        ax.bar(
            x + offset, means, width, yerr=errs, capsize=2, label=arm,
            color=cmap(k % 10), edgecolor="#333", linewidth=0.5,
        )

    ax.axhline(0.5, color="#b00020", lw=1, ls="--", label="random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=8)
    ax.set_ylabel("test c-index (pooled per-fold mean ± sd)")
    ax.set_title("Survival OS c-index by dataset × arm", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, ncol=min(4, len(arms) + 1), loc="upper center",
              bbox_to_anchor=(0.5, -0.20), frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "fig4_survival_cindex.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_csv(path: str, context: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise FigureDataError(f"{context}: could not read {path}: {exc}") from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True,
                   help="Wide per-experiment CSV (autobench.pipeline.collect.summaries_to_frame)")
    p.add_argument("--per-fold", dest="per_fold", required=True,
                   help="Long per-fold CSV (autobench.pipeline.collect.per_fold_frame)")
    p.add_argument("--out-dir", dest="out_dir", default=DEFAULT_OUT_DIR,
                   help="Directory to write PNGs into (default: figures/mock/, "
                        "matching make_mock_figures.py -- real figures overwrite "
                        "the mock of the same name once real data exists)")
    p.add_argument("--no-roster-filter", dest="roster_filter", action="store_false",
                   help="Plot every collected experiment instead of the 130 baseline "
                        "roster cells. NOT recommended: unfiltered, TCGA-LUAD "
                        "contributes 105 experiments against a roster of 26, and 35 "
                        "partial cox runs join the survival arms (see roster.py)")
    p.add_argument("--allow-incomplete-roster", dest="strict_roster",
                   action="store_false",
                   help="Plot even if a cohort does not have all 26 roster cells "
                        "(default: fail loudly rather than draw a partial grid)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_df = _load_csv(args.results, "--results")
    per_fold_df = _load_csv(args.per_fold, "--per-fold")

    if args.roster_filter:
        try:
            results_df, filtered_per_fold, report = filter_roster(
                results_df, per_fold_df, strict=args.strict_roster,
            )
        except RosterError as exc:
            raise SystemExit(f"[make_figures] {exc}") from exc
        per_fold_df = filtered_per_fold if filtered_per_fold is not None else per_fold_df
        print(format_report(report))
    else:
        print("[make_figures] WARNING: --no-roster-filter -- plotting ALL "
              f"{len(results_df)} collected experiments, including off-roster work")

    jobs = [
        ("fig1_leaderboard_heatmap", lambda: fig1_leaderboard_heatmap(results_df, args.out_dir)),
        ("fig4_survival_cindex", lambda: fig4_survival_cindex(results_df, per_fold_df, args.out_dir)),
    ]

    failures: list[str] = []
    for name, fn in jobs:
        try:
            path = fn()
        except FigureDataError as exc:
            print(f"[make_figures] SKIPPED {name}: {exc}", file=sys.stderr)
            failures.append(name)
        else:
            print(f"[make_figures] wrote {name} -> {path}")

    if failures:
        raise SystemExit(
            f"[make_figures] {len(failures)}/{len(jobs)} figure(s) not produced: "
            f"{', '.join(failures)} (see messages above for the missing column/data)"
        )


if __name__ == "__main__":
    main()
