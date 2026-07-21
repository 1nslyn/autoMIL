#!/usr/bin/env python3
"""Render the preprint REAL-data tables as styled figures.

  table1_dataset_stats.png  — 5-cohort dataset characteristics (3 task types)
  table2_grid_breakdown.png — per-dataset experiment grid (33 exps/dataset)

The roster (2026-07-17) spans three classification task types, so table1 stacks
three sub-tables: binary mutation (LUAD/LGG/GBM), 3-class immune subtype (PDAC),
and 3-class tumor grade (HNSC). Counts are from the roster figure; TCGA-LUAD/LGG
mutation counts also match the May tasks/baseline_summary/data/task_sizes.csv
baseline. Grid counts in table2 are VERIFIED by running
autobench.pipeline.config.generate_all_experiments over the 5 YAMLs.

Regenerate: `python paper/preprint/figures/make_dataset_table.py`
"""
from __future__ import annotations
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

OUT = os.path.join(os.path.dirname(__file__), "mock")
os.makedirs(OUT, exist_ok=True)
mpl.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 200, "savefig.bbox": "tight"})

HEADER_BG = "#3a4a63"
STRIPE = "#f4f6f9"
TOTAL_BG = "#e8ecf3"


def _style(tbl, n_body, widths, left_cols, mono_cells, has_total):
    """Apply the shared Notion-style cell formatting to a rendered table."""
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#d0d5dd")
        cell.set_linewidth(0.6)
        cell.set_width(widths[c])
        if r == 0:  # header
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(color="white", fontweight="bold")
            cell.set_height(cell.get_height() * 1.25)
            continue
        is_total = has_total and (r == n_body)
        cell.set_facecolor(TOTAL_BG if is_total else (STRIPE if r % 2 else "white"))
        if is_total:
            cell.set_text_props(fontweight="bold", color="#3a4a63")
        if c in left_cols:
            cell.set_text_props(ha="left")
            cell._text.set_x(0.04)
        if (r, c) in mono_cells:
            cell.set_text_props(fontfamily="DejaVu Sans Mono")


def _table_on_ax(ax, col_labels, body, widths, left_cols, mono_cells, has_total,
                 fontsize, fontscale, subtitle=None):
    """Draw one styled table onto a provided axis (used for stacked sub-tables)."""
    ax.axis("off")
    if subtitle:
        ax.set_title(subtitle, fontsize=9.5, fontweight="bold", loc="left",
                     pad=4, color="#3a4a63")
    tbl = ax.table(cellText=body, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, fontscale)
    _style(tbl, len(body), widths, left_cols, mono_cells, has_total)


def styled_table(out_name, title, col_labels, body, widths, figsize,
                 left_cols=(0,), mono_cells=(), footnote=None, fontscale=2.05,
                 fontsize=9, has_total=True):
    """Render one standalone table figure and save to OUT/out_name."""
    fig, ax = plt.subplots(figsize=figsize)
    _table_on_ax(ax, col_labels, body, widths, left_cols, mono_cells, has_total,
                 fontsize, fontscale)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)
    if footnote:
        fig.text(0.5, -0.02, footnote, ha="center", va="top", fontsize=6.8, color="#555")
    p = os.path.join(OUT, out_name)
    fig.savefig(p)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Table 1 — dataset characteristics (3 stacked sub-tables, one per task type)
# ---------------------------------------------------------------------------
# Binary mutation:  (dataset, gene, n, mutant, wildtype, os_deaths, os_pct)
BINARY = [
    ("TCGA-LUAD", "KRAS", 465, 171, 294, 167, 35.9),
    ("TCGA-LGG",  "IDH1", 491, 382, 109, 115, 23.4),
    ("CPTAC-GBM", "TP53",  99,  32,  67,  72, 72.7),
]
# 3-class immune subtype:  (dataset, task, n, low, medium, high, os_deaths, os_pct)
IMMUNE = [
    ("CPTAC-PDAC", "immune_class", 105, 35, 35, 35, 81, 77.1),
]
# 3-class tumor grade:  (dataset, task, n, g1, g2, g3, os_deaths, os_pct)
GRADE = [
    ("TCGA-HNSC", "tumor grade", 431, 54, 260, 100, 205, 47.6),
]

# 8 columns, shared width profile across all three sub-tables (sums ~1.0).
_WIDTHS = [0.165, 0.155, 0.105, 0.105, 0.115, 0.105, 0.135, 0.115]


def table1():
    bin_cols = ["Dataset", "Cls task", "Patients\n(n)", "Mutant\n(+)", "Wildtype\n(−)",
                "Prevalence", "OS deaths\n(events)", "OS\nprevalence"]
    bin_body = [[ds, gene, f"{n}", f"{mut}", f"{wt}", f"{mut / n * 100:.1f}%",
                 f"{d}", f"{op:.1f}%"]
                for ds, gene, n, mut, wt, d, op in BINARY]

    imm_cols = ["Dataset", "Cls task", "Patients\n(n)", "Low", "Medium", "High",
                "OS deaths\n(events)", "OS\nprevalence"]
    imm_body = [[ds, task, f"{n}", f"{lo}", f"{me}", f"{hi}", f"{d}", f"{op:.1f}%"]
                for ds, task, n, lo, me, hi, d, op in IMMUNE]

    grd_cols = ["Dataset", "Cls task", "Patients\n(n)", "G1", "G2", "G3",
                "OS deaths\n(events)", "OS\nprevalence"]
    grd_body = [[ds, task, f"{n}", f"{g1}", f"{g2}", f"{g3}", f"{d}", f"{op:.1f}%"]
                for ds, task, n, g1, g2, g3, d, op in GRADE]

    fig, axes = plt.subplots(
        3, 1, figsize=(8.6, 5.2),
        gridspec_kw={"height_ratios": [len(BINARY) + 1, len(IMMUNE) + 1, len(GRADE) + 1]},
    )
    fig.subplots_adjust(hspace=0.75, top=0.9, bottom=0.1)
    _table_on_ax(axes[0], bin_cols, bin_body, _WIDTHS, (0, 1), (), False, 7.6, 1.7,
                 subtitle="Binary mutation  (mutant vs. wildtype)")
    _table_on_ax(axes[1], imm_cols, imm_body, _WIDTHS, (0, 1), (), False, 7.6, 1.7,
                 subtitle="3-class immune subtype  (low / medium / high)")
    _table_on_ax(axes[2], grd_cols, grd_body, _WIDTHS, (0, 1), (), False, 7.6, 1.7,
                 subtitle="3-class tumor grade  (G1 / G2 / G3)")
    fig.suptitle("Dataset characteristics — preprint 5-cohort roster (3 TCGA + 2 CPTAC)",
                 fontsize=12, fontweight="bold", y=0.98)
    foot = ("Counts from the roster figure (2026-07-17); TCGA-LUAD/LGG mutation counts match the May "
            "task_sizes.csv baseline.  TCGA-HNSC grade: 54+260+100=414 gradeable of 431 (17 GX/NR "
            "dropped from the grade task; survival uses all 431).\nCPTAC-PDAC immune subtype is a TERTILE "
            "SPLIT of a continuous tumour-immune infiltration score, hence balanced by construction "
            "(35/35/35).\nOS deaths are cohort-level; the usable survival cohort is smaller after the "
            "OS_time filter (see EXPERIMENT_GRID.md 1.1).  OS prevalence = deaths / n;  Prevalence "
            "column = mutant fraction (binary only).")
    fig.text(0.5, 0.02, foot, ha="center", va="top", fontsize=6.6, color="#555")
    p = os.path.join(OUT, "table1_dataset_stats.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Table 2 — per-dataset experiment grid (unchanged: 33 exps/dataset)
# ---------------------------------------------------------------------------
# aggregator, classification count, survival cell text, row total
GRID = [
    ("clam_mb",    "3", "3  (nllsurv only)",  "6"),
    ("simple_mil", "3", "6  (cox + nllsurv)", "9"),
    ("abmil",      "3", "6  (cox + nllsurv)", "9"),
    ("dtfd_mil",   "3", "3  (nllsurv only)",  "6"),
    ("TITAN",      "1", "2  (cox + nllsurv)", "3"),
]


def table2():
    cols = ["Aggregator", "Classification\n(1 task × 3 enc)", "Survival OS\n(× 3 enc)", "Total"]
    body = [list(r) for r in GRID]
    body.append(["per dataset", "13", "20", "33"])
    # monospace the four tile-encoder model names (rows 1–4, col 0); TITAN/per-dataset stay plain
    mono = {(1, 0), (2, 0), (3, 0), (4, 0)}
    foot = ("Grid PER DATASET, verified via generate_all_experiments (pipeline/config.py:296). "
            "Identical for binary and 3-class tasks (class count does not change the grid). "
            "Survival loss\nfan-out is uneven: clam_mb & dtfd_mil run nllsurv only (cox needs a "
            "single-risk output / cross-patient risk set); simple_mil, abmil, TITAN run both.  "
            "×5 datasets ×5-fold  →  165 experiments, 825 fold-trainings.")
    return styled_table(
        "table2_grid_breakdown.png", "Per-dataset experiment grid — 33 experiments / dataset",
        cols, body, widths=[0.24, 0.33, 0.29, 0.14],
        figsize=(6.0, 3.5), left_cols=(0,), mono_cells=mono, footnote=foot, fontsize=8)


if __name__ == "__main__":
    print("Wrote:", table1())
    print("Wrote:", table2())
