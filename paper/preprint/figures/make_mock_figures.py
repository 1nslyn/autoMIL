#!/usr/bin/env python3
"""Generate ILLUSTRATIVE MOCK figures for the autoMIL preprint.

Every number here is fabricated to show *figure layout and intent*, anchored
only loosely to the May-baseline AUC ranges. NONE of these are real results.
Regenerate: `python paper/preprint/figures/make_mock_figures.py`.

Grid this illustrates (verified from benchmarks/datasets/{tcga,cptac}/*.yaml +
pipeline/config.generate_all_experiments):
  5 datasets (3 TCGA + 2 CPTAC) x (1 classification + 1 survival[cox,nllsurv]) x
  {clam_mb, simple_mil, abmil, dtfd_mil} x {uni_v2, virchow2, hoptimus1}
  + TITAN slide-encoder arm  ->  33 experiments/dataset, 165 total, 825 fold-trainings.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm  # noqa: F401
import matplotlib as mpl

rng = np.random.default_rng(20260714)
OUT = os.path.join(os.path.dirname(__file__), "mock")
os.makedirs(OUT, exist_ok=True)

# ---- shared style -----------------------------------------------------------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

DATASETS = ["LUAD", "LGG", "GBM", "PDAC", "HNSC"]
TASK = {"LUAD": "KRAS", "LGG": "IDH1", "GBM": "TP53", "PDAC": "immune", "HNSC": "grade"}
ROW_LABEL = [f"{d}\n{TASK[d]}" for d in DATASETS]

AGGS = ["clam_mb", "simple_mil", "abmil", "dtfd_mil"]
ENCS = ["uni_v2", "virchow2", "hoptimus1"]

# rough per-dataset classification "difficulty" centers (MOCK, anchored to prior AUC ranges)
BASE_AUC = {"LUAD": 0.68, "LGG": 0.85, "GBM": 0.70, "PDAC": 0.65, "HNSC": 0.72}
ENC_OFFSET = {"uni_v2": 0.015, "virchow2": 0.00, "hoptimus1": 0.025}   # encoder axis = big
AGG_OFFSET = {"clam_mb": 0.012, "simple_mil": 0.0, "abmil": 0.004, "dtfd_mil": -0.006}  # aggregator axis = small
TITAN_OFFSET = {"LUAD": 0.03, "LGG": 0.02, "GBM": 0.04, "PDAC": 0.03, "HNSC": 0.025}

MOCK_TAG = "⚠ MOCK DATA — not real results"


def _mock_note(fig):
    # top-left corner — clear of centered titles and multi-line x-axis labels
    fig.text(0.008, 0.988, MOCK_TAG, ha="left", va="top",
             fontsize=7.5, style="italic", color="#b00020")


# ---------------------------------------------------------------------------
# FIG 1 — Main classification leaderboard heatmap (datasets x agg×enc + TITAN)
# ---------------------------------------------------------------------------
def fig1_heatmap():
    cols = [(a, e) for a in AGGS for e in ENCS] + [("TITAN", "titan")]
    M = np.zeros((len(DATASETS), len(cols)))
    for i, d in enumerate(DATASETS):
        for j, (a, e) in enumerate(cols):
            if a == "TITAN":
                v = BASE_AUC[d] + TITAN_OFFSET[d] + rng.normal(0, 0.006)
            else:
                v = BASE_AUC[d] + ENC_OFFSET[e] + AGG_OFFSET[a] + rng.normal(0, 0.008)
            M[i, j] = np.clip(v, 0.5, 0.95)

    fig, ax = plt.subplots(figsize=(11, 3.6))
    im = ax.imshow(M, cmap="YlGnBu", vmin=0.5, vmax=0.9, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([e if a != "TITAN" else "TITAN" for a, e in cols], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(DATASETS)))
    ax.set_yticklabels(ROW_LABEL, fontsize=8)
    for i in range(len(DATASETS)):
        for j in range(len(cols)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if M[i, j] > 0.74 else "#222")
    # aggregator group brackets
    for k, a in enumerate(AGGS):
        x0 = k * 3 - 0.5
        ax.plot([x0, x0], [-0.6, len(DATASETS) - 0.5], color="#888", lw=0.6, clip_on=False)
        ax.text(x0 + 1.5, -0.85, a, ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.text(12, -0.85, "TITAN", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#5a2a82")
    ax.set_title("Fig 1 (MOCK) · Classification leaderboard — test AUC across 4 aggregators × 3 encoders + TITAN", pad=26)
    cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01)
    cb.set_label("test AUC", fontsize=8)
    _mock_note(fig)
    p = os.path.join(OUT, "fig1_leaderboard_heatmap.png")
    fig.savefig(p); plt.close(fig); return p


# ---------------------------------------------------------------------------
# FIG 2 — Encoder-vs-aggregator variance (THE HEADLINE)
# ---------------------------------------------------------------------------
def fig2_variance():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1.25]})

    # Panel A: mixed-effects variance decomposition (encoder dominates aggregator)
    comps = ["Dataset\n/ task", "Encoder", "Aggregator", "Recipe\n(pre-search)", "Residual"]
    frac = np.array([0.34, 0.41, 0.10, 0.09, 0.06])
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#bbbbbb"]
    axA.bar(comps, frac * 100, color=colors, edgecolor="#333", linewidth=0.6)
    for i, f in enumerate(frac):
        axA.text(i, f * 100 + 1, f"{f*100:.0f}%", ha="center", fontsize=8, fontweight="bold")
    axA.set_ylabel("% of test-AUC variance explained")
    axA.set_ylim(0, 50)
    axA.set_title("A · Variance decomposition\n(encoder ≫ aggregator)")
    axA.axvspan(0.5, 1.5, color="#dd8452", alpha=0.08)
    axA.axvspan(1.5, 2.5, color="#55a868", alpha=0.08)

    # Panel B: per-dataset AUC spread — vary encoder (agg fixed) vs vary aggregator (enc fixed)
    x = np.arange(len(DATASETS))
    enc_spread, agg_spread = [], []
    for d in DATASETS:
        enc_vals = [BASE_AUC[d] + ENC_OFFSET[e] + rng.normal(0, 0.006) for e in ENCS]
        agg_vals = [BASE_AUC[d] + AGG_OFFSET[a] + rng.normal(0, 0.006) for a in AGGS]
        enc_spread.append(enc_vals); agg_spread.append(agg_vals)
    enc_spread = np.array(enc_spread); agg_spread = np.array(agg_spread)
    w = 0.32
    for i in range(len(DATASETS)):
        axB.vlines(x[i] - w/1.4, enc_spread[i].min(), enc_spread[i].max(), color="#dd8452", lw=6, alpha=0.35)
        axB.scatter([x[i] - w/1.4]*len(ENCS), enc_spread[i], color="#dd8452", s=22, zorder=3, label="vary encoder" if i == 0 else None)
        axB.vlines(x[i] + w/1.4, agg_spread[i].min(), agg_spread[i].max(), color="#55a868", lw=6, alpha=0.35)
        axB.scatter([x[i] + w/1.4]*len(AGGS), agg_spread[i], color="#55a868", s=22, zorder=3, label="vary aggregator" if i == 0 else None)
    axB.set_xticks(x); axB.set_xticklabels([f"{d}\n{TASK[d]}" for d in DATASETS], fontsize=7.5)
    axB.set_ylabel("test AUC")
    axB.set_title("B · Per-dataset spread\nswapping encoder vs swapping aggregator")
    axB.legend(fontsize=7.5, loc="lower right", frameon=False)
    fig.suptitle("Fig 2 (MOCK) · Encoder choice moves AUC more than aggregator", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    _mock_note(fig)
    p = os.path.join(OUT, "fig2_encoder_vs_aggregator_variance.png")
    fig.savefig(p); plt.close(fig); return p


# ---------------------------------------------------------------------------
# FIG 3 — autoMIL recipe-search effect: ranking flips + AUC lift (FRAMEWORK CONTRIBUTION)
# ---------------------------------------------------------------------------
def fig3_recipe_search():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})

    # Panel A: bump chart — aggregator ranking before vs after equal-effort search
    default_auc = {"clam_mb": 0.702, "simple_mil": 0.690, "abmil": 0.694, "dtfd_mil": 0.684}
    searched_auc = {"clam_mb": 0.716, "simple_mil": 0.731, "abmil": 0.727, "dtfd_mil": 0.709}
    order_def = sorted(AGGS, key=lambda a: -default_auc[a])
    order_srch = sorted(AGGS, key=lambda a: -searched_auc[a])
    cmap = {"clam_mb": "#4c72b0", "simple_mil": "#dd8452", "abmil": "#55a868", "dtfd_mil": "#c44e52"}
    for a in AGGS:
        r0 = order_def.index(a); r1 = order_srch.index(a)
        axA.plot([0, 1], [r0, r1], "-o", color=cmap[a], lw=2, markersize=7)
        axA.text(-0.03, r0, f"{a}  {default_auc[a]:.3f}", ha="right", va="center", fontsize=8, color=cmap[a])
        axA.text(1.03, r1, f"{searched_auc[a]:.3f}  {a}", ha="left", va="center", fontsize=8, color=cmap[a])
    axA.set_xlim(-0.55, 1.6); axA.set_ylim(3.5, -0.5)
    axA.set_xticks([0, 1]); axA.set_xticklabels(["default recipe", "autoMIL-searched\n(equal effort)"], fontsize=8)
    axA.set_yticks(range(4)); axA.set_yticklabels([f"rank {i+1}" for i in range(4)], fontsize=8)
    axA.set_title("A · Aggregator ranking flips after\nequal-effort recipe search (RQ1)")
    for s in ["top", "right"]:
        axA.spines[s].set_visible(False)

    # Panel B: per-cell dumbbell — default -> searched composite lift (feasibility anchors)
    cells = ["CCRCC hi-grade", "ovarian HRD", "LUAD-KRAS", "HNSC-grade", "LGG-IDH1"]
    d0 = np.array([0.744, 0.814, 0.685, 0.742, 0.848])
    d1 = np.array([0.807, 0.851, 0.712, 0.771, 0.861])
    y = np.arange(len(cells))[::-1]
    for i in range(len(cells)):
        axB.plot([d0[i], d1[i]], [y[i], y[i]], color="#cccccc", lw=2, zorder=1)
        axB.scatter(d0[i], y[i], color="#999999", s=45, zorder=2, label="default" if i == 0 else None)
        axB.scatter(d1[i], y[i], color="#2a7", s=55, zorder=3, label="autoMIL" if i == 0 else None)
        axB.text(d1[i] + 0.004, y[i], f"+{(d1[i]-d0[i]):.3f}", va="center", fontsize=7.5, color="#2a7")
    axB.set_yticks(y); axB.set_yticklabels(cells, fontsize=8)
    axB.set_xlabel("validation composite")
    axB.set_xlim(0.70, 0.90)
    axB.set_title("B · Recipe-search lift per cell\n(CCRCC/HRD = real feasibility anchors)")
    axB.legend(fontsize=8, loc="lower left", frameon=False)
    for s in ["top", "right"]:
        axB.spines[s].set_visible(False)
    fig.suptitle("Fig 3 (MOCK) · autoMIL recipe search — rankings flip & composite lifts", fontsize=9.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    _mock_note(fig)
    p = os.path.join(OUT, "fig3_recipe_search_effect.png")
    fig.savefig(p); plt.close(fig); return p


# ---------------------------------------------------------------------------
# FIG 4 — Survival c-index (second task axis)
# ---------------------------------------------------------------------------
def fig4_survival():
    arms = ["clam_mb·nll", "simple_mil·cox", "abmil·cox", "dtfd_mil·nll", "TITAN"]
    arm_c = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#5a2a82"]
    base_c = {"LUAD": 0.60, "LGG": 0.64, "GBM": 0.63, "PDAC": 0.66, "HNSC": 0.61}
    arm_off = [0.00, 0.01, 0.008, -0.005, 0.03]  # TITAN best (Frontiers-style)
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    x = np.arange(len(DATASETS)); w = 0.15
    for k, arm in enumerate(arms):
        vals = [base_c[d] + arm_off[k] + rng.normal(0, 0.006) for d in DATASETS]
        err = [0.03 + rng.uniform(0, 0.01) for _ in DATASETS]
        ax.bar(x + (k - 2) * w, vals, w, yerr=err, capsize=2, color=arm_c[k],
               edgecolor="#333", linewidth=0.5, label=arm, error_kw={"lw": 0.7})
    ax.axhline(0.5, color="#b00020", lw=1, ls="--", label="random (0.5)")
    ax.set_xticks(x); ax.set_xticklabels([f"{d}\n{TASK[d]}\n{deaths}✝" for d, deaths in
                                          zip(DATASETS, [167, 115, 72, 81, 204])], fontsize=7.5)
    ax.set_ylabel("OS concordance index"); ax.set_ylim(0.45, 0.72)
    ax.set_title("Fig 4 (MOCK) · Survival (overall survival) c-index — second task axis, per dataset × model")
    ax.legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    _mock_note(fig)
    p = os.path.join(OUT, "fig4_survival_cindex.png")
    fig.savefig(p); plt.close(fig); return p


# ---------------------------------------------------------------------------
# FIG 5 — autoMIL search trajectory (how the agent explores)
# ---------------------------------------------------------------------------
def fig5_trajectory():
    n = 90
    idx = np.arange(1, n + 1)
    # candidate val composite: noisy, occasional improvements; running best staircases up
    cand = 0.744 + np.cumsum(rng.normal(0.0, 0.006, n)) * 0.15 + rng.normal(0, 0.02, n)
    cand = np.clip(cand, 0.66, 0.83)
    # inject a few strong hits
    for h, v in [(18, 0.781), (41, 0.796), (67, 0.807)]:
        cand[h] = v
    best = np.maximum.accumulate(cand)
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.scatter(idx, cand, s=16, color="#9bb", alpha=0.7, label="candidate recipe (val composite)")
    ax.step(idx, best, where="post", color="#4c72b0", lw=2, label="running best (val)")
    ax.axhline(0.744, color="#999", ls="--", lw=1, label="default recipe (0.744)")
    ax.annotate("frozen recipe → test\n(0.807 val)", xy=(67, 0.807), xytext=(70, 0.70),
                fontsize=7.5, arrowprops=dict(arrowstyle="->", color="#333"))
    ax.set_xlabel("search iteration (UCB-scored experiment tree node)")
    ax.set_ylabel("validation composite")
    ax.set_ylim(0.66, 0.84)
    ax.set_title("Fig 5 (MOCK) · autoMIL search trajectory for one cell — CCRCC-style, val-only selection")
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    _mock_note(fig)
    p = os.path.join(OUT, "fig5_search_trajectory.png")
    fig.savefig(p); plt.close(fig); return p


if __name__ == "__main__":
    outs = [fig1_heatmap(), fig2_variance(), fig3_recipe_search(), fig4_survival(), fig5_trajectory()]
    print("Wrote:")
    for p in outs:
        print("  ", p)
