#!/usr/bin/env python3
"""Apples-to-apples aggregation: our goldmark_exact results vs GOLDMARK authoritative.

Walks the goldmark_exact result tree, aggregates mean-per-split AUROC per
(cohort, task, encoder, model, arm) over the 5 folds (ONLY cells with all 5
folds — partial cells are reported but excluded from the comparison), maps our
encoders to GOLDMARK's, and joins against goldmark_authoritative.csv's
``mean_per_split`` (GOLDMARK's published metric: mean of 5 per-split val==test
AUROCs at the best-val-AUC epoch).

Our **goldmark arm** uses GOLDMARK's exact training protocol + the parity split
(val==test), so our goldmark-arm mean test AUROC is the protocol-matched analog
of GOLDMARK's mean_per_split. The **our arm** (native trainer, same split) is
shown for reference. clam_sb (instance-loss OFF) is the GMA-proxy = the closest
architectural match to GOLDMARK's single gated-attention head.

Usage:
    python aggregate_goldmark_compare.py \
        [--root /scratch/yinshuol/autoMIL/goldmark_exact] \
        [--gm   /scratch/yinshuol/autoMIL/goldmark-portal/goldmark_authoritative.csv] \
        [--out  /scratch/yinshuol/autoMIL/goldmark_exact/COMPARISON.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict

# our -> GOLDMARK encoder key
ENC_MAP = {"hoptimus1": "h-optimus-0", "uni_v2": "uni", "virchow2": "virchow2"}
# our cohort dir -> GOLDMARK cohort
COHORT_MAP = {"luad": "LUAD", "lgg": "LGG", "coad": "COAD"}
# our task -> GOLDMARK target (OncoKB gene symbol)
TASK_MAP = {"egfr": "EGFR", "kras": "KRAS", "idh1": "IDH1", "braf": "BRAF"}

FOLD_RE = re.compile(
    r"/goldmark_exact/(?P<cohort>[^/]+)/(?P<arm>[^/]+)/benchmark/results/"
    r"(?P<fw>[^/]+)/(?P<strat>[^/]+)/(?P<task>[^/]+)/(?P<enc>[^/]+)/"
    r"(?P<model>[^/]+)/fold_(?P<fold>\d+)/metrics.json"
)


def _auc(metrics: dict) -> float | None:
    v = metrics.get("auc_roc")
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


def collect(root: str) -> dict:
    """cell key -> {fold: (test_auc, val_auc)}"""
    cells: dict = defaultdict(dict)
    for dirpath, _dirs, files in os.walk(root):
        if "metrics.json" not in files:
            continue
        path = os.path.join(dirpath, "metrics.json")
        m = FOLD_RE.search(path)
        if not m:
            continue
        g = m.groupdict()
        try:
            d = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        ta = _auc(d.get("test_metrics", {}))
        va = _auc(d.get("val_metrics", {}))
        if ta is None:
            continue
        key = (g["cohort"], g["arm"], g["task"], g["enc"], g["model"])
        cells[key][int(g["fold"])] = (ta, va)
    return cells


def load_goldmark(gm_csv: str) -> dict:
    """(COHORT, TARGET, gm_encoder) -> mean_per_split float"""
    out = {}
    for r in csv.DictReader(open(gm_csv)):
        try:
            out[(r["cohort"], r["target"], r["encoder"])] = float(r["mean_per_split"])
        except (ValueError, KeyError):
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/yinshuol/autoMIL/goldmark_exact")
    ap.add_argument("--gm", default="/scratch/yinshuol/autoMIL/goldmark-portal/goldmark_authoritative.csv")
    ap.add_argument("--out", default="/scratch/yinshuol/autoMIL/goldmark_exact/COMPARISON.csv")
    ap.add_argument("--require_folds", type=int, default=5)
    args = ap.parse_args()

    cells = collect(args.root)
    gm = load_goldmark(args.gm)

    rows = []
    partial = []
    for key, folds in sorted(cells.items()):
        cohort, arm, task, enc, model = key
        n = len(folds)
        test_aucs = [folds[f][0] for f in sorted(folds)]
        val_aucs = [folds[f][1] for f in sorted(folds) if folds[f][1] is not None]
        our_mean = sum(test_aucs) / n
        if n < args.require_folds:
            partial.append((key, n, our_mean))
            continue
        gm_enc = ENC_MAP.get(enc)
        gm_cohort = COHORT_MAP.get(cohort)
        gm_target = TASK_MAP.get(task)
        gm_mean = gm.get((gm_cohort, gm_target, gm_enc)) if gm_enc else None
        delta = (our_mean - gm_mean) if gm_mean is not None else None
        rows.append({
            "cohort": cohort, "task": task, "encoder": enc, "model": model, "arm": arm,
            "n_folds": n,
            "our_mean_per_split": round(our_mean, 4),
            "our_per_split": ";".join(f"{a:.4f}" for a in test_aucs),
            "gm_encoder": gm_enc or "", "gm_mean_per_split": gm_mean if gm_mean is not None else "",
            "delta_ours_minus_gm": round(delta, 4) if delta is not None else "",
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fieldnames = ["cohort", "task", "encoder", "model", "arm", "n_folds",
                  "our_mean_per_split", "gm_mean_per_split", "delta_ours_minus_gm",
                  "gm_encoder", "our_per_split"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # ---- console summary ----
    print(f"Complete cells (5 folds): {len(rows)}   Partial (excluded): {len(partial)}")
    print(f"Wrote: {args.out}\n")
    hdr = f"{'cohort':5} {'task':5} {'encoder':10} {'model':10} {'arm':8} {'ours':>7} {'GMARK':>7} {'Δ':>8}"
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: (x["arm"], x["cohort"], x["task"], x["model"], x["encoder"])):
        gmv = f"{r['gm_mean_per_split']:.4f}" if isinstance(r["gm_mean_per_split"], float) else "  -  "
        dv = f"{r['delta_ours_minus_gm']:+.4f}" if isinstance(r["delta_ours_minus_gm"], float) else "   -   "
        print(f"{r['cohort']:5} {r['task']:5} {r['encoder']:10} {r['model']:10} {r['arm']:8} "
              f"{r['our_mean_per_split']:7.4f} {gmv:>7} {dv:>8}")

    # headline: goldmark-arm clam_sb (GMA-proxy) vs GOLDMARK
    prim = [r for r in rows if r["arm"] == "goldmark" and r["model"] == "clam_sb"
            and isinstance(r["delta_ours_minus_gm"], float)]
    if prim:
        md = sum(r["delta_ours_minus_gm"] for r in prim) / len(prim)
        print(f"\nPRIMARY (goldmark-arm clam_sb=GMA-proxy vs GOLDMARK, n={len(prim)}): "
              f"mean Δ = {md:+.4f}  ({'OURS AHEAD' if md > 0 else 'GOLDMARK AHEAD'})")
    if partial:
        print(f"\nPartial cells excluded ({len(partial)}):")
        for key, n, m in partial:
            print(f"  {'/'.join(key)}: {n}/5 folds (running mean {m:.4f})")


if __name__ == "__main__":
    main()
