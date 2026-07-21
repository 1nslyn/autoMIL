# Pre-launch review — preprint campaign

_Adversarial verification of the pivoted 5-cohort roster before committing GPU
time. Run 2026-07-20/21 by three independent reviewers (code correctness, data
integrity, scientific fit), each with read-only access to the live data on `fir`
so claims were checked against reality rather than against the docs._

**Roster under review:** TCGA-LUAD (KRAS, binary) · TCGA-LGG (IDH1, binary) ·
CPTAC-GBM (TP53, binary) · CPTAC-PDAC (immune subtype, 3-class) · TCGA-HNSC
(tumour grade, 3-class). See [`EXPERIMENT_GRID.md`](EXPERIMENT_GRID.md) §1.

| Review | Verdict |
|---|---|
| Data integrity | **Sound** — every roster claim reproduced exactly |
| Code correctness | **BLOCK** — 2 blocking bugs (both now fixed) |
| Scientific fit | **SERIOUS CONCERNS** — 2 open strategic decisions |

---

## 1. Blocking bugs found — fixed and verified

### B1. Informative censoring in the GDC OS join (the serious one)

`add_os_to_manifest.py` reduced the clinical export with `drop_duplicates`, i.e.
the **first row** per case. `diagnoses.days_to_last_follow_up` is a
*diagnoses*-level field that is frequently blank on a case's first row, so living
patients received `OS_event=0` with `OS_time=NaN` and were then removed by the
survival task's `dropna`.

Because only **Alive** cases lose their time this way, the dropout was entirely
censored — **129 patients across the three TCGA cohorts, and zero deaths.**

| Cohort | Survival cohort before | after | recovered | deaths recovered |
|---|--:|--:|--:|--:|
| TCGA-LUAD | 399 | **453** | +54 | **0** |
| TCGA-LGG | 445 | **488** | +43 | **0** |
| TCGA-HNSC | 397 | **429** | +32 | **0** |

This is textbook informative censoring: it inflated the event rate (LUAD
0.358 → 0.406) and would have biased every Cox / nllsurv fit and c-index on the
TCGA survival arms. It fails **silently** — a plausible wrong number in a
preprint table, not a crash.

Why it survived this long: the script's own reporting was structurally blind to
it. It printed only `OS_event` NaN counts (always 0 here) and never `OS_time`
NaNs (63 on LUAD). The line claiming to report what gets dropped counted zero.

**Fixed:** reduce per column over non-null values (`max` for the two time
fields); report `OS_time` NaNs and how many are censored; drop pre-existing
`OS_event`/`OS_time` so re-runs refresh rather than colliding into
`OS_event_x`/`_y`. CPTAC cohorts were unaffected — their `clinical.tsv` comes
from `fetch_gdc_clinical.py`, which already reduces per field.

> **Any survival number produced before 2026-07-21 must be regenerated, not reused.**

### B2. Stale task-CSV cache reused across a task-type change

`prepare.py` keyed cached task CSVs by **task name only**. The pivot changed `os`
from a Patho-Bench 8-way classification task (quartile×event code) to a survival
task — same name, different contract. CPTAC-PDAC still had a June `os.csv` with
`label` = `Q2_event0`; `prepare_all` would load it under the survival config.

It fails loudly (`missing the stratification column 'status'`) — but only after
GPU allocation. The silent variant was one file away: had the matching
`splits_0.csv` also survived, both branches would have been skipped and training
would have proceeded on stale 8-class splits under a survival task.

**Fixed:** validate a cached CSV's schema against `tdef.task_type` before reuse
and **fail loudly** with the exact purge commands. Deliberately *not* self-healing:
`prepare_all` runs once per **experiment** against the *shared* `benchmark_dir`
(`run_experiment.py`), so under the agentic loop many processes execute it
concurrently. An earlier version of this fix regenerated the CSV and `rmtree`'d
the splits in that path — a review measured 6 concurrent `rmtree` calls producing
5 `FileNotFoundError`s, and a purge could delete splits another process was
already training from. Keeping the path purely additive is what makes concurrent
prep safe.

> **Pre-launch action required.** CPTAC-PDAC still holds the stale pre-pivot
> `benchmark/dataset_csv/os.csv` (June, 8-way classification labels). Prep will now
> stop with instructions rather than silently proceed. Purge it before launching:
> ```bash
> PDAC=/home/yinshuol/projects/rrg-jma/shared/Pathology/CPTAC/CPTAC-PDAC
> rm -f  $PDAC/benchmark/dataset_csv/os.csv
> rm -rf $PDAC/benchmark/splits/standard/os
> ```
> More generally: **after any manifest rebuild, delete that cohort's
> `benchmark/dataset_csv/` and `benchmark/splits/`** — the cache detects a schema
> change but cannot detect that a same-schema CSV is merely out of date.

---

## 2. Non-blocking issues fixed

| # | Issue | Fix |
|---|---|---|
| N1 | Non-positive `OS_time` reached training. Cox is undefined at t≤0 and nllsurv would bin it at 0. LGG contained `OS_time = -1.0`; LUAD 3 rows at 0 | Filter `time > 0` in the survival task builder, with the drop reported |
| N2 | Merge column collision emitted `OS_event_x`/`_y` with no warning — reachable via the tutorial's own documented recipe | Raise on collision in `prepare_cptac_manifest.py`; `add_os` now drops-then-rejoins |
| N3 | An all-null label column produced a silently **empty** task CSV; failure surfaced far away as `min() arg is an empty sequence`. This is the shape a wrong `--case-col` join takes | Raise with a diagnostic naming the likely cause |
| N4 | Label values absent from `label_map` became NaN via `.map()` and flowed into the splits (e.g. a future 4th grade class) | Raise, listing the unmapped values |
| N5 | `add_grade_to_manifest.py` reported exclusions clinical-wide (`GX 18, G4 7`) rather than manifest-scoped (`GX 11, G4 4`), and could never report "no grade row" | Scope to the manifest's cases; fold in absent cases |

**Known-latent, not fixed** (not reachable in this roster, recorded so they are
not rediscovered): `fetch_gdc_clinical.py` has no GDC pagination guard (both
CPTAC cohorts fit one chunk); `prepare_cptac_manifest.py` overwrites a manifest
unconditionally; nnMIL's `roc_auc_score(multi_class='ovr')` **raises** when a
class is absent from a fold while CLAM/ABMIL/DTFD/TITAN `nanmean` over present
classes — divergent behaviour that current class supports should not trigger.

---

## 3. Open — strategic decisions (not yet actioned)

### O1. The headline claim is contradicted by our own prior evidence

`tasks/baseline_summary/REPORT.md` (210 configs, 35 task-pairs) measured this
exact contrast:

- **Encoder** spread: H-Optimus-1 0.635 / Virchow2 0.617 / UNI v2 0.615 — **2.0
  points**, described in our own text as *"essentially flat"*.
- **Aggregator** spread: `clam_mb` 0.637 vs `simple_mil` 0.607 — **3.0 points**.

On classification our data says **aggregator > encoder**, the reverse of the
headline. `BACKGROUND.md` states it plainly; it has never propagated into
`PLAN.md`.

Structural cause: the encoder axis is three 2024–25 ViT foundation models trained
on overlapping data at one magnification. The Frontiers precedent obtained its
encoder spread partly from **ResNet50** and a slide-level encoder; we removed
both ends of the range and kept the compressed middle. As designed the encoder
axis cannot exhibit variance.

**Options:** (a) restore dynamic range with a legacy encoder (ResNet50 /
CTransPath — cheap to extract); or (b) reframe honestly as a non-replication:
*"among modern patch foundation models, once recipe effort is equalised, encoder
choice does not dominate aggregator choice."* Option (b) is supported by data we
already hold and makes the agentic de-biasing layer load-bearing.

### O2. The agentic search would select on ~10 validation patients

`splits.py` carves val at `test_size=0.125` of train_val after an 80/20 outer
fold → **~10 val patients** for CPTAC-GBM (n=99) and CPTAC-PDAC (n=105), ~3.5 per
class for PDAC. Selection uses `(val_auc + val_bacc)/2` with `accept_margin`
defaulting to **δ = 0.0**, while discovery screens ~60 candidates per cell. The
maximum of 60 draws from that noise distribution is a large apparent lift that
will not survive to the sealed test block — i.e. Fig 3, the framework's money
figure, would be measuring selection noise on exactly the two cohorts the pivot
added.

**Recommended, zero GPU cost, before any agentic cell runs:** make the selection
composite an inner-CV mean rather than one held-out split; set
`scoring.accept_margin` per cohort scaled to the val SE (δ ≈ 0.05 on GBM/PDAC vs
~0.015 on LUAD); recompute `composite` orchestrator-side; and pre-register that
Fig 3's lift is reported on the sealed test block with the val lift shown as a
selection-bias diagnostic.

### O3. Power, figures, and TITAN

- **Power.** Minimum detectable paired AUC difference ≈ **0.10** on GBM/PDAC vs
  an expected encoder effect of ~0.02. They cannot resolve the headline contrast
  individually. Their real cost is noise: an unweighted mixed model lets the two
  noisiest cohorts inflate the residual and shrink *both* variance components
  toward zero — a power failure that reads as "nothing matters". Use
  precision-weighted / heteroscedastic variance components, treat GBM/PDAC as a
  declared small-sample secondary analysis, and consider ≥3 seeds there (they are
  the two cheapest cohorts in the grid).
- **Figures.** Fig 1 must not place binary AUC and macro-OvR AUC on one colour
  scale — use within-dataset centred ΔAUC and never print a cross-dataset column
  mean. Fig 2 should show the residual/fold-noise component alongside encoder and
  aggregator. Delete the figure-plan line that pre-draws "TITAN as the
  slide-level winner".
- **TITAN is a launch gate, not a nice-to-have.** `conch_v15` is unextracted for
  **all five** cohorts, so the grid currently yields 30 exps/dataset (150), not
  33 (165), with an empty TITAN row. Without TITAN *and* without a legacy anchor
  the encoder axis is three near-identical models and the finding is preordained.
- **Task type is perfectly aliased with cohort** (one classification task per
  cohort), so no task-type effect is estimable. Describe the roster's diversity
  as generalisation breadth, never as a designed axis.

---

## 4. Verified clean

- **Data integrity.** All five label distributions reproduce exactly at patient
  level; zero label conflicts across a patient's slides; zero duplicate slide ids
  and zero cross-case collisions; feature coverage exactly 1:1 and set-identical
  across all three encoders in all five cohorts.
- **HNSC `GRADE` derivation** independently recomputed from `clinical.tsv` —
  **0 mismatches**, 54/260/100 = 414 gradeable of 431. No case carries two
  distinct non-null grades, so "first non-null" is unambiguous rather than an
  arbitrary tie-break.
- **Multi-class wiring is genuine**, not merely configured: `label_map` inverts
  preserving ordinal order and `n_classes` reaches CLAM (incl.
  `subtyping=n_classes>2`), ABMIL, DTFD, TITAN and nnMIL; `evaluate.py` branches
  correctly for AUC / F1 / sens-spec.
- **The float-`GRADE` → `astype(int)` trap is safe**: `dropna` on the label column
  runs *before* the cast, so `IntCastingNaNError` cannot fire (confirmed the
  failure does occur if the order is reversed).
- **Splits are patient-level**, which matters because the CPTAC cohorts average
  ~2.4 slides/patient: `groupby("case_id")` → split cases → expand to slides,
  with an explicit leakage assertion.
- **5-fold is feasible for all ten task arms** — global minimum case-level class
  count is 22 (PDAC os-censored), well above `n_splits=5`.
- **YAML ↔ manifest ↔ feature-filename consistency** verified for all five.

**One reviewer claim did _not_ hold up:** the TITAN 512 px vs 224 px path was
called a launch foot-gun. It is handled — `submit_titan_extract.sh` extracts at
512 px, writes `20x_512px_0px_overlap/slide_features_titan`, then symlinks it to
`20x_224px_0px_overlap/features_titan`. The convention is deliberate.

---

## 5. Doc inconsistencies still to fix

1. `EXECUTION_PLAN.md` says TITAN is **4096-d**; the code is **768-d**
   (`pipeline/config.py`, `titan/prepare.py`). The doc is wrong.
2. `PLAN.md` §2 still claims `ab_mil` results exist on disk and adding it is
   "free re-aggregation". `EXECUTION_PLAN.md` refutes this — coverage is
   essentially zero. Changes the compute plan.
3. `PLAN.md` §3/§4 assert encoder ≫ aggregator with no acknowledgement of O1.
4. HNSC `n` is quoted as **431** in the roster tables; the gradeable n is **414**.
   431 applies to survival only — Table 1 must carry both, per task.
5. `PLAN.md` §4 defers regression on the premise that "no continuous target
   exists in any dataset". PDAC's immune subtype is an exactly-balanced 35/35/35
   tertile split — i.e. a binned continuous score. The premise was tested only
   against ovarian HRD. Disclose the tertile derivation in Table 1 regardless: a
   reviewer who sees 35/35/35 unlabelled will assume it is being hidden.
6. `registry.protected` ships **empty** everywhere. The frozen-substrate freeze is
   `PLAN.md` §5's central rigor claim and is currently documented but unenforced.
