# Running your MIL benchmark for the preprint (fir HPC)

Every member runs the **full MIL grid on their own dataset** and writes to the
lab-shared results tree. **5-fold** patient-stratified CV (the lab standard),
**one task per dataset**. The commands are identical for everyone — you just pass
your dataset name.

## Your assignment

| Member | Dataset | Task | `<dataset>` arg |
|--------|---------|------|-----------------|
| Leo | TCGA-LUAD | EGFR + KRAS | `tcga_luad` |
| Yeonwoo | TCGA-LGG | IDH1 | `tcga_lgg` |
| Keishi | TCGA-COAD | BRAF | `tcga_coad` |
| atatc | TCGA-SKCM | NRAS | `tcga_skcm` |
| Ryan | TCGA-CESC | PIK3CA | `tcga_cesc` |

Find your row — everywhere below, `<dataset>` is your arg (e.g. `tcga_lgg`).

---

## 1. One-time setup (on a fir login node)

```bash
ssh <you>@fir.alliancecan.ca
# If imports later fail with odd stdlib errors, that login node's /home is flaky —
# reconnect or try another (login2 / login3).

cd ~/scratch/autoMIL/autoMIL          # your repo checkout
git checkout main && git pull --ff-only   # get the latest configs + launchers
source .venv/bin/activate

# Make sure YOUR dataset root is in benchmarks/.env (already there if you
# extracted the dataset). Replace XXXX with your code, e.g. SKCM:
grep -q AUTOBENCH_TCGA_XXXX_ROOT benchmarks/.env || \
  echo 'AUTOBENCH_TCGA_XXXX_ROOT=/project/6114359/shared/Pathology/TCGA/TCGA-XXXX' >> benchmarks/.env
```

`HF_TOKEN` (for TITAN's gated download) should already be in your `benchmarks/.env`.

---

## 2. Run it — 3 sbatch commands, from the repo root

```bash
DS=tcga_lgg    # <-- YOUR dataset from the table

# (a) tile-encoder grid: CLAM + nnMIL + DTFD + ABMIL, 5-fold, 4 GPUs
sbatch benchmarks/scripts/slurm/submit_benchmark.sh $DS

# (b) extract TITAN features — ONCE, fold-independent (~half a day, 1 GPU)
EXTRACT=$(sbatch --parsable benchmarks/scripts/slurm/submit_titan_extract.sh $DS)

# (c) TITAN arm — auto-starts after (b) succeeds
sbatch --dependency=afterok:$EXTRACT benchmarks/scripts/slurm/submit_titan.sh $DS
```

Jobs are **idempotent** (finished experiments are skipped) and **auto-resubmit**
on the 24 h wall, so re-running any command is safe. Add a fold count as a 2nd
arg for a comparison run (e.g. `submit_benchmark.sh $DS 10`).

---

## 3. Where results land (lab-shared)

```
/project/6114359/shared/Pathology/autoMIL/phase2/<dataset>/benchmark_5fold/results/
  <framework>/standard/<task>/<encoder>/<model>/
      summary.json           # mean + 95% CI across folds
      fold_<i>/metrics.json  # per-fold val/test AUC + balanced accuracy
```

## 4. Check progress

```bash
squeue -u $USER
BD=/project/6114359/shared/Pathology/autoMIL/phase2/$DS/benchmark_5fold
find $BD/results -name summary.json | wc -l        # 22 when complete (21 tile + 1 TITAN); LUAD = 44
cat  $BD/results/_failed.json 2>/dev/null || echo "no failures"
tail -f logs/bench_mil_bench_<jobid>.out
```

---

## Notes

- **5-fold, one task per dataset.** The task is fixed per dataset (see table) and
  clears the 5-fold minority-class guard. Only Leo also runs LUAD at 10-fold
  (`submit_benchmark.sh tcga_luad 10`) for comparison — nobody else needs to.
- **Don't skip TITAN.** It's a required arm of the comparison. Step (b) — the
  512 px CONCH extraction — is the slow part (~half a day per dataset); the arm
  in step (c) is quick.
- **TITAN native recipe.** TITAN features are CONCH v1.5 @ 20×/512 px → 768-d
  (not the 224 px tile grid). This is deliberate — forcing a uniform patch size
  would handicap it. Its results row is a slide-level model (its own aggregation),
  not directly "same-aggregator" comparable to the tile-encoder rows.
- Tile features (H-optimus-1, UNI-v2, Virchow2) are **already extracted** for all
  slate datasets, so step (a) starts training immediately.
- One launcher, any dataset: `submit_benchmark.sh <dataset> [n_folds]` reads
  tasks/encoders/rosters straight from `benchmarks/datasets/<dataset>.yaml`.
