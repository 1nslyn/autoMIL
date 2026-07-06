# Reproducing the TCGA-LUAD MIL benchmark (fir HPC)

Runs the full MIL grid on TCGA-LUAD and writes results to the lab-shared results
tree. **5-fold** patient-stratified CV (the lab standard).

**What runs:** CLAM, nnMIL, DTFD, ABMIL (tile-encoder → MIL) + **TITAN** (slide
encoder, its own aggregation) × 3 encoders (H-optimus-1, UNI-v2, Virchow2) × 2
tasks (EGFR, KRAS) = **44 experiments**. Each writes per-fold `metrics.json` +
one `summary.json` (mean / 95% CI).

---

## 1. One-time setup (on fir)

```bash
# Use a HEALTHY login node — login1's /home mount is currently broken.
ssh yinshuol@login3.fir.alliancecan.ca        # or login2

cd ~/scratch/autoMIL/autoMIL
git checkout feat/mil-model-integration && git pull --ff-only
source .venv/bin/activate
python -c "import autobench, torch; print('ok', torch.__version__)"   # sanity
mkdir -p /scratch/yinshuol/autoMIL/logs
```

`benchmarks/.env` (paths + HF_TOKEN) is already configured on the cluster.

---

## 2. Run it (3 sbatch commands, from the repo root)

```bash
# (a) Extract TITAN features — ONCE, fold-independent (~half day, 1 GPU).
#     Skip if benchmark's TITAN arm already has features (see "Check" below).
EXTRACT=$(sbatch --parsable benchmarks/scripts/submit_luad_titan_extract.sh)

# (b) The tile-encoder grid (5-fold, 4 GPUs). No argument = 5-fold.
sbatch benchmarks/scripts/submit_luad_benchmark.sh

# (c) The TITAN arm — auto-starts after extraction succeeds.
sbatch --dependency=afterok:$EXTRACT benchmarks/scripts/submit_luad_titan.sh
```

Jobs are **idempotent** (finished experiments are skipped) and **auto-resubmit**
on the 24 h limit, so you can safely re-run any command.

---

## 3. Where results land

```
/project/6114359/shared/Pathology/autoMIL/phase2/tcga_luad/benchmark_5fold/results/
  <framework>/standard/<task>/<encoder>/<model>/
      summary.json          # mean + 95% CI across folds
      fold_<i>/metrics.json # per-fold val/test AUC + balanced accuracy
```

## 4. Check progress

```bash
squeue -u $USER
BD=/project/6114359/shared/Pathology/autoMIL/phase2/tcga_luad/benchmark_5fold
find $BD/results -name summary.json | wc -l          # 44 when complete
cat  $BD/results/_completed.json | python -c "import json,sys; print(len(json.load(sys.stdin)),'experiments done')"
cat  $BD/results/_failed.json 2>/dev/null || echo "no failures"
tail -f /scratch/yinshuol/autoMIL/logs/bench_luad_bench_<jobid>.out
```

---

## Notes

- **TITAN native recipe.** TITAN is extracted at CONCH v1.5 / 20×/512 px → 768-d
  (not the 224 px tile grid). This is deliberate — forcing a uniform patch size
  would handicap it. Its results row is a slide-level model (own aggregation),
  not directly "same-aggregator" comparable to the tile-encoder rows.
- **Another dataset?** Ensure `benchmarks/datasets/<name>.yaml` + its
  `AUTOBENCH_<NAME>_ROOT` exist and tile features are extracted
  (`submit_feature_extraction.sh <name>`), then run
  `run_benchmark.py --dataset <name> --frameworks clam nnmil dtfd abmil --n_folds 5`.
- **10-fold comparison (Leo).** Append the fold count: `sbatch
  submit_luad_benchmark.sh 10` and `sbatch --dependency=afterok:$EXTRACT
  submit_luad_titan.sh 10` → writes to a separate `benchmark_10fold/`. The shared
  TITAN extraction in step (a) serves both.
