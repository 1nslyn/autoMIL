# Generalization-Gate Calibration — K Threshold Empirical Choice

> Relocated from `.planning/phase-05-calibration.md` on 2026-08-06 when the
> planning tree was removed. This is still the live operator recipe for the
> deferred calibration pilot.

**Status:** SCAFFOLD — operator-fillable. Replace `_tbd_` placeholders when running pilot.
**Date:** _to-be-filled_
**Operator:** Leo
**Known-good change:** CCRCC node_0176 (clam_mb_v0176 + ce_smooth008 + sam_lookahead)
**Held-out cells:** 3 fresh CCRCC + 2 fresh CLWD = 5 total

**Pilot recipe:** Apply node_0176's variant module + config delta to each held-out cell
via `automil apply node_0176` (Phase 1 REG-08) or manually copy the overlay files if
Phase 1 has not shipped. Submit via `automil submit --node <new_id_per_cell> --files ...`.
Wait for completion (estimated ~6h per cell; saturate GPUs by submitting all 5
simultaneously). Gather the matrix via:

```bash
uv run automil nominate <candidate_id>
uv run automil promote <candidate_id> --calibrate
cat automil/archive/<candidate_id>/gate_evaluation.jsonl
```

The first JSONL line contains the `per_cell_results` array. The second line is the
decision summary (result, p_value, CI, wins).

---

## Per-cell deltas (paired vs node_0176's parent)

| cell_id | dataset | encoder | task | parent_primary_value | candidate_primary_value | delta | wins |
|---------|---------|---------|------|------------------|---------------------|-------|------|
| _tbd_   | ccrcc   | _tbd_   | _tbd_ | _tbd_           | _tbd_               | _tbd_ | _tbd_ |
| _tbd_   | ccrcc   | _tbd_   | _tbd_ | _tbd_           | _tbd_               | _tbd_ | _tbd_ |
| _tbd_   | ccrcc   | _tbd_   | _tbd_ | _tbd_           | _tbd_               | _tbd_ | _tbd_ |
| _tbd_   | clwd    | _tbd_   | _tbd_ | _tbd_           | _tbd_               | _tbd_ | _tbd_ |
| _tbd_   | clwd    | _tbd_   | _tbd_ | _tbd_           | _tbd_               | _tbd_ | _tbd_ |

---

## Statistical summary

- p_value (paired Wilcoxon, alternative='greater'): _tbd_
- Bonferroni-corrected alpha (K=5, alpha=0.05/5 = 0.01): _pass_ / _fail_
- BCa bootstrap 95% CI on median delta: [_tbd_, _tbd_]
- Wins (delta > 0): _N_ / 5

---

## Recommended K

Empirical recommendation: K = _tbd_

Rationale: with the observed delta distribution, K = _tbd_ produces a gate that:
- reliably passes node_0176-equivalent improvements (true positive)
- rejects no-improvement candidates (true negative — confirmed by cross-checking
  against a known-no-op change if available)
- aligns with the framework default `max(2, len(held_out_cells) // 3)` from D-151 / O-01

If the recommended K differs from the framework default currently in
`src/automil/templates/config.yaml.j2`, update the `gate.K` default:
```yaml
gate:
  K: <chosen K>    # locked by the calibration pilot; see docs/calibration-pilot.md
```

Commit with message: `gate: lock K=<N> from the calibration pilot`

---

## Sign-off

- [ ] Leo reviewed the per-cell delta matrix
- [ ] Recommended K is recorded in `src/automil/templates/config.yaml.j2` `gate.K` default
  (update only if the empirical recommendation differs from `max(2, len(held_out_cells) // 3)`)
- [ ] Calibration recorded in `CHANGELOG.md` under the next release

---

## Procedure checklist (step-by-step)

1. **Pick fresh cells.** Run `uv run automil cell list` — choose 3 CCRCC + 2 CLWD where
   `status=active` AND `consumed_seconds` is low (well below the 6h cap). Note cell_ids.
   Avoid cells where the agent has already submitted experiments (D-150 / Pitfall 1).

2. **Confirm node_0176 portability.** If REG-08 (Phase 1) has shipped, use
   `uv run automil apply node_0176`. Otherwise, manually identify node_0176's overlay files
   (clam_mb model edits + ce_smooth loss + sam_lookahead policy) and copy them.

3. **Register the manifest.** For node_0176's parent, declare all 5 fresh cells as held-out:
   ```bash
   uv run automil gate register-manifest <node_0176_parent_id> \
       --K 3 \
       --p-threshold 0.05 \
       --bootstrap-reps 1000 \
       --strategy operator-curated \
       --held-out-cells "<5 cell tuples>"
   ```

4. **Submit candidates.** For each held-out cell, submit a candidate that applies
   node_0176-equivalent changes. Saturate GPUs (submit all 5 simultaneously).

5. **Run --calibrate.** When all 5 cells are complete:
   ```bash
   uv run automil nominate <candidate_id>
   uv run automil promote <candidate_id> --calibrate
   cat automil/archive/<candidate_id>/gate_evaluation.jsonl
   ```

6. **Fill in this document.** Replace `_tbd_` with real values. Compute recommended K based on:
   - How many cells have delta > 0 (wins)?
   - Does Wilcoxon p <= 0.05/K_chosen pass?
   - Is the BCa CI lower bound > 0?

7. **Lock K** (if different from framework default). Edit
   `src/automil/templates/config.yaml.j2` `gate.K` default and commit.

8. **Sign off.** Check the boxes above and commit this document.
   Acceptance: zero `_tbd_` markers remain; all sign-off boxes are checked.
