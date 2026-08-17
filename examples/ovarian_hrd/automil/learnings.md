# Learnings

Persistent knowledge base for the autoMIL experiment loop.
Consolidated sections at top are read FIRST on every session restart.
Per-experiment entries are appended below.

## What Works

- **Disabling instance evaluation** is always better (+0.013 primary_value). Small dataset (N=206) can't support instance-level supervision.
- **Focal loss with gamma=1.0** is optimal. gamma=0.5 is mild, gamma>=1.5 degrades badly, gamma=3 is catastrophic (-0.170).
- **Gradient clipping at 0.5** helps stability (+0.005). gc=1.0 is too loose.
- **R-Drop regularization (alpha=1.0)** provides the biggest single regularization improvement (+0.004 on already-tuned config). Alpha=0.75/1.5/2.0/3.0 all worse. KL consistency is the right inductive bias for small N.
- **Step LR (halve every 20 epochs after warmup)** beats cosine, exponential, flat+cosine, warm restarts, and plateau schedules (+0.001 over cosine baseline).
- **Class-balanced focal loss** (alpha weighting) helps slightly over unweighted focal.
- The **current best config is simple**: hoptimus1 + CLAM_MB + focal(g=1) + no_inst_eval + gc=0.5 + R-Drop(a=1.0) + step_lr(halve/20ep). Primary_value: 0.850.
- Incremental improvements compound: baseline 0.814 -> no_inst 0.827 -> focal 0.832 -> gc 0.845 -> R-Drop 0.849 -> step_lr 0.850.

## What Doesn't Work

- **L2 feature normalization** (-0.063). Destroys magnitude information that CLAM attention relies on.
- **Label smoothing** (any amount). Hurts with focal loss.
- **Large model size** ("big" CLAM) alone: marginal benefit, not worth complexity.
- **Lower learning rates** (1e-4, 2e-4): consistently worse. 3e-4 is optimal.
- **Higher weight decay** (1e-3, 5e-4): all worse than 1e-4.
- **SGD optimizer** (-0.251). Catastrophic. AdamW is essential.
- **SAM optimizer** (-0.129). Too aggressive for small dataset.
- **EMA/SWA** (-0.026 to -0.051). Averaging hurts, possibly because landscape is noisy.
- **Other encoders** (uni_v2, hibou_l): all substantially worse than hoptimus1 on CLAM_MB.
- **Other architectures** (TransMIL -0.081, DTFD -0.057, ILRA -0.065, ViT -0.080, AB_MIL -0.037): CLAM_MB dominates for this task.
- **Feature augmentation** (patch dropout, noise, mixup, feature dropout, feature perturbation): minimal to negative impact. The features are already high quality.
- **Feature centering** (per-slide mean subtraction): -0.097. Removes useful signal.
- **PseMix pseudo-bag augmentation** (-0.048). Pseudo-bag splitting hurts attention learning.
- **Attention entropy maximization** (-0.081). Forcing uniform attention is counterproductive for HRD where few patches are informative.
- **Variance pooling** (-0.070). Adding variance branch dilutes mean signal.
- **Top-K sparse attention** (-0.286). Catastrophic. Hard masking loses too much context.
- **Dropout warmup** (0->0.1 over 20ep): near-miss (-0.002) but adds complexity for no gain.
- **Test-time dropout** (10 passes): no improvement, 5x slower.
- **Confidence penalty** (-entropy to loss): -0.081. Conflicts with focal loss.
- **Enhanced CLAM** (custom modifications): -0.066. Original architecture is well-designed.
- **Lookahead optimizer**: -0.016. No benefit over plain AdamW.
- **Multi-encoder fusion** (hoptimus1+uni_v2): -0.007. Two encoders don't complement each other on this task.

## Patterns and Heuristics

- CLAM models process slides individually (per-sample loop), so DataParallel provides no benefit. Use gradient accumulation instead.
- Each 5-fold CV takes ~60-100 min depending on model complexity. Budget accordingly.
- VRAM is ~0.4-0.5 GB for CLAM_MB. ViT/TransMIL use dramatically more (1.7-31 GB).
- Experiments near the current best (within 0.005 primary_value) that add complexity should be discarded per the simplicity criterion.
- R-Drop requires 2 forward passes per sample, increasing training time by ~30%.
- When a new regularization technique conflicts with R-Drop (e.g., LS, confidence penalty), R-Drop usually wins.
- Focus remaining efforts on architectural innovations that change how attention or pooling works, not on training recipe tweaks. The recipe is heavily optimized.

## Experiment Log

### Bootstrapped from 176 experiments (2026-03-15 to 2026-03-16)

Full history in results.tsv. Key milestones:
- Exp 1: Baseline hoptimus1+clam_mb, primary_value 0.814
- Exp 5: Disabled instance eval, primary_value 0.827 (+0.013)
- Exp 18: Added focal loss gamma=1, primary_value 0.832 (+0.004)
- Exp 52: Combined focal+alpha+no_inst, primary_value 0.840 (+0.008)
- Exp 57: Added gradient clipping 0.5, primary_value 0.845 (+0.005)
- Exp 76: Added noise aug 0.001, primary_value 0.846 (+0.0001, marginal)
- Exp 98: Added R-Drop alpha=1.0, primary_value 0.849 (+0.004)
- Exp 148: Switched to step LR (halve/20ep), primary_value 0.850 (+0.001)

176 total experiments. 9 kept, 167 discarded, 0 crashes.
Best primary_value: 0.850133. Best config: hoptimus1+CLAM_MB+R-Drop(a=1.0)+focal(g=1.0)+no_inst_eval+gc=0.5+step_lr(halve/20ep).

### Round 54-55 (2026-03-16)

- **AEM lambda=0.5**: primary_value 0.770 (-0.081). Forcing uniform attention is counterproductive.
- **AEM lambda=0.05**: primary_value 0.800 (-0.052). Still harmful even at 10x gentler.
- **Variance pooling**: primary_value 0.780 (-0.070). Doubling classifier input dilutes mean signal.
- **PseMix K=4**: primary_value 0.802 (-0.048). Pseudo-bag splitting + mixup hurts attention.
- **Top-K sparse attention (K=256)**: primary_value 0.564 (-0.286). Catastrophic. Hard masking destroys learning.
- **Coordinate PE (sinusoidal 2D, 0.1 scale)**: primary_value 0.851 (+0.001). **NEW BEST!** Spatial awareness helps.
  Key insight: spatial position matters for HRD prediction. Adding gentle positional encoding to features before CLAM attention lets the model learn tissue-structure-aware attention patterns.

### Round 56-58 (2026-03-16)

- **Coord PE scale 0.2**: primary_value 0.840 (-0.012). Too strong, PE overwhelms features.
- **Coord PE scale 0.05**: primary_value 0.851 (-0.0003). Almost identical to 0.1, very robust.
- **Instance masking (ACMIL-inspired)**: primary_value 0.565 (-0.286). Catastrophic, hard masking always fails.
- **MaxSoft pooling**: primary_value 0.816 (-0.036). Straight-through estimator doesn't work well with CLAM.
- **SupCon loss (weight=0.2)**: primary_value 0.806 (-0.046). Contrastive learning on 2D logits is too weak.
- **Bi-level LR (attn 2x)**: primary_value 0.767 (-0.084). Uniform LR is already optimal.
- **Learned attention temperature**: primary_value 0.844 (-0.008). Temperature hurts, fixed softmax is better.
- **Coord PE + R-Drop alpha=0.5**: primary_value 0.851 (-0.0006). Near-identical to alpha=1.0 with coord PE.
- **Late fusion hoptimus1+uni_v2**: CRASHED (model cleanup before probs extraction).

Key insight: The coord PE (scale=0.1) remains the only improvement since step LR. The model is extremely robust to all architectural changes, suggesting we may be near the data-limited ceiling for this task at ~0.851 primary_value.
