# Fixed DVH-loss re-run — Results (RunPod 4090, 2026-07-07)

Branch: adversarial-retraining. Fixed DVH loss = commit `d7f28a8` (exact tf.sort +
linear-interp percentile, averaged over the whole batch). Validation = pt_201–240 (40).
TF 2.18.0, batch 4, --no-jit, PTV weight 4.0, SE + augmentation, CT_MAX=4095. Train pt_1–200 only.

## Baselines (for comparison)
- Baseline (no DVH):            **DVH 2.536 / Dose 3.731**
- No-DVH single seeds:          DVH 2.34–2.67 (best seed 2.341)
- No-DVH 5-seed ensemble (best):**DVH 1.911 / Dose 3.289**
- Old BUGGY DVH@0.1 (pre-fix):  DVH 3.130 / Dose 4.068  (regressed)
- Competition winner:           DVH 1.478 / Dose 2.429

## Stage 1 — single-model DVH weight sweep (fixed loss)  [COMPLETE]
| Weight | DVH   | Dose  | vs baseline DVH | Note |
|--------|-------|-------|-----------------|------|
| 0.02   | **2.049** | 3.668 | **−19.2%** | WINNER; beats every no-DVH single seed (best 2.341) |
| 0.05   | 2.278 | 3.707 | −10.2% | also beats baseline |
| 0.10   | 3.234 | 4.497 | +27.5% (worse) | regresses even with the fix — weight too high |

**Verdict:** the percentile fix turned the DVH loss from a liability into a real gain, but
ONLY at low weight. The old estimator regressed to 3.130; the fixed loss at w=0.02 reaches
**2.049**, the first time the DVH loss beats the plain models as a single model. DVH score is
monotonic in the weight (0.02 < 0.05 < 0.10), so the sweet spot is low; 0.10 confirms too much
DVH weight still fights the MAE even with a correct percentile.

Model dirs: `results/64filter_100epoch_SE_DVH{0.02,0.05,0.1}_AUG_MASK_PTV4.0_NORM/`
(trimmed to epoch_100 + results.json). Old buggy model preserved at
`...SE_DVH0.1_..._NORM_PREFIX_BUGGY/`.

## Stage 2 — 5-seed DVH-loss ensemble at w=0.02  [COMPLETE]
Per-seed (100 epochs each, seeds 1–5), 2-way concurrent 11:18 → 12:44:
| Seed | DVH | Dose |
|------|-------|-------|
| 1 | 2.093 | 4.161 |
| 2 | 2.171 | 3.637 |
| 3 | 2.533 | 4.047 |
| 4 | 3.199 | 3.874 |
| 5 | 2.395 | 3.549 |
| avg | 2.478 | 3.854 |

**5-seed ensemble (average of predictions): DVH 1.837 / Dose 3.299**

| Ensemble | DVH | Dose |
|----------|-------|-------|
| no-DVH 5-seed (previous best) | 1.911 | 3.289 |
| **DVH-loss 5-seed @0.02 (NEW BEST)** | **1.837** | 3.299 |

- **−3.9% DVH vs the previous best**, Dose tied (+0.01, noise). New best DVH for the project.
- vs baseline 2.536/3.731: **−27.6% DVH, −11.6% Dose**.
- vs competition winner 1.478/2.429: DVH gap narrows from 0.433 → **0.359** (closes ~17% more of it).
- The seeds are high-variance and individually worse on Dose than no-DVH seeds, yet the
  ensemble still wins on DVH and matches Dose — the DVH loss adds useful, diverse signal that
  survives averaging.

Paths: seeds `results/64filter_100epoch_SE_DVH0.02_AUG_MASK_PTV4.0_NORM_seed{1..5}/models/epoch_100.keras`;
ensemble preds `results/ensemble_5seed_dvh0.02/validation-predictions/` (+ results.json, submission zip).

## Conclusion
The percentile fix (`d7f28a8`) is validated end-to-end: the DVH-aware loss, previously a
regression (3.130), now **improves both the single model (2.049, −19% vs baseline) and the
ensemble (1.837, a new project best)** at a low weight (0.02). High weights (0.10) still hurt.
Recommend w=0.02 as the DVH-loss setting and `ensemble_5seed_dvh0.02` as the new best model.

## Timing / cost
Stage 1 (3 weights, 2-way concurrent, 0.02 resumed): ~42 min. Stage 2 (5 seeds + ensemble,
2-way): ~86 min. Effective 4090 compute ≈ 2 hrs (~$1). 2-way concurrency held the GPU at
96–97% / 258 W (~1.63× throughput) vs ~68% single-job — cut Stage 2 from ~110 min to ~86 min.

## Exact commands
Stage 1 (per weight w in 0.02/0.05/0.1):
  python runpod_train.py --filters 64 --epochs 100 --use-se --use-aug --use-dvh --dvh-weight $w \
      --batch-size 4 --ptv-weight 4.0 --no-jit
Stage 2 (per seed s in 1..5, winning weight 0.02):
  python runpod_train.py --filters 64 --epochs 100 --use-se --use-aug --use-dvh --dvh-weight 0.02 \
      --batch-size 4 --ptv-weight 4.0 --no-jit --seed $s
  python ensemble_predict.py --models 64filter_100epoch_SE_DVH0.02_AUG_MASK_PTV4.0_NORM_seed{1,2,3,4,5} \
      --epoch 100 --output ensemble_5seed_dvh0.02
Concurrency wrappers (2-way, disk-bounded): /workspace/run_concurrent.sh, /workspace/run_stage2_concurrent.sh

## Environment / efficiency notes
- Two interruptions handled: /workspace disk-quota crash (freed 40 G; runs now trim to
  epoch_100 after each) and a session teardown (jobs relaunched detached via setsid + resumable).
- GPU utilisation: one batch-4 --no-jit job ≈ 68% util / 180 W. Ran 2 jobs concurrently
  (17.2 GB of 24 GB) → **96–97% util / 258 W, ~1.63× aggregate throughput** (each job ~14.7 s/epoch
  vs 12 s solo). Used for Stage 2's 5 seeds.
