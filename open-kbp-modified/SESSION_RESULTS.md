# Session Results — Test-time defence + Gao replication (RunPod 4090, 2026-07-05)

Branch: adversarial-retraining. Validation set: pt_201-240 (40 patients). TF 2.18.0.

## Task A — Test-time augmentation defence (non-adaptive FGSM/PGD on CT input)
Baseline `epoch_100` (clean DVH 2.387): test-time Gaussian NOISE on the CT recovers most
attack damage at ~zero clean cost:
- FGSM eps0.02: noise sigma0.1 -> +79.5% recovered (clean cost +0.07)
- FGSM eps0.05: noise sigma0.1 -> +59.9%
- PGD  eps0.02: noise sigma0.05 -> +104%
- PGD  eps0.05: noise sigma0.1 -> +92.9%
Robust `epoch_125` (clean DVH 2.110): attacks barely land (damage +0.18..0.59); flip/noise
recover the small remainder. Verdict: test-time noise meaningfully blunts the attack on the
baseline (non-adaptive attacker; adaptive/BPDA is the rigorous follow-up).
Full JSON: adversarial_defense_results/{epoch100,epoch125}/defense_summary.json

## Task B — Accuracy (DVH loss + ensemble)
Baseline (re-scored in-env, full val): DVH 2.536 / Dose 3.731 (matches documented 2.535).
B1 single model + DVH loss (w=0.1): DVH 3.130 / Dose 4.068  -> DVH LOSS HURT; not used further.
Per-seed (baseline config, no DVH loss, XLA on):
  seed1 2.550/3.544 | seed2 2.438/3.665 | seed3 2.341/3.878 | seed4 2.485/3.654 | seed5 2.670/3.664
5-SEED ENSEMBLE:  DVH 1.911 / Dose 3.289   <-- best (-24.6% DVH, -11.9% Dose vs baseline)
  Closes ~59% of the DVH gap to the competition winner (1.478 / 2.429), which also used a cascade.

## Models (on /workspace, persistent)
- Ensemble:  results/ensemble_5seed_noDVH/ (+ validation-predictions/)
- Seeds:     results/64filter_100epoch_SE_AUG_MASK_PTV4.0_NORM_seed{1..5}/models/epoch_100.keras
- B1 (DVH):  results/64filter_100epoch_SE_DVH0.1_AUG_MASK_PTV4.0_NORM/

## Commands
Task A:   python adversarial_defense.py --model models_in/epoch_100.keras --attack fgsm pgd \
            --epsilons 0.02,0.05 --defenses none smooth noise intensity flip --n-patients 10 \
            --output adversarial_defense_results/epoch100/
Seeds:    for s in 1 2 3 4 5; do python runpod_train.py --filters 64 --epochs 100 --use-se \
            --use-aug --batch-size 4 --ptv-weight 4.0 --seed $s; done
Ensemble: python ensemble_predict.py \
            --models 64filter_100epoch_SE_AUG_MASK_PTV4.0_NORM_seed{1,2,3,4,5} \
            --epoch 100 --output ensemble_5seed_noDVH

## Timing / cost
~2 hrs total 4090 time (~$1). XLA (no DVH loss) ~15 min/seed; DVH-loss/no-jit path ~20 min.
