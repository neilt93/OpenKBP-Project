# RunPod Claude Brief — Certified robustness for photon dose prediction

You are a Claude Code instance running ON a RunPod 4090 box (network volume at `/workspace`).
This session turns the *empirical* noise defence into a *provable* one. Two jobs, in order:

1. **Bridge experiment (cheap, do first):** run the ADAPTIVE (EOT) attack on the existing
   test-time noise defence and measure how much of the 60-104% recovery survives an attacker
   that knows the defence is there. This is the honest motivation slide for the whole story.
2. **Certification:** median randomised-smoothing certificate on the best model — per-voxel
   certified dose intervals (Gy) and certified DVH intervals under an L2 CT-perturbation ball.

All code is built and the pure certificate math is unit-tested off-GPU. You are executing
runs, watching the gotchas below, and reporting numbers.

---

## 0. Environment
- 1x RTX 4090 24 GB, TF 2.18.0 (must match the training version — mixed-precision models are
  version-sensitive). Batch 1 for both scripts. Do NOT raise it.
- Durable outputs live under `/workspace`. Validation set = `provided-data/validation-pats`
  (pt_201-240). Best single model: `results/64filter_100epoch_SE_DVH0.02_AUG_MASK_PTV4.0_NORM/models/epoch_100.keras`
  (DVH 2.049); baseline `results/64filter_100epoch_SE_AUG_MASK_PTV4.0_NORM/models/epoch_100.keras`
  (DVH 2.536). Use whichever is present; prefer the DVH0.02 winner.

```bash
cd /workspace/openkbp && git checkout adversarial-retraining && git pull
python tests/test_smoothing_certify.py     # sanity: 7 certificate-math tests must pass
```

---

## 1. Bridge experiment — adaptive attack on the noise defence  (~10-15 min for 10 patients)

```bash
MODEL=results/64filter_100epoch_SE_DVH0.02_AUG_MASK_PTV4.0_NORM/models/epoch_100.keras
python adversarial_adaptive.py --model $MODEL \
    --epsilons 0.02,0.05 --pgd-steps 10 --eot-samples 8 \
    --defense-sigma 0.1 --defense-samples 8 \
    --n-patients 10 --output adaptive_results/
```

Reads out four conditions per epsilon: clean, non-adaptive PGD (defended), and **adaptive
EOT-PGD (defended)**. The headline is the last two columns of the summary table:
`nonadapt rec%` vs `ADAPTIVE rec%`.

**What to expect / how to read it.** If adaptive recovery collapses far below non-adaptive
(e.g. 80% -> 10-20%), the empirical defence is largely obfuscated gradients (Athalye 2018) —
the expected, publishable result that motivates certification. If adaptive recovery stays
high, the noise defence is stronger than assumed; either way it is a real finding. `--defense-sigma
0.1` matches the best sigma from `SESSION_RESULTS.md`; if you have time, also try `0.05`.

**Cost knob:** EOT is `pgd-steps x eot-samples` forward+backward passes per patient per eps
(here 10x8 = 80). Drop `--eot-samples` to 4 for a faster first look; raise to 16 for a tighter
gradient estimate if the adaptive attack looks too weak (an under-powered EOT gradient can
*flatter* the defence — rule that out before concluding the defence holds).

---

## 2. Certification — median randomised smoothing  (~10-20 min for 10 patients, n=100)

```bash
python certify_smoothing.py --model $MODEL \
    --sigma 0.05 --n-samples 100 --batch-draws 8 \
    --radii 0.5,1.0,2.0 --tol-gy 1.0 --alpha 0.001 \
    --n-patients 10 --output certify_results/
```

Produces, per L2 radius R: per-voxel certified dose-interval widths (median / p95 / max, Gy),
the fraction of in-body voxels certified within `--tol-gy`, and certified DVH-metric intervals.
`certify_summary.json` has per-patient detail incl. `dvh_interval_widths_gy`.

**How to read the certificate.** Narrower interval = stronger guarantee. `~HU/vox` in the
table is the per-voxel RMS of the L2 ball (R/sqrt(V) x 4095) — the honest per-voxel size of
the perturbation being certified against. Expect the whole-volume L2 ball to look small per
voxel; that tension is inherent to smoothing and should be reported, not hidden.

**Tuning sigma (the key trade-off).** Larger `--sigma` certifies a larger radius but widens
the intervals (less accurate under noise). Sweep `--sigma 0.02,0.05,0.10` (separate runs,
separate `--output` dirs) to map the accuracy/robustness curve. `--n-samples` controls
confidence tightness and cost linearly; 100 is a fast first pass, use 500-1000 for a final
number. `--alpha 0.001` = 99.9% confidence.

**Uncertified flag:** if a run prints `certified_low/high = False` for some radius, the sample
count is too small for that radius/sigma (the order-statistic rank fell off the end) — raise
`--n-samples` or lower the radius. The code clamps to the data range and flags it rather than
emitting a fake bound.

---

## 3. Report back
For each script: the summary table, the JSON path, and one sentence on the finding. For the
bridge experiment specifically, state the non-adaptive vs adaptive recovery gap — that single
number decides whether the certified-defence framing is motivated.

## 4. Phase 2 (only if 1-2 land and there is GPU budget)
Fine-tune the base model on Gaussian-noised CTs (SmoothAdv, Salman et al.) so it predicts well
UNDER the smoothing noise, then re-certify — this is the lever that widens the certified radius.
That is a `runpod_train.py` change (add train-time input noise at `--sigma`), not built yet;
flag it back rather than improvising.
