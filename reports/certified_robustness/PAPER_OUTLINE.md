# Paper Outline — Certified Robustness of Dose Prediction

Working title: *Certified Robustness of Deep-Learning Dose Prediction: Provable DVH Bounds and
Adaptive-Attack-Resistant Test-Time Smoothing in Head-and-Neck Radiotherapy*

Target venue: **Medical Physics** or **Physics in Medicine & Biology** (med-phys, not top-tier ML).
Status: results in hand for the core claims; two experiments + writing remain (see §Gaps).
NOTE: separate from the ASTRO 2026 poster (that is the earlier CT-perturbation study; text locked).

## Purpose

DL dose predictors are moving toward clinical use, but their reliability under perturbed CT input
has only been characterized by *average* accuracy. Prior work characterizes vulnerability (Gao et
al. 2025, Phys. Med. Biol.) without mitigation or guarantee. We ask: (1) can we give provable,
clinically-meaningful (DVH / gamma) guarantees on predicted dose under bounded CT perturbations,
and (2) does a cheap test-time defense survive a defense-aware adaptive attacker?

**Verified gap (lit review):** no prior work provides certified/provable adversarial robustness for
dose prediction with DVH guarantees. See `literature/LITERATURE_REVIEW.md` (cited, verified).

## Methods

- Model: 3D U-Net (SE, InstanceNorm), OpenKBP H&N, 40-patient validation. (First pass on the
  baseline model, DVH 2.536; DVH0.02 model to be retrained.)
- (1) **Adaptive robustness:** EOT-PGD adaptive attack against test-time Gaussian-noise smoothing
  (noise is differentiable → EOT is the correct adaptive attack, not BPDA), escalated in strength,
  vs the non-adaptive attack. Code: `open-kbp-modified/adversarial_adaptive.py`.
- (2) **Certification:** median (percentile) randomized smoothing — the correct certificate for
  voxel-wise *regression* (mean has none; Chiang et al. 2020) — with finite-sample order-statistic
  confidence bounds, pushed through monotonically to **certified DVH intervals** (valid because DVH
  metrics are monotone in per-voxel dose). Swept over σ; radius reported in L2 + HU (both per-voxel
  RMS and single-voxel max). Code: `open-kbp-modified/certify_smoothing.py`,
  `provided_code/smoothing_certify.py` (unit-tested).
- **Metrics:** voxel MAE, DVH score, **gamma passing rate (3%/3mm and 3%/2mm vs TG-218 limits)** —
  gamma reported EMPIRICALLY (clean vs perturbed); certified-gamma flagged as harder future work
  (gamma's DTA is spatial/non-monotone, so the clean push-through does not apply).

## Results (verified, full 40-patient validation — see experiment_results/)

- **Defense HOLDS vs adaptive attack** (surprise, opposite of the original hypothesis): ~85–100%
  recovery at ε=0.02 (~82 HU), ~72–75% at ε=0.05 (~205 HU); survives escalation (EOT 16, PGD 20).
  Adaptive genuinely bit (adaptive+def > nonadaptive+def) yet defense absorbed most. Frame as
  "resisted OUR escalated EOT attack," never "provably robust."
- **First certified DVH bounds for dose prediction:** e.g. σ=0.02, R=0.01 → D95 PTV70 within
  ±0.45 Gy, mean brainstem ±0.42 Gy, 99.9% confidence, 40 patients. Clean σ–radius tradeoff.
- **Honest limitation:** tight guarantees only at small radii (R_max = 1.45σ at n=100 / 2.17σ at
  n=500); fundamental high-dimensional smoothing ceiling, not an execution flaw.

## Conclusion

First certified robustness for dose prediction, in clinical terms; test-time smoothing gives strong
empirical robustness that *exceeds* the certifiable radius; small certified radius is intrinsic to
smoothing in high dimensions; training on the noise (SmoothAdv) is the path to widen it.

## Threat-model framing (CORRECTED per lit review — important)

- Present ε=0.01–0.05 (41–205 HU) as **adversarial worst-cases**, NOT typical scanner noise:
  benign CT variability is ~1.5–4.9 HU single-scanner (Kim, PMC5768003) / ~3–11 HU inter-scanner
  (Lamba, AJR 2014). Do NOT use the "10–50 HU quantum noise" phrase without a dedicated citation
  (refuted in this pass).
- σ=0.1 (~410 HU) smoothing = certification mechanism, not realistic image content — state it.
- L2 ball ≠ the structured clinical perturbations (P2 bone-shift, P4 resolution) our earlier work
  flagged. Acknowledge; certified-L2 does not cover those.
- Define "adversarial/certified robustness" vs RT "robust optimization" up front (Eriksson &
  Bokrantz collision).

## Gaps to close before submission (ranked)

1. **SmoothAdv experiment** — train on the noise, re-certify to widen the radius. The pivotal run;
   turns a limitation paper into a method paper. (runpod_train.py change, not yet built.)
2. **Harden the adaptive claim** — 1+ additional adaptive attack variant; confirm ε=0.05 leak is a
   real ceiling, not under-powered EOT.
3. **Gamma-index results** — empirical GPR (3%/3mm, 3%/2mm) clean-vs-perturbed vs TG-218 limits.
4. **Contour-safety figure** — `save_contour_overlay_figures.py` output (subtle-threat evidence).
5. **HU-magnitude citations** — metal/kernel/slice-thickness figures (lit review open question).
6. **UQ/conformal contrast** — find a named evidential-DL/conformal DVH paper (open question).
7. **Near-submission preprint scan** — protect the "first" claim.
8. **Write it.**
