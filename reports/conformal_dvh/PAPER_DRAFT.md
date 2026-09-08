# Conformal DVH intervals for dose prediction — Intro + Related Work (draft)

Draft scaffolding for the conformal strand. Numbers marked [FILL] come from the pod run
(`conformal_dvh.py` on real `per_patient_dvh`). Positioning reflects the settled novelty search
(reports/conformal_dvh/EXPERIMENT_PLAN.md, "Novelty" section).

## Contributions (state up front)
1. **First conformal prediction intervals on DVH metrics of a radiotherapy dose predictor** —
   distribution-free, finite-sample coverage on D95, Dmean, D0.1cc, etc. (marginal per-criterion
   and a joint per-patient band over all 23 criteria).
2. **First study of conformal DVH coverage *collapse* under clinically realistic CT perturbations**
   — using a 26-condition P1–P5 battery, we show at what image-degradation severity the 90%
   guarantee actually breaks (headline: P4/resolution first).
3. A **weighted-CP baseline** showing covariate-shift reweighting does **not** rescue coverage under
   these perturbations, because they worsen the conditional error P(Y|X), not just the covariate
   distribution — distinguishing our setting from prior covariate-shift conformal work.
4. Placement of conformal coverage alongside empirical robustness and adversarial certificates on
   the **same model and patients** — an honest ledger of what each guarantee is worth.

## 1. Introduction (skeleton)
- Dose prediction is nearing clinical use (auto-planning, online adaptive RT), so per-patient
  *trustworthy* error bars matter, not just average accuracy.
- Existing uncertainty quantification for dose/DVH is heuristic or parametric (evidential, Bayesian,
  ensembles, reference-cohort bands) — none gives a distribution-free coverage *guarantee*.
- Conformal prediction gives exactly that, but only under exchangeability; deployment breaks
  exchangeability (scanner/kernel/CBCT variation). We provide conformal DVH intervals **and** map
  where their guarantee fails under a realistic CT-degradation battery.
- Summary of results: nominal coverage on clean data [FILL]; collapse profile per family
  (P4 drops below 80% at L[FILL]; intensity families hold); joint band; weighted-CP does not rescue.

## 2. Related Work

### 2.1 Uncertainty quantification for dose / DVH prediction (all NON-conformal)
- **Deep Evidential Learning** for dose prediction constructs DVH confidence intervals via
  evidential/parametric UQ (Ti et al., *Comput. Biol. Med.* 2024) — benchmarked vs MC-dropout and
  deep ensembles; no coverage guarantee.
- **Reference-cohort error bands**: knowledge-based DVH prediction with ~68% Gaussian-assumption
  bands (Covele/Moore et al., *JACMP* 2021).
- **Fixed-tolerance hit-rates**: knowledge-based checks report whether error falls within a fixed
  10% bound (Nature *Sci. Rep.* 2021) — the logical inverse of conformal (fixes width, not coverage).
- **Bayesian/ensemble propagated pDVHs**: dose-accumulation UQ propagates voxel uncertainty into
  probabilistic DVHs with empirical (Gaussian ±3σ) coverage (arXiv 2606.11012). Closest neighbor on
  the *DVH-coverage concept*, but **not conformal** — no distribution-free finite-sample guarantee.
→ None provides a distribution-free coverage guarantee on DVH metrics; our contribution 1 is clean.

### 2.2 Conformal prediction on image-derived clinical metrics (NOT dose/DVH)
- Organ/lesion **volume** intervals (Lambert et al., MICCAI 2024, arXiv 2407.19938).
- Scalar **radiomic** features via split conformal + CQR (ConRad, arXiv 2607.08084).
- Segmentation-derived **area/organ-size** intervals (COMPASS, ICLR 2026, arXiv 2509.22240).
→ Establishes conformal on image-derived scalars in general; **none targets dose or DVH**. We do not
  claim novelty for the general pattern — only for the dose/DVH target.

### 2.3 Conformal under distribution / covariate shift
- Weighted conformal prediction (Tibshirani et al., 2019) corrects **covariate** shift via
  likelihood-ratio reweighting; Lambert (volume) and COMPASS (area) both recover coverage under
  covariate shift this way.
→ Our perturbations worsen the *conditional* error P(Y|X), which reweighting cannot fix (we show
  this with a weighted-CP baseline). Our novelty is documenting **coverage collapse under CT
  perturbations for DVH**, not covariate-shift conformal per se.

### 2.4 Conformal in radiotherapy (segmentation, not dose)
- U-Net auto-contouring uncertainty via adaptive prediction sets (pixel-level), e.g. PMC13291182 —
  target is contours, not dose/DVH.

### 2.5 Not prior art (name-collision / different field)
- arXiv 2409.20412 "Conformal Prediction for Dose-Response Models" — causal/pharmacological drug
  dosing, not radiotherapy.

### Foundations
Median (percentile) randomised smoothing (Chiang et al., 2020); split-conformal coverage
(Vovk et al.; Lei et al., 2018); weighted conformal (Tibshirani et al., 2019); CV+/Jackknife+
(Barber et al., 2021).

## 3. Method (pointer)
Split/CV+ conformal on normalized per-criterion residuals; marginal q_j and joint max-score band Q
(tighter than Bonferroni, which is infeasible at n≤100); coverage-vs-severity over the P1–P5 battery.
Full method + validity in `EXPERIMENT_PLAN.md`; engine in `conformal_dvh.py`.

## To-do before submission
- [ ] Fill coverage numbers from the pod run.
- [ ] Coverage-vs-severity figure + threshold table (`conformal_plots.py`).
- [ ] Re-run novelty search (2026 preprints move fast); confirm arXiv IDs resolve.
- [ ] Verify exact venues/years for Lambert / ConRad / COMPASS / Ti / Covele at write-up.
