# Conformal DVH intervals + coverage stress-test — experiment plan

## The claim (why this is a paper, not an exercise)
Distribution-free, finite-sample **conformal prediction intervals on DVH metrics** ("D95 lies in
[a,b] with 90% coverage, guaranteed"), then the twist: conformal coverage holds only under
exchangeability and is known to collapse under distribution shift. We own a 26-condition CT
perturbation battery (P1–P5), so we answer a question nobody has for dose prediction — **at what
CT-degradation severity does the 90% guarantee actually break** — and complete a guarantee
triptych on one model + one patient cohort:

| Guarantee | Status | What it's worth |
|---|---|---|
| Empirical robustness (EOT defense) | holds | strong, but unprovable |
| Adversarial certificate (median smoothing) | provable | clinically vacuous radius (~0.03 HU) |
| **Conformal coverage (this)** | statistical | informative width, but only under exchangeability |

The honest accounting of what each buys — and where each fails — is the contribution.

## Novelty (settled 2026-09-07 — deep lit search, 25 claims adversarially verified)
**Verdict: "first conformal DVH intervals on a radiotherapy dose predictor" is DEFENSIBLE for the
exact task** — no source applies conformal prediction to dose/DVH metrics of a dose predictor.
Two guardrails:
- **Do NOT** claim broad novelty for "conformal on image-derived clinical metrics" — already done
  on organ/lesion **volume** (Lambert et al., MICCAI 2024, arXiv 2407.19938), **radiomic features**
  (ConRad, arXiv 2607.08084, split-conformal + CQR), and **segmentation area** (COMPASS, ICLR 2026,
  arXiv 2509.22240). None target dose/DVH.
- **Do NOT** claim "first conformal under distribution shift in medical imaging" — Lambert and
  COMPASS both handle **covariate shift** via weighted-CP reweighting. Position our twist as
  **"first study of conformal DVH coverage COLLAPSE under CT / adversarial perturbations."**

Nearest neighbors that are NOT prior art (verified):
- arXiv 2409.20412 (conformal dose-response) = causal/pharmacological drug dosing, not radiotherapy.
- arXiv 2606.11012 (dose-accumulation DVH coverage) = Bayesian/ensemble propagated pDVHs, **not**
  conformal; reports empirical (Gaussian ±3σ) coverage, no distribution-free guarantee.
- Radiotherapy conformal exists only for **segmentation/contouring** (pixel sets, PMC13291182), not dose.
- All DVH-UQ in RT (Deep Evidential Learning, reference-cohort ±68% bands, MC-dropout/ensembles) is
  non-conformal → not prior art for the conformal claim.

**Reviewer defense to pre-empt:** show the coverage-collapse gap is distinct from covariate-shift
recovery — add a **weighted-CP reweighting baseline** (Lambert/COMPASS style) and show it does NOT
rescue coverage under our CT perturbations (or by how much). **Re-run the search at submission** —
fast-moving area with 2026 preprints.

## Design (validated locally; see `conformal_dvh.py --self-test`)
- **Split / CV+ conformal.** Calibrate on CLEAN, train-disjoint patients; the calibrator is fixed.
  Coverage is then measured as the TEST distribution shifts — widths are frozen, so shift shows up
  as **coverage dropping**, the conformal failure signature. Data pools are small (40 held-out
  locally); use **CV+/Jackknife+** if a larger clean pool (official 100-patient test set) isn't
  available, to avoid wasting calibration data. `--cal-frac` controls the split.
- **Nonconformity score (v1):** normalized absolute residual sⱼ = |m̂ⱼ − mⱼ| / σⱼ, σⱼ = robust MAD
  scale per criterion, so a 1 Gy parotid-mean error and a PTV70 D95 error aren't pooled naively.
- **Multiplicity — the make-or-break decision (23 criteria/patient):** report BOTH
  (a) **marginal** per-criterion 90% intervals (interpretable) and
  (b) a **joint** per-patient band via one max-normalized score → simultaneous coverage over all 23
  criteria. This is the honest analog of the parked collective-DVH certificate; state the width cost.
- **Extension (name, don't build v1):** CQR for adaptive intervals (needs an auxiliary quantile
  model — no native quantile output); weighted/covariate-shift conformal as a partial fix under
  shift (density-ratio estimation on 128³ CTs is hard → future work).

## Experiment grid (inference-only, no training)
1. Calibrate at 90% on clean calibration patients.
2. For each of the 26 conditions, compute empirical coverage on the test patients — marginal
   (per criterion) and joint (per patient). `conformal_dvh.py` emits `coverage_vs_severity.csv`.
3. Figures mirror ASTRO: coverage-vs-severity curves (5 family lines + 90% nominal line) and a
   threshold table — **"severity at which the 90% guarantee falls below 80%," per family.**
4. Predicted result: **P4 (resolution) coverage collapses first**, cross-validating the ASTRO
   finding through an independent lens; intensity families stay near nominal.

## Status
- ✅ **Enabler patch** — `evaluate_metrics.py` now persists `per_patient_dvh` (per-patient,
  per-criterion signed residual). Zero GPU: the numbers were already computed, only the cohort mean
  was being saved.
- ✅ **Engine** — `conformal_dvh.py`: split-conformal quantile, MAD normalization, marginal + joint
  coverage, coverage-vs-severity CSV. Coverage math validated by `--self-test` (clean 0.90; under a
  2.2× shift marginal → 0.55, joint → 0.007).
- ⏳ **Real numbers** — blocked on regenerating the metrics with the enabler patch (needs the
  prediction CSVs + ground truth on the pod/SanDisk; predictions aren't local). Once `per_patient_dvh`
  exists in the metric JSONs, `python conformal_dvh.py` produces the coverage table with no GPU.
- ✅ **Novelty check DONE** (2026-09-07) — see the Novelty section above. "First conformal DVH
  intervals on a dose predictor" is clean; frame the twist as "coverage collapse under CT
  perturbations," add a weighted-CP baseline, and re-run the search at submission.

## Next actions
1. ✅ Lit search done (see Novelty section).
2. Regenerate metrics with the enabler patch (re-run `evaluate_metrics.py` over the existing
   prediction CSVs — no new inference if predictions are still on disk).
3. Run `conformal_dvh.py`, add the coverage-vs-severity figure + threshold table, write up as the
   third leg of the triptych.
4. ✅ **Weighted-CP baseline implemented** (`weighted_threshold`/`weighted_coverage`) + validated:
   weighted CP recovers *covariate* shift (0.84→0.93) but NOT *response* shift (0.64→0.64) — CT
   perturbations worsen P(Y|X), which reweighting provably can't fix. This is the reviewer pre-empt.
   Also added a **Bonferroni joint baseline**: max-score joint is tighter at equal validity
   (Q≈2.88 vs 2.93) and Bonferroni is **infeasible at realistic n≤100** (needs the ~99.6th
   percentile of ≤100 points → +inf) — an independent argument for the max-score band.
5. Re-run the novelty search immediately before submission (2026 preprints are appearing fast).
