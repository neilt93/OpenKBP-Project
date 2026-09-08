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
- ⏳ **Novelty check** — settle "first conformal DVH intervals" before anchoring the framing on
  "first"; conformal in medical imaging is moving fast. If prior art exists, reframe to "first under
  distribution shift," which still stands. (Task tracked below.)

## Next actions
1. Lit search: conformal / split-conformal prediction on dose prediction, DVH metrics, or
   segmentation-derived clinical metrics. Decide "first" vs "first under shift."
2. Regenerate metrics with the enabler patch (re-run `evaluate_metrics.py` over the existing
   prediction CSVs — no new inference if predictions are still on disk).
3. Run `conformal_dvh.py`, add the coverage-vs-severity figure + threshold table, write up as the
   third leg of the triptych.
