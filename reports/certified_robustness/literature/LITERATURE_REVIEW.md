# Literature Review — Certified Robustness for Radiotherapy Dose Prediction

Cited gap analysis (deep-research harness, 5 angles, 21 primary sources fetched, 25 claims
adversarially verified 3-vote, 19 confirmed / 6 refuted). Generated 2026-07-31. **Every claim
below carries a citation; refuted sub-claims are listed at the end so we never cite them.**

## Gap (one sentence)

**No published work provides certified / provable adversarial robustness (randomized smoothing,
interval bound propagation, or Lipschitz certification) for radiotherapy dose prediction with
clinically-meaningful DVH guarantees.** Nearest neighbours either only *empirically characterize*
dose-prediction vulnerability, or *certify adjacent tasks* (object detection, camera pose, 2D
medical segmentation) that never touch volumetric dose regression.

## Axis 1 — Novelty check (certified robustness in dose prediction)

- **CONFIRMED (3-0): the intersection is unclaimed.** The authoritative survey — Li, Xie, Li,
  *SoK: Certified Robustness for Deep Neural Networks*, IEEE S&P 2023 (arXiv:2009.04131) — frames
  the field around classification + robust training and never surfaces dose prediction or
  volumetric medical regression. **Caveat (refuted 0-3):** do NOT argue absence from the survey's
  omission alone; argue it from direct examination of the dose-prediction papers below.

## Axis 2 — Certified robustness for regression / dense prediction

- **CONFIRMED (3-0): classification smoothing does not transfer to regression; median smoothing is
  the canonical fix.** Cohen, Rosenfeld, Kolter, *Certified Adversarial Robustness via Randomized
  Smoothing*, ICML 2019 (PMLR v97) gives the classification certificate R=(σ/2)(Φ⁻¹(p_A)−Φ⁻¹(p_B)).
  Chiang et al., *Detection as Regression: Certified Object Detection with Median Smoothing*,
  NeurIPS 2020 (arXiv:2007.03730): "to enable certified regression, where standard mean smoothing
  fails, we propose median smoothing." **This is our methodological template** — but it is scoped
  to object detection / bounding boxes, NOT dose or volumetric medical regression.
- **CONFIRMED (3-0): adjacent certified regression/segmentation exists but never touches dose.**
  *Certified Adversarial Robustness via Randomized α-Smoothing for Regression Models*, NeurIPS 2024
  — benchmark is visual positioning / camera pose (self-driving), not medical imaging.
  Laousy et al., *Certification of Deep Learning Models for Medical Image Segmentation*, MICCAI
  2023 (arXiv:2310.03664) — certified Dice via randomized smoothing + diffusion, but on 2D chest
  X-ray / skin / colonoscopy, not volumetric CT or dose. **Caveat (refuted 0-3):** do NOT amplify
  either paper's "first" claim (prior regression-smoothing and certified-segmentation work exists);
  cite them only as existence proofs of the technique in adjacent domains.

## Axis 3 — Empirical adversarial robustness of dose prediction

- **CONFIRMED (3-0): Gao et al. 2025 is the nearest neighbour — characterization only, no
  mitigation, no certificate.** Gao, Mody, Rao, Dankers, Staring, *On factors that influence deep
  learning-based dose prediction of head and neck tumors*, Phys. Med. Biol. 70(11), DOI
  10.1088/1361-6560/adcfeb (2025). Studies PGD (ε=16 HU) and MI-FGSM plus Poisson noise (λ=20)
  across DoseNet, HDUNet, SwinUNETR, U-NAS, C3D, DOSE-PYFER; Poisson degradation 0–0.3 Gy vs
  adversarial 0.2–7.8 Gy; SwinUNETR most resilient; noise at inference, no retraining.
  **Framing caveat:** robustness is one of several factors (resolution, loss, architecture,
  efficiency), not the paper's thesis — cite as the empirical precedent, not a robustness-first paper.
- **CONFIRMED (3-0): other "robust" dose-prediction work is scenario optimization, not adversarial
  certification.** Eriksson & Bokrantz 2022, *Robust automated radiation therapy treatment planning
  ...*, Med. Phys., DOI 10.1002/mp.15622: "robust" = physical setup (0.5 cm) and range/density
  (±3%) over 45 scenarios via robust optimization — no smoothing/IBP/Lipschitz certificate. Poel
  et al. 2023 (glioblastoma DL dose prediction, PMC10486555): robustness assessed empirically via
  worst-case test set + manual contour variation, no certificate. **Action:** explicitly define
  adversarial/certified robustness vs scenario robust-optimization up front to prevent conflation.

## Axis 4 — Realistic CT noise magnitudes (threat-model check) — **corrective finding**

- **CONFIRMED (3-0): benign CT-number variability is far below our epsilons.** Kim et al.,
  *Tolerance levels of CT number to electron density table...*, PMC5768003: 20-month single-scanner
  CT-number constancy (n=375) SDs — water 1.5 HU, air 2.6 HU, lung 3.0 HU, 50% bone 4.9 HU. Lamba
  et al., AJR 2014, DOI 10.2214/AJR.12.10037 (GE vs Siemens, 48 patients): inter-scanner soft-tissue
  offsets liver −2.9, spleen −6.6, anterior fat +10.5 HU (range −6.6 to +10.5), all p<0.05.
  **Conclusion for the paper:** ε=0.01 (~41 HU) already exceeds the largest benign offset (~11 HU),
  so 41–205 HU are **adversarial worst-cases, NOT typical scanner noise** — defensible as a
  worst-case threat model, indefensible as "representative noise." Lamba stress small differences
  can still be clinically meaningful for narrow-threshold tasks, so don't over-claim benign
  variation is negligible either.
- **NOT SUPPORTED (mixed / refuted): the "10–50 HU" quantum-noise figure and the fine magnitudes.**
  A water-phantom QC noise ≈4.85–4.94 HU claim was refuted (0-3); a tight CT-to-electron-density
  bound was split (1-2). Metal/dental-artifact magnitudes, reconstruction-kernel variability, and
  slice-thickness HU shifts were requested but **no surviving primary source quantifies them.**
  **Action:** do NOT use the "10–50 HU" phrasing (incl. in our own CLAUDE.md) without a dedicated
  citation; obtain primary metal-artifact / kernel-variability phantom studies before asserting them.
- **σ=0.1 (~410 HU) smoothing defense** is defensible ONLY as a certification mechanism (randomized
  smoothing operates at large σ by construction), never as a claim about realistic image content —
  state this distinction explicitly.

## Axis 5 — Adjacent framing (certified vs UQ/conformal)

- **CONFIRMED (definitional): a certified DVH interval ≠ a UQ/conformal DVH interval.** The certified
  guarantee (Cohen 2019; Chiang 2020) is a deterministic per-instance worst-case bound over an
  L2 perturbation ball; UQ/conformal intervals give marginal/probabilistic coverage over a data
  distribution. Different guarantee semantics → defensible novelty axis. **Caveat:** no surviving
  claim located a specific evidential-DL / conformal-prediction paper producing *DVH* confidence
  intervals — the UQ-for-dose landscape is UNVERIFIED here; search before naming a competitor.

## Reviewer caveats to pre-empt (from the search)

1. **Small certified radius in high dimensions** — a 128³ volumetric model yields small L2 radii;
   report the radius honestly, in HU-equivalent terms (both per-voxel RMS and single-voxel max).
2. **L2 vs structured perturbations** — smoothing certifies L2 balls; clinically-realistic CT
   variation (kernel, slice-thickness/resolution, bone/metal, setup deformation) is structured and
   not an L2 ball. Acknowledge explicitly; this is exactly our P2/P4 gap.
3. **Noise realism** — present 41–205 HU as adversarial worst-cases, not typical noise; get
   citations for any HU-magnitude claim (§Axis 4).
4. **σ=0.1 is a mechanism, not realism** — state it.
5. **Terminology** — "robust" collides with RT robust-optimization; define our sense up front.

## Open questions (need follow-up before submission)

- Is there a specific evidential-DL / conformal DVH-interval paper for dose prediction to contrast against? (Angle 5 unresolved.)
- Citable HU-deviation figures for metal/dental artifacts, kernel variability, slice-thickness. (Angle 4 unresolved.)
- Any arXiv/medRxiv preprint after the 2023 SoK / Gao 2025 that already certifies dose prediction? (Near-submission scan to protect "first".)
- Is the L2→DVH-metric translation itself novel/defensible (vs a voxel-wise bound only)?

## Sources (primary unless noted)

- Li, Xie, Li — SoK: Certified Robustness for DNNs, IEEE S&P 2023 — https://arxiv.org/pdf/2009.04131
- Cohen, Rosenfeld, Kolter — Randomized Smoothing, ICML 2019 — https://proceedings.mlr.press/v97/cohen19c/cohen19c.pdf
- Chiang et al. — Detection as Regression (median smoothing), NeurIPS 2020 — https://arxiv.org/abs/2007.03730
- α-Smoothing for Regression Models, NeurIPS 2024 — https://proceedings.neurips.cc/paper_files/paper/2024/hash/f21a76d688be0553c943a6b6c1d4bb1f-Abstract-Conference.html
- Laousy et al. — Certified Medical Image Segmentation, MICCAI 2023 — https://arxiv.org/abs/2310.03664
- Gao et al. — DL dose prediction factors (Phys. Med. Biol. 2025) — https://iopscience.iop.org/article/10.1088/1361-6560/adcfeb | https://pubmed.ncbi.nlm.nih.gov/40267938/
- Eriksson & Bokrantz — Scenario robust dose prediction, Med. Phys. 2022 — https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.15622
- Poel et al. — Glioblastoma DL dose robustness, 2023 — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10486555/
- Kim et al. — CT number tolerance / constancy — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5768003/
- Lamba et al. — Inter-scanner CT number variability, AJR 2014 — https://www.ajronline.org/doi/10.2214/AJR.12.10037

## Refuted — DO NOT CITE these framings

- "SoK survey omission proves absence" (0-3).
- "α-smoothing was the FIRST regression smoothing" (0-3).
- "Laousy was the FIRST certified medical segmentation" (0-3).
- Water-phantom QC noise ≈4.85–4.94 HU / "10–50 HU too high" (0-3).
- Tight CT-to-electron-density bound <5 HU/material (1-2).
