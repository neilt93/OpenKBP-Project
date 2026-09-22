# ASTRO #79011 — complete data & findings handoff

Self-contained package for building the poster (or paper). Everything needed is below; no repo
access required. Study = robustness of a deep-learning head-and-neck dose predictor to five
families of clinically realistic CT perturbations.

---

## 1. Poster identity
- **Official title (submitted to ASTRO, abstract LOCKED — use this verbatim):** "Assessing the Generalizability and Robustness of Deep-Learning Dose Prediction in Head-and-Neck Radiotherapy to Clinically Realistic CT Perturbations"
- (A shortened variant, "Robustness of Deep-Learning Dose Prediction in Head-and-Neck Radiotherapy to Clinically Realistic CT Perturbations," was used on an earlier draft — prefer the official title above.)
- **Abstract #:** 79011 (ASTRO 2026 ePoster; PQA 05: Physics; Tue Sept 29). Digital kiosk ePoster; 7-min slot; interactive element encouraged. Upload deadline (final): Monday.
- **Authors:** Neil Tripathi¹, Rahim Chowdhury², Lei Ren³, Amit Sawant³, Birjoo Vaishnav³ (presenter of record). ¹New York University; ²UMD St. Joseph Medical Center; ³University of Maryland School of Medicine, Dept. of Radiation Oncology.
- **Format requested by presenter:** the AAPM poster style (44"×44" square, section-header bars: ABSTRACT, INTRODUCTION, METHODS, RESULTS, DISCUSSION, CONCLUSIONS, FUTURE WORK, REFERENCES, CONTACT). No vendor/commercial names (ACCME).

## 2. Model & data
- **Model:** 3D U-Net with squeeze-and-excitation blocks; masked mean-absolute-error loss; 4× planning-target-volume (PTV) weighting. Baseline accuracy: **DVH score 2.535, dose score 3.731 Gy**.
- **Dataset:** OpenKBP head-and-neck cohort (Babier et al., Med Phys 2021). 200 training / 40 test patients (test = pt_201–240). 128×128×128 CT; 10 OAR/PTV structures (Brainstem, SpinalCord, L/R Parotid, Esophagus, Larynx, Mandible, PTV56, PTV63, PTV70); dose normalized to 70 Gy prescription. 23 DVH criteria total (per-structure D_0.1cc and mean for OARs; D_99/D_95/D_1 for PTVs).
- **Evaluation:** perturbations applied to the **test-input CT only** (structures/dose unchanged); model re-predicts; metrics computed vs each patient's **own unperturbed prediction** (self-referential ΔDVH, ΔMAE). 26 conditions = baseline + 5 families × 5 levels (P4 uses L0–L4; others L1–L5).

## 3. The five perturbation families and severity parameters
(Ranges chosen to meet or exceed ACR CT-simulation QA thresholds.)

| Family | Physical model | Level parameters (L1→L5; P4 L0→L4) |
|---|---|---|
| **P1 acquisition noise** | heteroscedastic Gaussian, higher σ in bone | σ_soft/σ_bone HU: 8/12, 15/25, 30/50, 60/100, 100/160 |
| **P2 HU calibration shift** | bone-weighted sigmoid HU shift (± sign) | Δ_soft/Δ_bone HU: 5/50, 10/100, 25/250, 50/500, 100/1000 |
| **P3 bias field** | low-frequency cosine-harmonic intensity field | amplitude HU: 10, 20, 50, 100, 200 |
| **P4 spatial-resolution loss** | anisotropic Gaussian blur (stronger in z) | σ_z/σ_xy vox: 0.5/0.25, 1.0/0.5, 2.0/1.0, 3.0/1.5, 4.0/2.0 |
| **P5 dental streak artifact** | radial streaks from mandible centroid + metal spot | amplitude HU / n_streaks: 150/8, 300/12, 500/16, 800/20, 1200/24 |

Clinical reference ranges (for the "vs clinical range" comparison): typical single-scanner noise ≈10–50 HU; inter-scanner HU calibration drift ≈10–50 HU; realistic cross-scanner slice-thickness / reconstruction-kernel variation ≈1–3 voxels of effective blur.

## 4. Threshold criterion (state once, use everywhere)
A severity level is **"clinically visible"** when the **cohort-mean change in any DVH criterion, relative to each patient's unperturbed prediction, exceeds 1.0 Gy**. Corroborating metric (reported, not the gate): fraction of the 40 patients whose voxel-wise dose MAE rises >0.5 Gy. (Gamma index was not computed — do not mention it.)

## 5. Full quantitative results

### 5a. Aggregate scores per condition (40 patients)
`delta_dvh` / `delta_dose` = change in the OpenKBP DVH-score / dose-score (Gy) vs baseline; `%` = relative.

| Condition | dose_score | dvh_score | Δdose (Gy) | Δdvh (Gy) | Δdose % | Δdvh % |
|---|---|---|---|---|---|---|
| baseline | 3.731 | 2.535 | — | — | — | — |
| P1_noise/L1 | 3.731 | 2.535 | 0.001 | 0.000 | 0.01 | 0.01 |
| P1_noise/L2 | 3.733 | 2.538 | 0.002 | 0.003 | 0.07 | 0.13 |
| P1_noise/L3 | 3.737 | 2.543 | 0.006 | 0.009 | 0.16 | 0.34 |
| P1_noise/L4 | 3.742 | 2.559 | 0.011 | 0.024 | 0.29 | 0.96 |
| P1_noise/L5 | 3.742 | 2.571 | 0.011 | 0.036 | 0.29 | 1.43 |
| P2_bone_shift/L1 | 3.730 | 2.531 | −0.001 | −0.004 | −0.02 | −0.15 |
| P2_bone_shift/L2 | 3.740 | 2.544 | 0.010 | 0.010 | 0.26 | 0.38 |
| P2_bone_shift/L3 | 3.751 | 2.519 | 0.020 | −0.016 | 0.54 | −0.61 |
| P2_bone_shift/L4 | 3.793 | 2.572 | 0.062 | 0.037 | 1.67 | 1.46 |
| P2_bone_shift/L5 | 3.916 | 2.818 | 0.186 | 0.284 | 4.97 | 11.19 |
| P3_bias_field/L1 | 3.732 | 2.535 | 0.001 | 0.000 | 0.03 | 0.01 |
| P3_bias_field/L2 | 3.730 | 2.534 | −0.000 | −0.001 | −0.01 | −0.04 |
| P3_bias_field/L3 | 3.736 | 2.534 | 0.006 | −0.000 | 0.15 | −0.01 |
| P3_bias_field/L4 | 3.733 | 2.524 | 0.002 | −0.011 | 0.06 | −0.44 |
| P3_bias_field/L5 | 3.759 | 2.545 | 0.028 | 0.011 | 0.76 | 0.42 |
| P4_resolution/L0 | 3.739 | 2.573 | 0.008 | 0.038 | 0.21 | 1.50 |
| P4_resolution/L1 | 3.780 | 2.636 | 0.049 | 0.101 | 1.32 | 4.00 |
| **P4_resolution/L2** | 3.885 | 2.729 | 0.154 | 0.194 | 4.14 | 7.67 |
| **P4_resolution/L3** | 3.982 | 2.824 | 0.252 | 0.289 | 6.74 | 11.40 |
| **P4_resolution/L4** | 4.124 | 2.995 | 0.393 | 0.460 | 10.55 | 18.15 |
| P5_dental/L1 | 3.729 | 2.541 | −0.002 | 0.007 | −0.05 | 0.26 |
| P5_dental/L2 | 3.730 | 2.545 | −0.001 | 0.011 | −0.03 | 0.42 |
| P5_dental/L3 | 3.730 | 2.552 | −0.000 | 0.017 | −0.01 | 0.67 |
| P5_dental/L4 | 3.731 | 2.557 | 0.000 | 0.022 | 0.01 | 0.87 |
| P5_dental/L5 | 3.732 | 2.555 | 0.002 | 0.021 | 0.04 | 0.82 |

### 5b. Worst per-criterion DVH shift (the threshold analysis — this drives the headline)
Maximum absolute change on **any single DVH criterion** (cohort mean, Gy), the sentinel criterion, and how many criteria exceed 1.0 Gy.

| Condition | max criterion shift (Gy) | sentinel criterion | # criteria >1 Gy | frac patients dose-MAE >0.5 Gy |
|---|---|---|---|---|
| P1_noise L5 (worst P1) | 0.14 | SpinalCord D0.1cc | 0 | 0.00 |
| P2_bone_shift L4 | 0.41 | Larynx D0.1cc | 0 | 0.03 |
| P2_bone_shift L5 | 0.61 | PTV56 D1 | 0 | 0.17 |
| P3_bias_field (worst) | 0.09 | RightParotid mean | 0 | 0.00 |
| **P4_resolution L0** | 0.22 | Larynx D0.1cc | 0 | 0.00 |
| **P4_resolution L1** | 0.59 | Larynx D0.1cc | 0 | 0.00 |
| **P4_resolution L2** | **1.15** | **Larynx D0.1cc** | **1** | 0.05 |
| **P4_resolution L3** | **1.68** | Larynx D0.1cc | 1 | 0.12 |
| **P4_resolution L4** | **2.84** | Larynx D0.1cc | 2 | 0.23 |
| P5_dental (worst) | 0.33 | PTV70 D1 | 0 | 0.00 |

**Key result:** P4 (resolution) is the ONLY family whose worst DVH criterion crosses 1.0 Gy in the tested range — it crosses at **L2 (2.0/1.0-voxel blur)** and rises to **2.84 Gy at L4**. The sentinel is **larynx near-maximum dose (D0.1cc)**. Every other family stays below 1 Gy on every criterion, even at extreme severity (P2 only reaches 0.61 Gy at an implausible 1000 HU offset).

## 6. Threshold table (poster centerpiece — color Verdict green/green/green/red/green)

| Family | Sweep tested | Clinically visible at (>1 Gy any DVH criterion) | vs. clinical range | Verdict |
|---|---|---|---|---|
| P1 noise | 8/12 → 100/160 HU | never (max 0.14 Gy) | scanner noise ≈10–50 HU | Robust |
| P2 HU shift | 5/50 → 100/1000 HU | never (max 0.61 Gy at 1000 HU) | calibration drift ≈10–50 HU | Robust in clinical range |
| P3 bias field | 10 → 200 HU | never (max 0.09 Gy) | RF/scatter, tens of HU | Robust |
| **P4 resolution** | 0.5/0.25 → 4.0/2.0 vox | **L2 (2.0/1.0 vox): 1.15 Gy on larynx D0.1cc** | slice/kernel variation ≈1–3 vox | **SENSITIVE — the failure mode** |
| P5 dental streak | 150/8 → 1200/24 | never (max 0.33 Gy) | streaks common in H&N | Robust |

## 7. Figures (all PNGs included in the zip; use any subset)
All are the **original figures we generated** for this study, 40 test patients.

**Orientation note (RESOLVES Birjoo's comment).** `ct_slices.png` and `dose_difference_maps.png`
are genuine OpenKBP data (patient pt_201, read from the real `ct.csv` via `load_ct_volume`), but the
"axial slice 64" caption is a **mislabel** — they are actually **coronal** views. The generator
sliced volume axis 0, and OpenKBP's raw axes are (A-P, L-R, S-I) (verified in
`provided_code/network_functions.py`), so an axis-0 slice is the L-R×S-I (coronal) plane — hence the
head-and-shoulders appearance. Fix committed in `visualize_results.py` (`SLICE_PLANE="axial"` now
slices the S-I axis for a true transverse view); **regenerate these two figures with the model +
OpenKBP data to obtain correctly-oriented, correctly-labeled axial panels** (verify in-plane
up/left on first render). All quantitative results are unaffected — only these two illustrative
panels need regeneration.

1. **`ct_slices.png`** — CT perturbation examples: one patient, unperturbed CT + all 5 perturbed CTs at intermediate severity (top row) and difference maps (bottom). Resolution visibly blurs boundaries; intensity families leave edges intact. *(Poster Figure 1.)*
2. **`fig_panel5_maxcrit_gy.png`** — Threshold / severity curve: max per-criterion DVH shift (Gy) vs severity level, five family lines + dashed 1.0 Gy visibility line. P4 crosses between L1 and L2 and climbs; others hug the floor. *(Poster Figure 2 — the money figure.)*
3. **`dose_difference_maps.png`** — predicted dose (top) + difference-from-baseline (bottom) per family; P4 shows spatially structured boundary errors, others near-zero. *(Poster Figure 3.)*
4. **`degradation_curves.png`** — dose-score (Gy) and DVH-score degradation vs severity, per family; Resolution dominant, Bone Shift late take-off at L4–L5.
5. **`degradation_heatmap.png`** — family × severity-level grid of degradation magnitude.
6. **`structure_radar.png`** — per-structure error ratio vs baseline at the highest level; Resolution spreads most on larynx, mandible, parotids.
7. **`summary_bars.png`** — bar chart of dose- and DVH-score across all 26 conditions.
8. **`fig_panel5_dvh_pct.png`** — aggregate DVH-score degradation (%) vs severity (twin of Fig 2).
- (Optional kiosk interactive element) a GIF stepping one patient through P4 L0→L4; script exists but needs the model + data to run (not in this package).

## 8. Poster body text (formal, ready to paste)

**ABSTRACT.** *Purpose:* Deep-learning dose prediction models are increasingly deployed across institutions, where they encounter CT images from different scanners, reconstruction kernels, and acquisition protocols. This study quantifies the robustness of a head-and-neck dose prediction model to five families of clinically realistic CT perturbations and identifies the image-quality factors that materially affect predicted dose. *Methods:* A 3D U-Net dose predictor was evaluated on 40 test patients under 26 CT conditions spanning acquisition noise, HU calibration shift, low-frequency bias field, spatial-resolution loss, and dental streak artifacts, at severities meeting or exceeding ACR CT-simulation quality-assurance limits. A perturbation level was defined as clinically visible when the cohort-mean change in any DVH criterion exceeded 1.0 Gy relative to each patient's unperturbed prediction. *Results:* Four of five families produced no clinically visible change at any tested severity; spatial-resolution degradation was the only family to exceed 1.0 Gy, reaching 2.84 Gy at the larynx (+18.2% DVH score) at the highest severity. *Conclusions:* The model is robust to intensity-based CT variability but sensitive to spatial-resolution loss; multi-institution QA should prioritize CT spatial-resolution and reconstruction-kernel consistency.

**INTRODUCTION.** Deep-learning dose prediction is increasingly used to support automated and adaptive treatment planning. A model trained at one institution must generalize to CT images acquired under different conditions, including variation in scanner hardware, reconstruction kernel, dose level, and imaging artifacts. In contrast to worst-case adversarial perturbations, these represent benign, physically plausible sources of image variability. This study characterizes the sensitivity of a head-and-neck dose prediction model to five families of clinically realistic CT perturbations and determines the severity at which each produces clinically significant changes in predicted dose, in order to identify the CT image-quality factors relevant to safe multi-institution deployment.

**METHODS.** *Dataset:* OpenKBP head-and-neck cohort (Babier et al., 2021); 200 training and 40 test patients; 128³ CT; ten OAR/target structures; dose normalized to a 70 Gy prescription. *Model:* 3D U-Net with squeeze-and-excitation blocks, masked mean-absolute-error loss, 4× PTV weighting (baseline DVH score 2.54, dose score 3.73 Gy). *Perturbations:* five families applied to the test-input CT at five severity levels each (26 conditions), ranges meeting or exceeding ACR CT-simulation QA. *Analysis:* impact quantified as change in DVH metrics and voxel-wise MAE vs each patient's unperturbed prediction; clinically visible = cohort-mean shift on any DVH criterion > 1.0 Gy.

**RESULTS.** Four of five families left every DVH criterion below the 1.0 Gy threshold across the full tested range. Spatial-resolution loss (P4) was the sole family to exceed it: larynx near-maximum dose crossed 1.0 Gy at L2 (2.0/1.0-voxel blur) and reached 2.84 Gy at L4 (+18.2% DVH score). Bone-weighted HU calibration shift became measurable only at an implausible 1000 HU offset (0.61 Gy, below threshold). Acquisition noise, bias field, and dental artifacts did not exceed 0.34 Gy on any criterion, even beyond ACR-level severities.

**DISCUSSION.** The model was robust to intensity-based CT variability at severities beyond typical clinical ranges because these perturbations preserve the structural boundaries the network uses to localize anatomy. Spatial-resolution loss degrades this boundary information, producing spatially structured dose errors that increase approximately linearly with blur and appear first in geometrically complex structures (larynx, parotid glands, mandible). Because realistic inter-scanner differences in slice thickness and reconstruction kernel span the severities at which resolution loss becomes clinically visible, spatial resolution — rather than intensity calibration — is the principal robustness concern for cross-institution deployment.

**CONCLUSIONS.** A head-and-neck deep-learning dose prediction model tolerates clinically realistic intensity-based CT perturbations but is systematically sensitive to spatial-resolution degradation, which becomes clinically visible within the range of realistic inter-scanner variation. Multi-institution QA should prioritize CT spatial-resolution and reconstruction-kernel consistency, and dose-based rather than image-quality metrics should be used to assess CBCT and synthetic-CT suitability.

**FUTURE WORK.** CBCT-characteristic degradations (scatter/cupping, ring artifacts, limited-FOV truncation) to establish commissioning tolerances for online adaptive radiotherapy; distribution-free (conformal) and certified per-patient DVH prediction intervals under bounded CT perturbation; and training-time augmentation to improve resolution robustness.

**REFERENCES.**
1. Babier A, Mahon R, McNiven A, Diamant A, Chan TCY. *Med Phys.* 2021;48:4932-4948 (OpenKBP).
2. Gao Y, et al. *Phys Med Biol.* 2025;70:115006.
3. American College of Radiology. CT Quality Control and Accreditation guidelines.

**CONTACT.** Birjoo Vaishnav, PhD, DABR — University of Maryland School of Medicine, Baltimore, MD — bvaishnav@som.umaryland.edu — 410-427-2039.

## 9. Existing artifacts (in repo, if wanted)
Repo: github.com/neilt93/OpenKBP-Project, branch `adversarial-retraining`.
- Data: `open-kbp-modified/openkbp_hn_robustness/metrics/` (summary.csv, per_patient/*.json).
- Figures: `open-kbp-modified/openkbp_hn_robustness/figures/*.png`.
- Built poster: `reports/report2_ct_perturbation/astro_poster/ASTRO_79011_poster.{pptx,pdf}` (+ preview PNG).
- Prior full write-up: `reports/report2_ct_perturbation/ct_perturbation_robustness_report.pdf`.
- Template/style reference (the presenter-requested format): `AdversarialAAPM.pptx` (prior FGSM/PGD poster).

## 10. Notes / caveats
- The interactive GIF and any new figures need the model + patient data (not in git; on RunPod/SanDisk) — inference only, no training.
- `delta_dvh` in summary.csv is the change in the aggregate OpenKBP DVH *score* (mean-abs-error over criteria); the **1 Gy threshold is applied to individual DVH criteria** (§5b), where P4 crosses at L2. Do not conflate the two.
- Adversarial (FGSM/PGD) robustness is a **separate** study (the AAPM poster / SERA paper) — keep this poster scoped to the realistic CT perturbations.
