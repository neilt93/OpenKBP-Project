# ASTRO #79011 poster: build instructions and data package (for an LLM)

This package contains everything needed to build the ASTRO 2026 ePoster #79011. No repository access
is required. Read section 0 first. Do not use em dashes anywhere in the poster text.

## 0. Instructions for the LLM building the poster
1. Use the ASTRO 2026 ePoster template (16:9 landscape). Match its fonts and color scheme.
2. Use the official title verbatim (section 1). The abstract is locked and publishes as submitted.
3. Sections to include: Introduction/Purpose, Methods and Materials, Results, Discussion,
   Conclusions, Future Work, References, Contact. Optionally an Acknowledgements line.
4. Fill the whole page. Do not leave large empty areas. Use a multi column layout, place figures at
   a readable size, and add a severity table (section 4) as its own panel so the page is full.
5. Figures to place (files in this package):
   a. CT perturbation examples (`ct_slices_axial.png`).
   b. Threshold / severity curve (`fig_panel5_maxcrit_gy.png`), the key quantitative figure.
   c. Dose difference maps (`dose_difference_axial.png`).
   d. Resolution severity progression (`p4_ct_progression.png` and `p4_dose_progression.png`).
   e. DVH curves under resolution loss (`dvh_Larynx.png`, `dvh_PTV70.png`).
   f. Optional: `degradation_curves.png`, `structure_radar.png`, `summary_bars.png`.
6. Orientation of the CT/dose panels: all provided CT and dose panels are true axial (transverse)
   cross-sections, already verified. See section 6.
7. Language must be formal and scientific. No em dashes. Use commas, periods, or parentheses.
8. No vendor or commercial names anywhere (ACCME rule).
9. Voice and structure: reuse the language, tone, and section structure of the group's prior AAPM
   poster (full text in `AAPM_POSTER_TEXT.md`). This ASTRO study is the follow-up announced in that
   poster's future-work line; the only substantive change is that the perturbations are now
   clinically realistic physical models of CT variability rather than adversarial attacks. Mirror
   its phrasing patterns (for example the "Key Finding:" callout in Results and the short
   declarative bullets in Conclusions).
10. The poster must explain, in plain language, what a perturbation family and a severity level
    are before presenting results. Use the explanation paragraph at the top of section 4 verbatim
    or near-verbatim; do not assume the reader knows what "L2" or "severity" means.

## 1. Title, authors, contact
- Official title (use verbatim): "Assessing the Generalizability and Robustness of Deep-Learning
  Dose Prediction in Head-and-Neck Radiotherapy to Clinically Realistic CT Perturbations"
- Author block (exact, from the submitted abstract; use verbatim):
  N. Tripathi(1), R. Chowdhury(2), L. Ren(3), A. Sawant(3), and B. D. Vaishnav(3);
  (1) Department of Computer Science, New York University, New York, NY;
  (2) University of Maryland Medical System, St. Joseph Medical Center, Baltimore, MD;
  (3) Department of Radiation Oncology, University of Maryland, School of Medicine, Baltimore, MD.
  Presenter of record: Birjoo Vaishnav.
- Contact: Birjoo Vaishnav, PhD, DABR, University of Maryland School of Medicine, Baltimore, MD.

## 2. Model and data
- Model: a 3D U-Net with squeeze-and-excitation blocks, trained with a masked mean-absolute-error
  loss and 4x planning-target-volume weighting. Baseline accuracy: DVH score 2.54, dose score 3.73 Gy.
- Data: OpenKBP head-and-neck cohort (Babier et al., Med Phys 2021). 200 training and 40 test
  patients. Each patient has a 128 x 128 x 128 CT, ten organ-at-risk and target structures, and dose
  normalized to a 70 Gy prescription. There are 23 standard DVH criteria in total.
- Evaluation: each perturbation is applied to the test-input CT only (structures and reference dose
  unchanged). The model re-predicts, and impact is measured against each patient's own unperturbed
  prediction.

## 3. Perturbation families
Five families of clinically realistic CT perturbations, each at five severity levels. Parameter
ranges meet or exceed ACR CT-simulation quality-assurance limits.
- P1 acquisition noise: heteroscedastic Gaussian noise, higher in bone than soft tissue.
- P2 HU calibration shift: bone-weighted systematic HU offset (models scanner drift or cross-scanner
  variation).
- P3 bias field: low-frequency cosine-harmonic intensity variation (models scatter or RF
  non-uniformity).
- P4 spatial-resolution loss: anisotropic Gaussian blur, stronger along the slice axis (models
  thicker slices or smoother reconstruction kernels).
- P5 dental streak artifact: radial metal streaks anchored to the mandible.

## 4. Severity quantification and what it means (put this on the poster as a panel)

What a perturbation family and a severity level are (explain this on the poster before any
results): Each perturbation family is a controlled physical model of one real-world source of CT
image variability: quantum noise, scanner calibration drift, low-frequency shading, loss of spatial
resolution, or dental metal artifact. Severity is the strength of that perturbation, expressed in
physical units: Hounsfield units (HU) for the four intensity families, and voxels of Gaussian blur
for spatial resolution. Each family is applied at five increasing severity levels. The levels are
chosen so that the lowest fall within routine clinical variation, the middle levels approach the
tolerance limits of the ACR CT quality assurance program, and the highest intentionally exceed
anything expected in practice, in order to locate the point at which the model fails. Because every
patient is compared against their own unperturbed prediction, the reported change isolates the
effect of the image perturbation alone.

For each level: the physical parameter, a plain-language equivalent, whether it is within the
realistic clinical range, and the resulting DVH-score change (percent, cohort mean over 40 patients).

### P1 acquisition noise (soft/bone Gaussian sigma, HU)
| Level | Parameter | Real-world equivalent | In clinical range? | DVH-score change |
|---|---|---|---|---|
| L1 | 8 / 12 HU | normal-quality scan (typical noise 10 to 50 HU) | yes | +0.01% |
| L2 | 15 / 25 HU | a noisier acquisition | yes | +0.13% |
| L3 | 30 / 50 HU | markedly noisy scan | borderline | +0.34% |
| L4 | 60 / 100 HU | severe, beyond routine practice | no | +0.96% |
| L5 | 100 / 160 HU | extreme | no | +1.43% |

### P2 HU calibration shift (soft/bone offset, HU)
| Level | Parameter | Real-world equivalent | In clinical range? | DVH-score change |
|---|---|---|---|---|
| L1 | 5 / 50 HU | mild calibration drift (typical drift 10 to 50 HU) | yes | -0.15% |
| L2 | 10 / 100 HU | plausible cross-scanner difference | yes | +0.38% |
| L3 | 25 / 250 HU | large miscalibration | no | -0.61% |
| L4 | 50 / 500 HU | implausible | no | +1.46% |
| L5 | 100 / 1000 HU | grossly implausible | no | +11.19% |

### P3 bias field (amplitude, HU)
| Level | Parameter | Real-world equivalent | In clinical range? | DVH-score change |
|---|---|---|---|---|
| L1 to L5 | 10, 20, 50, 100, 200 HU | scatter or RF non-uniformity (tens of HU realistic) | L1 to L2 | +0.01% up to +0.42% |

### P4 spatial-resolution loss (slice/in-plane Gaussian sigma, voxels)
| Level | Parameter | Real-world equivalent | In clinical range? | DVH-score change |
|---|---|---|---|---|
| L0 | 0.5 / 0.25 vox | near-native resolution | yes | +1.50% |
| L1 | 1.0 / 0.5 vox | mild slice-thickness or kernel smoothing | yes | +4.00% |
| L2 | 2.0 / 1.0 vox | thick-slice or smooth-kernel acquisition | yes | +7.67% |
| L3 | 3.0 / 1.5 vox | pronounced resolution loss | borderline | +11.40% |
| L4 | 4.0 / 2.0 vox | severe blur | no | +18.15% |

### P5 dental streak artifact (amplitude HU / number of streaks)
| Level | Parameter | Real-world equivalent | In clinical range? | DVH-score change |
|---|---|---|---|---|
| L1 to L5 | 150/8, 300/12, 500/16, 800/20, 1200/24 | dental metal artifact, common in H&N | L2 to L4 | +0.26% up to +0.87% |

## 5. Threshold criterion and headline result
Criterion (state once, use everywhere): a severity level is clinically visible when the cohort-mean
change in any single DVH criterion, relative to each patient's unperturbed prediction, exceeds 1.0 Gy.

Worst per-criterion DVH shift (cohort mean, Gy) and the sentinel criterion:
| Family (worst level) | Max criterion shift | Sentinel criterion | Crosses 1.0 Gy? |
|---|---|---|---|
| P1 noise (L5) | 0.14 Gy | SpinalCord D0.1cc | no |
| P2 HU shift (L5, 1000 HU) | 0.61 Gy | PTV56 D1 | no |
| P3 bias field (worst) | 0.09 Gy | RightParotid mean | no |
| P4 resolution (L2) | 1.15 Gy | Larynx D0.1cc | yes, first crossing at L2 |
| P4 resolution (L4) | 2.84 Gy | Larynx D0.1cc | yes |
| P5 dental (worst) | 0.33 Gy | PTV70 D1 | no |

Headline: four of five families never reach clinical visibility at any tested severity. Only
spatial-resolution loss does. It first crosses 1.0 Gy at level L2 (a 2.0/1.0-voxel blur, within the
realistic range of cross-scanner slice-thickness and kernel differences), and reaches 2.84 Gy at the
larynx at L4 (a plus 18.2 percent DVH-score change). The sensitive structures are geometrically
complex ones (larynx, parotid glands, mandible).

## 6. Figure orientation (resolved, verified visually)
The OpenKBP volume axes are (anterior-posterior, left-right, superior-inferior). Slicing along the
third axis (Z, superior-inferior) gives the true axial (transverse) cross-section; this was
confirmed visually against `ct_orientation_xyz.png` (X is coronal, Y is sagittal, Z is axial). All
CT and dose panels in this package (`ct_slices_axial.png`, `dose_difference_axial.png`,
`p4_ct_progression.png`, `p4_dose_progression.png`) are true axial slices taken at the
superior-inferior index with the largest planning-target-volume area (slice 53 for pt_201), cropped
to the body. Per the presenter's request the panels are flipped 180 degrees in y (anterior points
down, the mandible is at the bottom of each panel). Use them exactly as provided; no further
orientation change is needed.

## 7. Figures in this package
- `ct_orientation_xyz.png`: original CT sliced along X, Y, Z (orientation reference; Z is axial).
- `ct_slices_axial.png`: original CT plus each perturbation family at level L2, with difference maps.
- `dose_difference_axial.png`: predicted dose plus difference from baseline per family.
- `p4_ct_progression.png`, `p4_dose_progression.png`: resolution loss L0 to L4 (the failure mode).
- `dvh_Larynx.png`, `dvh_PTV70.png`: cumulative DVH curves, baseline vs resolution L0 to L4.
- `fig_panel5_maxcrit_gy.png`: max per-criterion DVH shift vs severity, with the 1.0 Gy line.
- `fig_panel5_dvh_pct.png`: aggregate DVH-score change vs severity.
- `degradation_curves.png`, `degradation_heatmap.png`, `structure_radar.png`, `summary_bars.png`.
- `panel4_threshold_table.md`, `panel4_threshold_table.csv`: the threshold table.

## 8. Ready-to-paste poster text (formal, no em dashes)

Note on the abstract: the officially submitted abstract is locked and publishes as submitted. It
begins "Purpose/Objective(s): The primary goal of this work is to establish a framework for
assessing the generalizability of a deep-learning dose prediction..." and was emailed by the
presenter. If the full submitted abstract text is supplied alongside this package, use it verbatim
for the ABSTRACT panel and use the text below for the remaining panels. Rewrite the panels below in
the voice of `AAPM_POSTER_TEXT.md` (instruction 9 in section 0) while keeping every number
unchanged.

Purpose. Deep-learning dose prediction models are increasingly deployed across institutions, where
they encounter CT images from different scanners, reconstruction kernels, and acquisition protocols.
This study quantifies the robustness of a head-and-neck dose prediction model to five families of
clinically realistic CT perturbations and identifies the image-quality factors that materially
affect predicted dose.

Methods. A 3D U-Net dose predictor was evaluated on 40 test patients under 26 CT conditions spanning
acquisition noise, HU calibration shift, low-frequency bias field, spatial-resolution loss, and
dental streak artifacts, at severities meeting or exceeding ACR CT-simulation quality-assurance
limits. A perturbation level was defined as clinically visible when the cohort-mean change in any
DVH criterion exceeded 1.0 Gy relative to each patient's unperturbed prediction.

Results. Four of five families produced no clinically visible change at any tested severity. Spatial
resolution loss was the only family to exceed the 1.0 Gy threshold. It first crossed the threshold at
level L2, a 2.0/1.0-voxel blur within the realistic range of cross-scanner variation, and reached
2.84 Gy at the larynx at the highest severity, a 18.2 percent DVH-score change. Bone-weighted HU
calibration shift became measurable only at an implausible 1000 HU offset (0.61 Gy, below threshold).
Acquisition noise, bias field, and dental artifacts did not exceed 0.34 Gy on any criterion.

Discussion. The model was robust to intensity-based CT variability, including acquisition noise,
calibration drift, bias field, and dental artifacts, at severities beyond typical clinical ranges,
because these perturbations preserve the structural boundaries the network uses to localize anatomy.
Spatial-resolution loss degrades this boundary information, producing spatially structured dose
errors that increase approximately linearly with blur and appear first in geometrically complex
structures such as the larynx, parotid glands, and mandible. Because realistic inter-scanner
differences in slice thickness and reconstruction kernel span the severities at which resolution loss
becomes clinically visible, spatial resolution rather than intensity calibration is the principal
robustness concern for cross-institution deployment.

Conclusions. A head-and-neck deep-learning dose prediction model tolerates clinically realistic
intensity-based CT perturbations but is systematically sensitive to spatial-resolution degradation,
which becomes clinically visible within the range of realistic inter-scanner variation.
Multi-institution quality assurance should prioritize CT spatial-resolution and reconstruction-kernel
consistency, and dose-based rather than image-quality metrics should be used to assess CBCT and
synthetic-CT suitability.

Future work. Extend the battery to CBCT-characteristic degradations (scatter and cupping, ring
artifacts, and limited-field-of-view truncation) to establish commissioning tolerances for online
adaptive radiotherapy; develop distribution-free and certified per-patient DVH prediction intervals
under bounded CT perturbation; and apply training-time augmentation to improve resolution robustness.

References.
1. Babier A, Mahon R, McNiven A, Diamant A, Chan TCY. Med Phys. 2021;48:4932-4948 (OpenKBP).
2. Gao Y, et al. Phys Med Biol. 2025;70:115006.
3. American College of Radiology. CT Quality Control and Accreditation guidelines.

## 9. Raw data files
- `summary.csv`: aggregate dose-score and DVH-score with deltas for all 26 conditions.
- `detailed_results.json`: per-structure mean DVH errors for all conditions.
