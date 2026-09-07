# ASTRO 2026 ePoster #79011 — filled panels

**Assessing the Generalizability and Robustness of Deep-Learning Dose Prediction in
Head-and-Neck Radiotherapy to Clinically Realistic CT Perturbations**
Neil Tripathi, Rahim Chowdhury, Lei Ren, Amit Sawant, Birjoo Vaishnav — University of Maryland
School of Medicine. PQA 05: Physics · Tue Sept 29, 12:30–1:45 PM.

Hook (top of poster): *"How much CT degradation can a dose-prediction model tolerate before
its errors become clinically visible?"*

All values below are computed from the committed sweep (`../..//open-kbp-modified/openkbp_hn_robustness/metrics/`)
by `build_astro_panels.py`. Regenerate with: `python build_astro_panels.py`.

---

## Panel 1 — Background & Purpose
Deep-learning dose prediction is approaching clinical use, but a model trained at one institution
will see CTs from different scanners, reconstruction kernels, and imaging practices. We stress-test
a head-and-neck dose-prediction model across five families of clinically realistic CT perturbations
to find, for each family, the severity at which prediction errors become clinically visible — and
therefore which CT-quality factors actually matter for safe cross-institution deployment.

## Panel 2 — Threshold criterion (STATE ONCE, USE EVERYWHERE)
A severity level is **"clinically visible"** when the **cohort-mean change in any standard DVH
criterion, relative to each patient's own baseline prediction, exceeds 1.0 Gy.**
Corroboration (reported, not the gate): per-patient voxel-wise dose MAE — the fraction of the 40
patients whose dose MAE rises >0.5 Gy. (Gamma index not computed → not mentioned.)

## Panel 3 — Methods (compressed)
- **Model:** 3D U-Net + squeeze-and-excitation blocks; masked MAE loss, 4× PTV weighting;
  OpenKBP (200 train / 40 test). Baseline DVH score 2.535, dose score 3.731.
- **Evaluation:** 26 CT conditions on the 40 test patients — baseline + severity sweeps of five
  perturbation families, ranges chosen to meet or exceed ACR CT-simulation QA thresholds:
  P1 heteroscedastic noise · P2 bone-weighted HU calibration shift · P3 low-frequency bias field ·
  P4 anisotropic resolution loss (blur) · P5 dental streak artifacts.
- **Metrics:** ΔDVH (per criterion and DVH-score) and ΔMAE vs each patient's own baseline prediction.

## Panel 4 — CENTERPIECE threshold table
*(criterion: mean shift on any DVH criterion > 1.0 Gy)*

| Family | Sweep range tested | Clinically visible at | vs. clinical range | Verdict |
|---|---|---|---|---|
| P1 noise | L1 (8/12 HU) → L5 (100/160 HU) | never (max 0.14 Gy) | scanner noise ≈10–50 HU (≈L1–L2) | 🟢 Robust |
| P2 HU shift | L1 (5/50 HU) → L5 (100/1000 HU) | never (max 0.61 Gy @ 1000 HU) | calibration drift ≈10–50 HU (≈L2) | 🟢 Robust in clinical range |
| P3 bias field | L1 (10 HU) → L5 (200 HU) | never (max 0.09 Gy) | RF/scatter, tens of HU | 🟢 Robust |
| **P4 resolution** | L0 (0.5/0.25 vox) → L4 (4.0/2.0 vox) | **L2 (2.0/1.0 vox): 1.15 Gy on Larynx D0.1cc** | slice/kernel variation ≈1–3 vox (≈L1–L3) | 🔴 **SENSITIVE — the failure mode** |
| P5 dental streak | L1 (150HU/8) → L5 (1200HU/24) | never (max 0.33 Gy) | streaks common in H&N (L2–L4 realistic) | 🟢 Robust |

Color the Verdict column green/green/green/red/green. Single-glance story: **four of five
families are safe; resolution is the one that bites — inside the clinically realistic range.**
The larynx near-max dose (D0.1cc) is the sentinel: blur at the airway boundary is where it breaks.

## Panel 5 — Severity curves
- `fig_panel5_maxcrit_gy.png` — **the money figure.** Max per-criterion DVH shift (Gy) vs severity,
  all five families on shared axes, dashed line at the 1.0 Gy visibility threshold. P4 crosses
  between L1 and L2; every other family hugs the floor.
- `fig_panel5_dvh_pct.png` — twin panel: aggregate DVH-score degradation (%). P4 rises linearly to
  +18.2% at L4; P2 shows a late take-off (+11.2% at L5, 1000 HU); P1/P3/P5 flat.

## Panel 6 — What failure looks like
Static qualitative figures already exist (`../figures/dose_difference_maps.png`,
`../figures/ct_slices.png`): P4 produces spatially structured dose errors concentrated at organ
boundaries. Caption: *"Loss of edge definition degrades dose prediction near structure boundaries."*
**Interactive element — script ready, needs one pod run:** `generate_p4_gif.py` steps one patient
across the P4 sweep (baseline, L0→L4), rendering a dose wash + Larynx/PTV70 DVH overlay per frame
into a GIF (+ frame PNGs). It reuses the verified `run_inference.py` path, so it's consistent with
the sweep numbers. Run on the pod:
`python generate_p4_gif.py --model <best>/models/epoch_100.keras --data-dir <validation-pats> --patient pt_205 --fps 2`

## Panel 7 — Conclusions (in threshold language)
- The model tolerates **intensity-based** CT variability (noise, bias field, dental streaks)
  through and beyond ACR-level severities — no DVH criterion moves >1 Gy anywhere in range.
- **HU calibration shift** only matters at extreme, clinically implausible offsets (~1000 HU);
  realistic drift (10–50 HU) is safe.
- **Spatial-resolution loss is the dominant, systematic failure mode** — a DVH criterion crosses
  1 Gy at L2 (σ_z/xy = 2.0/1.0 vox), within the range of real cross-scanner variation, and rises
  to 2.84 Gy (Larynx D0.1cc) / +18.2% DVH at L4.
- **Practical read:** cross-institution deployment QA should prioritize **resolution /
  reconstruction-kernel consistency** over intensity calibration.

## Panel 8 — Ongoing work (one line, keep strands separate)
*"Ongoing work: provable (certified) bounds on DVH metrics under bounded CT perturbation, and
hardening via training-time augmentation."*

---

## 7-minute talk track (for Birjoo)
1. **Hook (45s).** Models trained at one site get deployed on other scanners. Question: how much CT
   degradation before dose-prediction errors become *clinically visible* (>1 Gy on a DVH criterion)?
2. **Setup (60s).** One H&N U-Net, 40 test patients, 26 CT conditions across five realistic
   perturbation families, severities up to/beyond ACR QA thresholds. Everything measured against
   each patient's own baseline prediction.
3. **Threshold table (90s).** Walk the five rows. Four are green — no DVH criterion moves >1 Gy even
   at extreme severity. One is red: resolution.
4. **The money curve (90s).** `fig_panel5_maxcrit_gy.png`: P4 crosses 1 Gy between L1 and L2 — inside
   the realistic slice/kernel range — and keeps climbing; everyone else is flat on the floor. The
   larynx near-max dose is the sentinel: it's a boundary-localization failure.
5. **Why it matters / practical read (75s).** Deployment QA should standardize resolution and
   reconstruction kernels first; intensity calibration is not the risk. HU shifts only bite at
   implausible ~1000 HU.
6. **Ongoing work (30s).** One sentence on certified bounds — a separate line of work.
7. **Close (20s).** "Resolution consistency is the single most impactful cross-site QC lever."

## What's done vs. what remains
- ✅ Threshold criterion decided (per-criterion 1 Gy) and it works — P4 crosses at L2, all others never.
- ✅ Panel 4 table (`panel4_threshold_table.{md,csv}`).
- ✅ Panel 5 figures (`fig_panel5_maxcrit_gy.png`, `fig_panel5_dvh_pct.png`).
- ✅ Panel 7 conclusions + 7-min talk track (above).
- ✅ **Panel 6 GIF generator written** (`generate_p4_gif.py`) — reuses the verified inference path.
  ⏳ Needs one pod run (model + validation data; ~6 forward passes) to produce the actual GIF; the
  volumes weren't saved locally so it can't run on the Mac. Static figures cover the panel meanwhile.

## Logistics
- Upload fee $85 tier (Aug 8–Sep 21); **hard upload deadline Sept 21.**
- Birjoo is presenter of record; poster to him for review well before upload.
- Embed the P4 GIF (interactive element) once regenerated.
- No vendor/commercial names anywhere (ACCME).
