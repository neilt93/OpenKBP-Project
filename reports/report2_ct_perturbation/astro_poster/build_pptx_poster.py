#!/usr/bin/env python3
"""Build the ASTRO #79011 poster by cloning the AAPM template (same 44x44 format, section-header
bars, fonts, colours, U-Net diagram) and swapping in the CT-perturbation content + figures.

Clone-and-edit (not build-fresh) so the styling Birjoo asked for is inherited exactly. Text is
replaced preserving each box's font; the AAPM logo + adversarial-specific figures are removed; the
realistic-perturbation figures (already generated, no pod) are added in the results column; the
U-Net method diagram is kept.

Open the output in PowerPoint for any final nudging of figure/caption positions.
"""
import copy
import os
import shutil

from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.normpath(os.path.join(HERE, "..", "..", "..", "AdversarialAAPM.pptx"))
FIGDIR = os.path.normpath(os.path.join(HERE, "..", "..", "..",
                                       "open-kbp-modified", "openkbp_hn_robustness", "figures"))
OUT = os.path.join(HERE, "ASTRO_79011_poster.pptx")

TITLE = ("Robustness of Deep-Learning Dose Prediction in Head-and-Neck Radiotherapy "
         "to Clinically Realistic CT Perturbations")
AUTHORS = ["Neil Tripathi¹, Rahim Chowdhury², Lei Ren³, Amit Sawant³, Birjoo Vaishnav³",
           "¹New York University   ²UMD St. Joseph Medical Center   "
           "³University of Maryland School of Medicine, Dept. of Radiation Oncology"]

ABSTRACT = [
    "Deep-learning dose prediction is nearing clinical deployment, where models trained at one "
    "institution encounter CTs from different scanners, reconstruction kernels, and imaging "
    "practices. We stress-tested a 3D U-Net head-and-neck dose predictor against five families of "
    "clinically realistic CT perturbations — acquisition noise, HU calibration shift, "
    "low-frequency bias field, spatial-resolution loss, and dental streak artifacts — across "
    "26 conditions on 40 test patients, at severities meeting or exceeding ACR CT-simulation QA.",
    "Defining a level as “clinically visible” when the cohort-mean change in any DVH "
    "criterion exceeds 1.0 Gy from each patient’s own baseline, four of five families never "
    "reach visibility. Only spatial-resolution loss does — crossing 1.0 Gy at a clinically "
    "realistic severity (~2 mm-equivalent blur) and rising to 2.84 Gy at the larynx (+18.2% DVH).",
    "Cross-institution deployment QA should therefore prioritize CT spatial-resolution and "
    "reconstruction-kernel consistency over intensity calibration.",
]
INTRO = [
    "Deep-learning dose prediction supports automated and adaptive treatment planning, but a model "
    "trained at one site must generalize to CTs acquired elsewhere — different scanners, "
    "kernels, dose levels, and artifacts. Unlike worst-case adversarial attacks, these are benign, "
    "physically realistic sources of image variability.",
    "We ask a clinically framed question: how much CT degradation can a dose-prediction model "
    "tolerate before its errors become clinically visible? We answer it per perturbation family, "
    "identifying which CT-quality factors actually matter for safe multi-institution deployment.",
]
METHODS = [
    "Data set | OpenKBP head-and-neck cohort (Babier et al. 2021): 200 training / 40 test patients; "
    "128³ CT with 10 OAR/PTV structures; dose normalized to a 70 Gy prescription.",
    "Realistic perturbation model | Five families applied to the test-input CT at 5 severity levels "
    "each (26 conditions), ranges meeting or exceeding ACR CT-simulation QA:",
    "  P1 heteroscedastic acquisition noise  •  P2 bone-weighted HU calibration shift  •  "
    "P3 low-frequency bias field  •  P4 anisotropic resolution loss (Gaussian blur)  •  "
    "P5 dental metal streak artifacts.",
    "Impact = ΔDVH and ΔMAE vs each patient’s own clean-CT baseline prediction. "
    "Visibility threshold: cohort-mean shift on any DVH criterion > 1.0 Gy.",
]
MODEL_BLOCK = [
    "Dose Prediction Model | 3D U-Net with squeeze-and-excitation blocks; masked MAE loss with 4× "
    "PTV weighting. Baseline DVH score 2.54, dose score 3.73 Gy.",
]
RESULTS_SUMMARY = [
    "Four of five families leave every DVH criterion below the 1.0 Gy visibility threshold across "
    "the entire tested range. Spatial-resolution loss (P4) is the sole failure mode: its "
    "worst-affected criterion (larynx near-max dose) crosses 1.0 Gy at L2 (2.0/1.0-voxel blur) and "
    "reaches 2.84 Gy at L4 (+18.2% DVH score).",
    "HU calibration shift becomes measurable only at an implausible 1000 HU bone offset (0.61 Gy, "
    "still sub-threshold). Noise, bias field, and dental streaks never exceed 0.34 Gy on any "
    "criterion, even beyond ACR-level severities.",
]
FIG1_CAP = ("Figure 1. Example CT slices for one patient: original and each perturbation family at a "
            "mid severity (top), with difference-from-original maps (bottom). Resolution loss "
            "visibly blurs tissue boundaries; intensity perturbations leave structure edges intact.")
FIG2_CAP = ("Figure 2. Worst-case per-criterion DVH shift (Gy) vs severity for each family, with the "
            "1.0 Gy clinical-visibility line. P4 (resolution) crosses between L1 and L2 and keeps "
            "climbing; all other families remain near the floor.")
FIG3_CAP = ("Figure 3. Predicted dose (top) and difference-from-baseline maps (bottom) per "
            "perturbation. Resolution degradation produces spatially structured dose errors "
            "concentrated at organ boundaries; other families are near-zero.")
DISCUSSION = [
    "The model is robust to intensity-based CT variability — noise, calibration drift, bias "
    "field, dental streaks — through and beyond ACR-level severities, because these preserve the "
    "structural edges the network uses to localize organ boundaries.",
    "Spatial-resolution loss removes exactly that edge content, producing boundary-localized dose "
    "errors that grow approximately linearly with blur and appear first in geometrically complex "
    "structures (larynx, parotids, mandible).",
    "Because realistic cross-scanner slice-thickness and reconstruction-kernel differences span the "
    "severities where P4 becomes visible, spatial resolution — not HU calibration — is the "
    "clinically relevant robustness risk.",
]
CONCLUSIONS = [
    "A head-and-neck deep-learning dose predictor tolerates clinically realistic intensity CT "
    "perturbations but is systematically sensitive to spatial-resolution degradation, which becomes "
    "clinically visible (>1 Gy on a DVH criterion) within the range of real cross-scanner variation.",
    "Multi-institution deployment QA should prioritize CT spatial-resolution and reconstruction-"
    "kernel consistency over intensity calibration; dose-level (not image-quality) metrics should "
    "gate CBCT / synthetic-CT quality.",
]
FUTURE = [
    "Extend the battery to CBCT-characteristic degradations (scatter/cupping, ring artifacts, "
    "limited-FOV truncation) for online adaptive-RT commissioning tolerances; provable (certified) "
    "and conformal per-patient DVH bounds under bounded CT perturbation; and training-time "
    "augmentation to restore the resolution margin.",
]
REFERENCES = [
    "1. Babier A, Mahon R, McNiven A, Diamant A, Chan TCY. Med Phys. 2021;48:4932-4948 (OpenKBP).",
    "2. Gao Y, et al. Phys Med Biol. 2025;70:115006.",
    "3. American College of Radiology. CT Quality Control / Accreditation guidelines.",
]

# match-substring -> new lines. Section-header boxes (all-caps) are left untouched.
RULES = [
    ("Adversarial Robustness Evaluation", [TITLE]),
    ("Rahim Chowdhury", AUTHORS),
    ("increasingly used in radiotherapy", ABSTRACT),                 # ABSTRACT body (left col)
    ("treatment planning models are increasingly", INTRO),          # INTRODUCTION body
    ("Data set", METHODS),                                          # METHODS body
    ("Dose Prediction Model", MODEL_BLOCK),                         # methods model sub-block
    ("Adversarial perturbation model", [""]),                       # drop (covered in METHODS)
    ("Measurable dosimetric changes", RESULTS_SUMMARY),            # RESULTS summary text
    ("Deep learning dose prediction models demonstrated", DISCUSSION),  # DISCUSSION body
    ("We adopted adversarial perturbation", CONCLUSIONS),          # CONCLUSIONS body
    ("The noise models", FUTURE),                                  # FUTURE WORK body
    ("Babier A", REFERENCES),                                      # REFERENCES
    ("Figure 1 demonstrates", [FIG1_CAP]),
    ("Figure 2 above demonstrates", [FIG2_CAP]),
    ("Figure 3. Robustness analysis", [FIG3_CAP]),
    ("Perturbed", [""]),                                           # triptych label (removed figs)
]


def _capture_style(tf):
    for p in tf.paragraphs:
        if p.runs:
            r = p.runs[0]
            col = None
            try:
                if r.font.color is not None and r.font.color.type is not None:
                    col = r.font.color.rgb
            except Exception:
                pass
            return {"name": r.font.name or "Calibri", "size": r.font.size, "bold": r.font.bold,
                    "color": col}
    return {"name": "Calibri", "size": Pt(28), "bold": None, "color": None}


def set_text(shape, lines):
    tf = shape.text_frame
    st = _capture_style(tf)
    tf.clear()
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        run = p.add_run(); run.text = line
        run.font.name = st["name"]
        if st["size"] is not None:
            run.font.size = st["size"]
        run.font.bold = st["bold"]
        if st["color"] is not None:
            run.font.color.rgb = st["color"]


def main():
    shutil.copyfile(TEMPLATE, OUT)
    prs = Presentation(OUT)
    slide = prs.slides[0]

    # 1) text replacement
    for shape in list(slide.shapes):
        if not shape.has_text_frame:
            continue
        cur = shape.text_frame.text
        for sub, lines in RULES:
            if sub in cur:
                set_text(shape, lines)
                break

    # 2) remove AAPM logo + adversarial figures; keep the U-Net diagram (left ~12.3, top ~30)
    for pic in [s for s in slide.shapes if s.shape_type == 13]:
        L, T = pic.left / 914400.0, pic.top / 914400.0
        keep_unet = (11.0 < L < 14.0) and (28.0 < T < 33.0)
        if keep_unet:
            continue
        if L < 1.0 or L >= 20.0 or T >= 37.0:          # logo / results figs / bottom triptych
            pic._element.getparent().remove(pic._element)

    # 3) add the realistic-perturbation figures in the results column
    def add(fig, left, top, width):
        path = os.path.join(FIGDIR, fig)
        slide.shapes.add_picture(path, Inches(left), Inches(top), width=Inches(width))

    add("ct_slices.png", 20.7, 15.3, 16.2)              # Figure 1: perturbation examples
    # money figure lives in the astro_poster dir (not FIGDIR)
    slide.shapes.add_picture(os.path.join(HERE, "fig_panel5_maxcrit_gy.png"),
                             Inches(23.5), Inches(23.6), width=Inches(10.5))  # Figure 2
    add("dose_difference_maps.png", 20.7, 31.2, 16.2)   # Figure 3

    # 4) nudge caption boxes to sit just below each figure
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        t = shape.text_frame.text
        if t.startswith("Figure 1."):
            shape.top = Inches(22.6)
        elif t.startswith("Figure 2."):
            shape.top = Inches(30.0)
        elif t.startswith("Figure 3."):
            shape.top = Inches(38.2)

    prs.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
