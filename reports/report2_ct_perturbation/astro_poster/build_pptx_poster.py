#!/usr/bin/env python3
"""Build the ASTRO #79011 poster by cloning the AAPM template (same 44x44 format, section-header
bars, colours) and replacing text + figures with the CT-perturbation study.

Fixes over the first pass: concrete Calibri font (no theme-token names), explicit fit-checked font
sizes + shrink-to-fit so text never overflows/overlaps, formal scientific prose, and a clean
non-overlapping figure/caption stack with preserved aspect ratios.

Open in PowerPoint for a final logo (top-left) if desired.
"""
import os
import shutil

from PIL import Image
from pptx import Presentation
from pptx.enum.text import MSO_AUTO_SIZE
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.normpath(os.path.join(HERE, "..", "..", "..", "AdversarialAAPM.pptx"))
FIGDIR = os.path.normpath(os.path.join(HERE, "..", "..", "..",
                                       "open-kbp-modified", "openkbp_hn_robustness", "figures"))
OUT = os.path.join(HERE, "ASTRO_79011_poster.pptx")
FONT = "Calibri"

TITLE = ("Robustness of Deep-Learning Dose Prediction in Head-and-Neck Radiotherapy "
         "to Clinically Realistic CT Perturbations")
AUTHORS = ["Neil Tripathi¹, Rahim Chowdhury², Lei Ren³, Amit Sawant³, Birjoo Vaishnav³",
           "¹New York University    ²UMD St. Joseph Medical Center    "
           "³University of Maryland School of Medicine, Department of Radiation Oncology"]

ABSTRACT = [
    "Purpose. Deep-learning dose prediction models are increasingly deployed across institutions, "
    "where they encounter CT images from different scanners, reconstruction kernels, and acquisition "
    "protocols. This study quantifies the robustness of a head-and-neck dose prediction model to five "
    "families of clinically realistic CT perturbations and identifies the image-quality factors that "
    "materially affect predicted dose.",
    "Methods. A 3D U-Net dose predictor was evaluated on 40 test patients under 26 CT conditions "
    "spanning acquisition noise, HU calibration shift, low-frequency bias field, spatial-resolution "
    "loss, and dental streak artifacts, at severities meeting or exceeding ACR CT-simulation "
    "quality-assurance limits. A perturbation level was defined as clinically visible when the "
    "cohort-mean change in any dose-volume-histogram (DVH) criterion exceeded 1.0 Gy relative to each "
    "patient's unperturbed prediction.",
    "Results. Four of five perturbation families produced no clinically visible change at any tested "
    "severity. Spatial-resolution degradation was the only family to exceed the 1.0 Gy threshold, "
    "reaching 2.84 Gy at the larynx (+18.2% DVH score) at the highest severity.",
    "Conclusions. The model is robust to intensity-based CT variability but sensitive to "
    "spatial-resolution loss. Multi-institution quality assurance should prioritize CT "
    "spatial-resolution and reconstruction-kernel consistency.",
]
INTRO = [
    "Deep-learning dose prediction is increasingly used to support automated and adaptive treatment "
    "planning. A model trained at one institution must generalize to CT images acquired under "
    "different conditions, including variation in scanner hardware, reconstruction kernel, dose "
    "level, and imaging artifacts. In contrast to worst-case adversarial perturbations, these "
    "represent benign, physically plausible sources of image variability.",
    "This study characterizes the sensitivity of a head-and-neck dose prediction model to five "
    "families of clinically realistic CT perturbations and determines the severity at which each "
    "produces clinically significant changes in predicted dose, in order to identify the CT "
    "image-quality factors relevant to safe multi-institution deployment.",
]
METHODS = [
    "Dataset. The OpenKBP head-and-neck cohort (Babier et al., 2021) was used: 200 training and 40 "
    "test patients, each with a 128³ CT, ten organ-at-risk and target structures, and dose "
    "normalized to a 70 Gy prescription.",
    "Perturbations. Five families were applied to the test-input CT at five severity levels each "
    "(26 conditions total), with parameter ranges meeting or exceeding ACR CT-simulation "
    "quality-assurance limits: P1 heteroscedastic acquisition noise; P2 bone-weighted HU "
    "calibration shift; P3 low-frequency bias field; P4 anisotropic spatial-resolution loss; "
    "P5 dental metal streak artifacts.",
    "Analysis. Impact was quantified as the change in DVH metrics and voxel-wise mean absolute error "
    "relative to each patient's unperturbed prediction. A level was deemed clinically visible when "
    "the cohort-mean shift in any DVH criterion exceeded 1.0 Gy.",
]
MODEL_BLOCK = [
    "Model. A 3D U-Net with squeeze-and-excitation blocks was trained with a masked mean-absolute-"
    "error loss and 4× planning-target-volume weighting (baseline DVH score 2.54; dose score "
    "3.73 Gy).",
]
RESULTS_SUMMARY = [
    "Four of five perturbation families left every DVH criterion below the 1.0 Gy visibility "
    "threshold across the full tested range. Spatial-resolution loss (P4) was the sole family to "
    "exceed it: the most affected criterion, larynx near-maximum dose, crossed 1.0 Gy at severity "
    "level L2 (2.0/1.0-voxel Gaussian blur) and reached 2.84 Gy at L4 (+18.2% DVH score). "
    "Bone-weighted HU calibration shift became measurable only at an implausible 1000 HU offset "
    "(0.61 Gy, below threshold). Acquisition noise, bias field, and dental artifacts did not exceed "
    "0.34 Gy on any criterion, even beyond ACR-level severities.",
]
DISCUSSION = [
    "The model was robust to intensity-based CT variability, including acquisition noise, "
    "calibration drift, bias field, and dental artifacts, at severities beyond typical clinical "
    "ranges. These perturbations preserve the structural boundaries the network uses to localize "
    "anatomy.",
    "Spatial-resolution loss degrades this boundary information, producing spatially structured dose "
    "errors that increase approximately linearly with blur and appear first in geometrically complex "
    "structures such as the larynx, parotid glands, and mandible.",
    "Because realistic inter-scanner differences in slice thickness and reconstruction kernel span "
    "the severities at which resolution loss becomes clinically visible, spatial resolution, rather "
    "than intensity calibration, is the principal robustness concern for cross-institution "
    "deployment.",
]
CONCLUSIONS = [
    "A head-and-neck deep-learning dose prediction model tolerates clinically realistic "
    "intensity-based CT perturbations but is systematically sensitive to spatial-resolution "
    "degradation, which becomes clinically visible within the range of realistic inter-scanner "
    "variation.",
    "Multi-institution quality assurance should prioritize CT spatial-resolution and reconstruction-"
    "kernel consistency, and dose-based rather than image-quality metrics should be used to assess "
    "CBCT and synthetic-CT suitability.",
]
FUTURE = [
    "Planned extensions include CBCT-characteristic degradations (scatter and cupping, ring "
    "artifacts, limited-field-of-view truncation) to establish commissioning tolerances for online "
    "adaptive radiotherapy; distribution-free (conformal) and certified per-patient DVH prediction "
    "intervals under bounded CT perturbation; and training-time augmentation to improve resolution "
    "robustness.",
]
REFERENCES = [
    "1. Babier A, Mahon R, McNiven A, Diamant A, Chan TCY. Med Phys. 2021;48:4932-4948 (OpenKBP).",
    "2. Gao Y, et al. Phys Med Biol. 2025;70:115006.",
    "3. American College of Radiology. CT Quality Control and Accreditation guidelines.",
]
FIG1_CAP = ("Figure 1. Representative axial CT for one patient: unperturbed image and each "
            "perturbation family at intermediate severity (top row), with corresponding difference "
            "maps (bottom row). Spatial-resolution loss visibly blurs tissue boundaries, whereas "
            "intensity-based perturbations preserve structural edges.")
FIG2_CAP = ("Figure 2. Maximum per-criterion DVH shift versus perturbation severity for each family; "
            "the dashed line marks the 1.0 Gy clinical-visibility threshold. Spatial-resolution loss "
            "(P4) crosses the threshold between levels L1 and L2; all other families remain near "
            "zero.")
FIG3_CAP = ("Figure 3. Predicted dose (top row) and difference-from-baseline maps (bottom row) for "
            "each perturbation family. Spatial-resolution loss produces spatially structured dose "
            "errors concentrated at organ boundaries; other families produce negligible change.")

BODY, CAP, SUB = 24, 18, 22
# match-substring -> (new lines, font size). Section-header boxes (all-caps) are left untouched.
RULES = [
    ("Adversarial Robustness Evaluation", ([TITLE], 72)),
    ("Rahim Chowdhury", ([AUTHORS[0]], 40)),
    ("increasingly used in radiotherapy", (ABSTRACT, BODY)),
    ("treatment planning models are increasingly", (INTRO, BODY)),
    ("Data set", (METHODS, BODY)),
    ("Dose Prediction Model", (MODEL_BLOCK, SUB)),
    ("Adversarial perturbation model", ([""], SUB)),
    ("Measurable dosimetric changes", (RESULTS_SUMMARY, BODY)),
    ("Deep learning dose prediction models demonstrated", (DISCUSSION, BODY)),
    ("We adopted adversarial perturbation", (CONCLUSIONS, BODY)),
    ("The noise models", (FUTURE, BODY)),
    ("Babier A", (REFERENCES, 20)),
    ("Figure 1 demonstrates", ([FIG1_CAP], CAP)),
    ("Figure 2 above demonstrates", ([FIG2_CAP], CAP)),
    ("Figure 3. Robustness analysis", ([FIG3_CAP], CAP)),
    ("Perturbed", ([""], CAP)),
]


def set_text(shape, lines, size, bold=None):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    tf.clear()
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(6)
        run = p.add_run(); run.text = line
        run.font.name = FONT
        run.font.size = Pt(size)
        if bold is not None:
            run.font.bold = bold


def main():
    shutil.copyfile(TEMPLATE, OUT)
    prs = Presentation(OUT)
    slide = prs.slides[0]

    # 1) text replacement (concrete font + fit sizes)
    for shape in list(slide.shapes):
        if not shape.has_text_frame:
            continue
        cur = shape.text_frame.text
        for sub, (lines, size) in RULES:
            if sub in cur:
                set_text(shape, lines, size)
                break

    # authors affiliation line: put both lines in the authors box
    for shape in slide.shapes:
        if shape.has_text_frame and shape.text_frame.text.startswith("Neil Tripathi"):
            set_text(shape, AUTHORS, 34)
        # header "METHODS AND MATERIALS" overruns its box -> shorten to "METHODS"
        elif shape.has_text_frame and shape.text_frame.text.strip().startswith("METHODS AND"):
            set_text(shape, ["METHODS"], 54, bold=True)

    # 2) remove AAPM logo + adversarial figures; keep the U-Net diagram (left ~12.3, top ~30)
    for pic in [s for s in slide.shapes if s.shape_type == 13]:
        L, T = pic.left / 914400.0, pic.top / 914400.0
        if (11.0 < L < 14.0) and (28.0 < T < 33.0):
            continue                                   # keep U-Net
        if L < 1.0 or L >= 20.0 or T >= 37.0:
            pic._element.getparent().remove(pic._element)

    # 3) clean figure + caption stack in the results column (left 20.6, width 16, aspect-preserved)
    COL_L, COL_W = 20.6, 16.0

    def aspect(path):
        im = Image.open(path); return im.size[0] / im.size[1]

    def add(path, top, width):
        left = COL_L + (COL_W - width) / 2.0
        slide.shapes.add_picture(path, Inches(left), Inches(top), width=Inches(width))
        return top + width / aspect(path)             # bottom edge

    ct = os.path.join(FIGDIR, "ct_slices.png")
    money = os.path.join(HERE, "fig_panel5_maxcrit_gy.png")
    dose = os.path.join(FIGDIR, "dose_difference_maps.png")

    # shrink the results-summary box so it does not run into Figure 1
    for shape in slide.shapes:
        if shape.has_text_frame and shape.text_frame.text.startswith("Four of five"):
            shape.top = Inches(7.5); shape.height = Inches(4.6)

    b1 = add(ct, 12.6, COL_W)          # Fig 1 examples
    cap1_top = b1 + 0.1
    b2 = add(money, cap1_top + 1.9, 10.5)   # Fig 2 curve (narrower, centred)
    cap2_top = b2 + 0.1
    b3 = add(dose, cap2_top + 1.9, COL_W)   # Fig 3 dose maps
    cap3_top = b3 + 0.1

    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        t = shape.text_frame.text
        if t.startswith("Figure 1."):
            shape.left, shape.top, shape.width, shape.height = (
                Inches(COL_L), Inches(cap1_top), Inches(COL_W), Inches(1.7))
        elif t.startswith("Figure 2."):
            shape.left, shape.top, shape.width, shape.height = (
                Inches(COL_L), Inches(cap2_top), Inches(COL_W), Inches(1.7))
        elif t.startswith("Figure 3."):
            shape.left, shape.top, shape.width, shape.height = (
                Inches(COL_L), Inches(cap3_top), Inches(COL_W), Inches(1.7))

    prs.save(OUT)
    print(f"Wrote {OUT}")
    print(f"Figure bottoms: F1 {b1:.1f}  F2 {b2:.1f}  F3 {b3:.1f} (references at ~39)")


if __name__ == "__main__":
    main()
