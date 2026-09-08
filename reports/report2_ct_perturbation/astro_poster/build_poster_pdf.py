#!/usr/bin/env python3
"""Assemble a single showable PDF draft of the ASTRO #79011 poster.

Page 1: poster-at-a-glance -- title, hook, threshold criterion, the colored threshold
        table, and conclusions/practical read.
Page 2: the two severity figures at full size with captions.

Self-contained (matplotlib only). Recomputes the table from the committed sweep JSONs
so it stays a single source of truth. Output: ASTRO_79011_poster_draft.pdf
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "open-kbp-modified", "openkbp_hn_robustness", "metrics", "per_patient"))
THRESH_GY = 1.0

FAMS = [
    ("P1_noise",      "P1 noise",         [1, 2, 3, 4, 5], "8/12 → 100/160 HU"),
    ("P2_bone_shift", "P2 HU shift",      [1, 2, 3, 4, 5], "5/50 → 100/1000 HU"),
    ("P3_bias_field", "P3 bias field",    [1, 2, 3, 4, 5], "10 → 200 HU"),
    ("P4_resolution", "P4 resolution",    [0, 1, 2, 3, 4], "0.5/0.25 → 4.0/2.0 vox"),
    ("P5_dental",     "P5 dental streak", [1, 2, 3, 4, 5], "150/8 → 1200/24"),
]
GREEN, RED = "#d4efdf", "#f5b7b1"

base = json.load(open(os.path.join(METRICS, "baseline.json")))
bstruct = base["per_structure"]

def row_for(fam, levels):
    best = None  # (level, shift, crit) at first crossing
    maxabs = 0.0
    for lvl in levels:
        s = json.load(open(os.path.join(METRICS, f"{fam}_L{lvl}.json")))["per_structure"]
        shifts = {k: s[k] - bstruct[k] for k in s if k in bstruct}
        mk = max(shifts, key=lambda k: abs(shifts[k]))
        maxabs = max(maxabs, abs(shifts[mk]))
        if best is None and abs(shifts[mk]) > THRESH_GY:
            best = (lvl, shifts[mk], mk)
    if best is None:
        return f"never (max {maxabs:.2f} Gy)", "Robust", GREEN
    lvl, sh, mk = best
    return f"L{lvl}: {sh:.2f} Gy ({mk.replace('_',' ')})", "SENSITIVE", RED

rows = []
for fam, label, levels, rng in FAMS:
    vis, verdict, color = row_for(fam, levels)
    rows.append([label, rng, vis, verdict, color])

# ---------------------------------------------------------------- PDF -----------
pdf_path = os.path.join(HERE, "ASTRO_79011_poster_draft.pdf")
with PdfPages(pdf_path) as pdf:
    # ---- Page 1: at a glance ----
    fig = plt.figure(figsize=(16, 9)); fig.patch.set_facecolor("white")
    fig.text(0.5, 0.955, "Robustness of Deep-Learning Dose Prediction in H&N Radiotherapy",
             ha="center", fontsize=20, fontweight="bold")
    fig.text(0.5, 0.918, "to Clinically Realistic CT Perturbations", ha="center", fontsize=20,
             fontweight="bold")
    fig.text(0.5, 0.885, "N. Tripathi, R. Chowdhury, L. Ren, A. Sawant, B. Vaishnav  ·  "
             "University of Maryland School of Medicine", ha="center", fontsize=11, color="#555")
    fig.text(0.5, 0.862, "ASTRO 2026 ePoster #79011  ·  PQA 05: Physics  ·  Tue Sept 29",
             ha="center", fontsize=10, color="#888")

    fig.text(0.5, 0.815, '"How much CT degradation can a dose-prediction model tolerate\n'
             'before its errors become clinically visible?"', ha="center", va="top",
             fontsize=14, style="italic", color="#c0392b",
             bbox=dict(boxstyle="round,pad=0.6", fc="#fdf2f2", ec="#c0392b"))

    # left text column
    L = 0.06
    fig.text(L, 0.70, "THRESHOLD CRITERION", fontsize=12, fontweight="bold")
    fig.text(L, 0.665, "A severity level is \"clinically visible\" when the cohort-mean\n"
             "change in ANY standard DVH criterion, vs each patient's own\n"
             "baseline prediction, exceeds 1.0 Gy.", fontsize=10.5, va="top")

    fig.text(L, 0.575, "METHODS", fontsize=12, fontweight="bold")
    fig.text(L, 0.545, "3D U-Net + SE blocks, masked MAE, 4x PTV weight (OpenKBP).\n"
             "40 test patients x 26 CT conditions: baseline + 5 perturbation\n"
             "families x 5 severities, ranges meeting/exceeding ACR QA.\n"
             "Baseline DVH 2.535, Dose 3.731 Gy.", fontsize=10.5, va="top")

    fig.text(L, 0.42, "CONCLUSIONS", fontsize=12, fontweight="bold")
    fig.text(L, 0.39,
             "• Intensity variability (noise, bias field, dental) tolerated\n"
             "   through and beyond ACR severities — no criterion > 1 Gy.\n"
             "• HU calibration shift only bites at implausible ~1000 HU;\n"
             "   realistic drift (10–50 HU) is safe.\n"
             "• Spatial-resolution loss is the dominant failure mode —\n"
             "   crosses 1 Gy at L2 (2.0/1.0 vox), within real cross-scanner\n"
             "   variation, reaching 2.84 Gy (Larynx D0.1cc) at L4.\n"
             "• Practical read: deployment QA should prioritize resolution /\n"
             "   reconstruction-kernel consistency over intensity calibration.",
             fontsize=10.5, va="top")

    # right: table
    axt = fig.add_axes([0.44, 0.44, 0.52, 0.30]); axt.axis("off")
    col_labels = ["Family", "Range tested", "Clinically visible at", "Verdict"]
    cell_text = [[r[0], r[1], r[2], r[3]] for r in rows]
    tbl = axt.table(cellText=cell_text, colLabels=col_labels, loc="center",
                    cellLoc="left", colWidths=[0.17, 0.24, 0.42, 0.17])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.7)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#34495e"); cell.set_text_props(color="white", fontweight="bold")
        elif c == 3:
            cell.set_facecolor(rows[r - 1][4]); cell.set_text_props(fontweight="bold")
    # right: money figure
    axf = fig.add_axes([0.44, 0.05, 0.52, 0.34]); axf.axis("off")
    axf.imshow(plt.imread(os.path.join(HERE, "fig_panel5_maxcrit_gy.png")))

    fig.text(0.06, 0.03, "DRAFT for review — interactive P4 GIF pending a pod run.  "
             "Reproduce: build_astro_panels.py / build_poster_pdf.py", fontsize=8, color="#999")
    pdf.savefig(fig, dpi=200); plt.close(fig)

    # ---- Page 2: figures ----
    fig = plt.figure(figsize=(16, 9)); fig.patch.set_facecolor("white")
    fig.text(0.5, 0.96, "Severity curves", ha="center", fontsize=18, fontweight="bold")
    a1 = fig.add_axes([0.04, 0.14, 0.46, 0.74]); a1.axis("off")
    a1.imshow(plt.imread(os.path.join(HERE, "fig_panel5_maxcrit_gy.png")))
    a1.set_title("Max per-criterion DVH shift (Gy) vs severity", fontsize=11)
    a2 = fig.add_axes([0.52, 0.14, 0.46, 0.74]); a2.axis("off")
    a2.imshow(plt.imread(os.path.join(HERE, "fig_panel5_dvh_pct.png")))
    a2.set_title("Aggregate DVH-score degradation (%) vs severity", fontsize=11)
    fig.text(0.5, 0.08, "P4 (resolution) crosses the 1.0 Gy line between L1 and L2 and keeps "
             "climbing; every other family stays on the floor.", ha="center", fontsize=11)
    pdf.savefig(fig, dpi=200); plt.close(fig)

print(f"Wrote {pdf_path}")
