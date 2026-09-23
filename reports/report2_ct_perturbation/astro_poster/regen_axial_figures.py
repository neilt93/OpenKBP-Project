#!/usr/bin/env python3
"""Regenerate ct_slices + dose_difference_maps as TRUE AXIAL slices, matching the AAPM adversarial
figure's exact orientation/recipe (save_adversarial_ct_figures.py), no GPU.

Reads OpenKBP CTs, perturbed CTs, PTV masks, and the already-computed prediction CSVs from the
SanDisk warehouse (CPU only). Axes (D,H,W)=(A-P,L-R,S-I); axial slice = (D,H) at the max-PTV-area
S-I index; crop to the body bbox; display in conventional radiological AXIAL orientation
(anterior up, patient-left on the right).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SD = "/Volumes/Neil's SanDisk/OpenKBP-Warehouse/open-kbp-modified"
VAL = f"{SD}/provided-data/validation-pats"
PERT = f"{SD}/openkbp_hn_robustness/data_perturbed"
PRED = f"{SD}/openkbp_hn_robustness/predictions"
SHAPE = (128, 128, 128)
PID, LEVEL = "pt_201", "L2"
CT_MAX, HU_OFFSET = 4095.0, 1024.0
WIN_LO, WIN_HI = 40.0 - 200.0, 40.0 + 200.0        # soft-tissue window (true HU), as in the AAPM fig
FAMILIES = [("P1_noise", "Acq. Noise"), ("P2_bone_shift", "Bone Shift"),
            ("P3_bias_field", "Bias Field"), ("P4_resolution", "Resolution"),
            ("P5_dental", "Dental Art.")]
OUT = "/tmp"


def load_vol(path):
    df = pd.read_csv(path, index_col=0)
    v = np.zeros(int(np.prod(SHAPE)))
    v[df.index.values] = df["data"].values
    return v.reshape(SHAPE)


def load_mask(path):
    df = pd.read_csv(path, index_col=0)
    m = np.zeros(int(np.prod(SHAPE)), dtype=bool)
    m[np.array(df.index).squeeze()] = True
    return m.reshape(SHAPE)


def ptv_slice_index():
    """S-I index (axis 2) with the largest total PTV area — the clinically relevant axial slice."""
    area = np.zeros(SHAPE[2])
    for name in ("PTV56", "PTV63", "PTV70"):
        p = f"{VAL}/{PID}/{name}.csv"
        if os.path.exists(p):
            area += load_mask(p).sum(axis=(0, 1))
    return int(np.argmax(area)) if area.any() else SHAPE[2] // 2


def main():
    w = ptv_slice_index()
    print(f"axial S-I slice (max PTV area) = {w}")
    base_ct = load_vol(f"{VAL}/{PID}/ct.csv")           # stored HU (trueHU + 1024)
    base_dose = load_vol(f"{PRED}/baseline/{PID}.csv")  # Gy

    def ax(vol):
        return vol[:, :, w]                             # (D,H) axial slice

    # crop to the body bbox (drop the air border) — computed from the CT
    body = ax(base_ct) / CT_MAX > 0.12
    ys, xs = np.where(body)
    mg = 6
    d0, d1 = max(ys.min() - mg, 0), min(ys.max() + mg + 1, body.shape[0])
    h0, h1 = max(xs.min() - mg, 0), min(xs.max() + mg + 1, body.shape[1])
    crop = lambda a: a[d0:d1, h0:h1]
    hu = lambda a: crop(ax(a)) - HU_OFFSET              # display in true HU
    # Conventional radiological AXIAL orientation: rows = A-P (anterior at top, since mandible sits
    # at low axis0 and cord at high axis0), cols = L-R (patient-left at right, parotid geometry).
    # No transpose (the adversarial code's img.T rotated the axial plane 90°, which read as coronal).
    show = lambda axi, img, **kw: axi.imshow(img, origin="upper", interpolation="bilinear", **kw)
    body_c = crop(body).astype(float)

    # ---- CT slices ----
    n = len(FAMILIES) + 1
    fig, A = plt.subplots(2, n, figsize=(2.6 * n, 6))
    show(A[0, 0], hu(base_ct), cmap="gray", vmin=WIN_LO, vmax=WIN_HI)
    A[0, 0].set_title("Original"); A[1, 0].set_ylabel("Difference (HU)")
    for r in (0, 1):
        A[r, 0].set_xticks([]); A[r, 0].set_yticks([])
    A[1, 0].imshow(np.zeros_like(hu(base_ct)), origin="upper", cmap="seismic", vmin=-80, vmax=80)
    for i, (fam, lab) in enumerate(FAMILIES):
        c = i + 1
        pv = load_vol(f"{PERT}/{fam}/{LEVEL}/{PID}/ct.csv")
        show(A[0, c], hu(pv), cmap="gray", vmin=WIN_LO, vmax=WIN_HI)
        diff = (crop(ax(pv)) - crop(ax(base_ct))) * body_c       # HU difference within body
        show(A[1, c], diff, cmap="seismic", vmin=-80, vmax=80)
        A[0, c].set_title(f"{lab} {LEVEL}")
        for r in (0, 1):
            A[r, c].set_xticks([]); A[r, c].set_yticks([])
    fig.suptitle(f"Example CT Slices — {PID} (axial slice {w})", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{OUT}/ct_slices_axial.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Dose predictions ----
    dmax = float(np.percentile(base_dose[base_dose > 0], 99))
    doseax = lambda a: crop(ax(a))
    fig, A = plt.subplots(2, n, figsize=(2.6 * n, 6))
    show(A[0, 0], doseax(base_dose), cmap="jet", vmin=0, vmax=dmax)
    A[0, 0].set_title("Baseline"); A[1, 0].set_ylabel("Dose diff (Gy)")
    for r in (0, 1):
        A[r, 0].set_xticks([]); A[r, 0].set_yticks([])
    A[1, 0].imshow(np.zeros_like(doseax(base_dose)), origin="upper", cmap="RdBu_r", vmin=-5, vmax=5)
    for i, (fam, lab) in enumerate(FAMILIES):
        c = i + 1
        pdose = load_vol(f"{PRED}/{fam}/{LEVEL}/{PID}.csv")
        show(A[0, c], doseax(pdose), cmap="jet", vmin=0, vmax=dmax)
        show(A[1, c], doseax(pdose) - doseax(base_dose), cmap="RdBu_r", vmin=-5, vmax=5)
        A[0, c].set_title(f"{lab} {LEVEL}")
        for r in (0, 1):
            A[r, c].set_xticks([]); A[r, c].set_yticks([])
    fig.suptitle(f"Dose Predictions — {PID} (axial slice {w})", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{OUT}/dose_difference_axial.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/ct_slices_axial.png and {OUT}/dose_difference_axial.png")


if __name__ == "__main__":
    main()
