#!/usr/bin/env python3
"""Regenerate the P4 (resolution) severity-progression figures as TRUE AXIAL slices (axis 2, S-I),
same recipe as regen_axial_figures.py: max-PTV-area slice, body-bbox crop, soft-tissue window.
CPU only; reads SanDisk warehouse CTs, perturbed CTs, and saved prediction CSVs.
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
PID = "pt_201"
LEVELS = ["L0", "L1", "L2", "L3", "L4"]
LEVEL_PARAMS = ["0.5/0.25", "1.0/0.5", "2.0/1.0", "3.0/1.5", "4.0/2.0"]
CT_MAX, HU_OFFSET = 4095.0, 1024.0
WIN_LO, WIN_HI = 40.0 - 200.0, 40.0 + 200.0
OUT = os.path.dirname(os.path.abspath(__file__))


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
    area = np.zeros(SHAPE[2])
    for name in ("PTV56", "PTV63", "PTV70"):
        p = f"{VAL}/{PID}/{name}.csv"
        if os.path.exists(p):
            area += load_mask(p).sum(axis=(0, 1))
    return int(np.argmax(area)) if area.any() else SHAPE[2] // 2


def main():
    w = ptv_slice_index()
    print(f"axial S-I slice (max PTV area) = {w}")
    base_ct = load_vol(f"{VAL}/{PID}/ct.csv")
    base_dose = load_vol(f"{PRED}/baseline/{PID}.csv")

    ax = lambda vol: vol[:, :, w]
    body = ax(base_ct) / CT_MAX > 0.12
    ys, xs = np.where(body)
    mg = 6
    d0, d1 = max(ys.min() - mg, 0), min(ys.max() + mg + 1, body.shape[0])
    h0, h1 = max(xs.min() - mg, 0), min(xs.max() + mg + 1, body.shape[1])
    crop = lambda a: a[d0:d1, h0:h1]
    hu = lambda a: crop(ax(a)) - HU_OFFSET
    show = lambda axi, img, **kw: axi.imshow(img, origin="upper", interpolation="bilinear", **kw)
    body_c = crop(body).astype(float)

    # ---- CT progression: baseline + P4 L0..L4 (top), CT difference (bottom) ----
    n = len(LEVELS) + 1
    fig, A = plt.subplots(2, n, figsize=(2.6 * n, 6))
    show(A[0, 0], hu(base_ct), cmap="gray", vmin=WIN_LO, vmax=WIN_HI)
    A[0, 0].set_title("Original")
    A[1, 0].set_ylabel("Difference (HU)")
    A[1, 0].imshow(np.zeros_like(hu(base_ct)), origin="upper", cmap="seismic", vmin=-120, vmax=120)
    for r in (0, 1):
        A[r, 0].set_xticks([]); A[r, 0].set_yticks([])
    for i, (lev, par) in enumerate(zip(LEVELS, LEVEL_PARAMS)):
        c = i + 1
        pv = load_vol(f"{PERT}/P4_resolution/{lev}/{PID}/ct.csv")
        show(A[0, c], hu(pv), cmap="gray", vmin=WIN_LO, vmax=WIN_HI)
        show(A[1, c], (crop(ax(pv)) - crop(ax(base_ct))) * body_c,
             cmap="seismic", vmin=-120, vmax=120)
        A[0, c].set_title(f"{lev} ({par} vox)")
        for r in (0, 1):
            A[r, c].set_xticks([]); A[r, c].set_yticks([])
    fig.suptitle(f"Resolution Loss Severity Progression, CT — {PID} (axial slice {w})", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{OUT}/p4_ct_progression.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Dose progression: baseline + P4 L0..L4 predicted dose (top), dose diff (bottom) ----
    dmax = float(np.percentile(base_dose[base_dose > 0], 99))
    doseax = lambda a: crop(ax(a))
    fig, A = plt.subplots(2, n, figsize=(2.6 * n, 6))
    show(A[0, 0], doseax(base_dose), cmap="jet", vmin=0, vmax=dmax)
    A[0, 0].set_title("Baseline")
    A[1, 0].set_ylabel("Dose diff (Gy)")
    A[1, 0].imshow(np.zeros_like(doseax(base_dose)), origin="upper", cmap="RdBu_r", vmin=-5, vmax=5)
    for r in (0, 1):
        A[r, 0].set_xticks([]); A[r, 0].set_yticks([])
    for i, (lev, par) in enumerate(zip(LEVELS, LEVEL_PARAMS)):
        c = i + 1
        pdose = load_vol(f"{PRED}/P4_resolution/{lev}/{PID}.csv")
        show(A[0, c], doseax(pdose), cmap="jet", vmin=0, vmax=dmax)
        show(A[1, c], doseax(pdose) - doseax(base_dose), cmap="RdBu_r", vmin=-5, vmax=5)
        A[0, c].set_title(f"{lev} ({par} vox)")
        for r in (0, 1):
            A[r, c].set_xticks([]); A[r, c].set_yticks([])
    fig.suptitle(f"Resolution Loss Severity Progression, Predicted Dose — {PID} (axial slice {w})",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(f"{OUT}/p4_dose_progression.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/p4_ct_progression.png and {OUT}/p4_dose_progression.png")


if __name__ == "__main__":
    main()
