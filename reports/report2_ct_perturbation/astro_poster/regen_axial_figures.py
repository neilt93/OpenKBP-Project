#!/usr/bin/env python3
"""Regenerate ct_slices + dose_difference_maps as TRUE AXIAL (transverse) slices, no GPU.

Reads OpenKBP CTs, perturbed CTs, and the already-computed prediction CSVs straight from the
SanDisk warehouse (CPU only, pandas/matplotlib). OpenKBP raw axes are (A-P, L-R, S-I); an axial
slice fixes S-I (axis 2). We pick the S-I slice carrying the most dose so the panel shows the
treated region. Outputs to /tmp first for visual orientation check.
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
LEVEL = "L2"
FAMILIES = [("P1_noise", "Acq. Noise"), ("P2_bone_shift", "Bone Shift"),
            ("P3_bias_field", "Bias Field"), ("P4_resolution", "Resolution"),
            ("P5_dental", "Dental Art.")]
OUT = "/tmp"


def load_csv_vol(path):
    df = pd.read_csv(path, index_col=0)
    v = np.zeros(int(np.prod(SHAPE)))
    v[df.index.values] = df["data"].values
    return v.reshape(SHAPE)


def axial(vol, idx):
    """Axial (transverse) slice: fix S-I (axis 2) -> (A-P, L-R). flipud puts anterior at top."""
    return np.flipud(vol[:, :, idx])


def main():
    base_ct = load_csv_vol(f"{VAL}/{PID}/ct.csv")
    base_dose = load_csv_vol(f"{PRED}/baseline/{PID}.csv")
    # choose the axial (S-I) index with the most dose -> shows the treated region
    z = int(np.argmax(base_dose.sum(axis=(0, 1))))
    print(f"axial S-I index (max-dose) = {z}")

    # ---- CT slices ----
    fig, ax = plt.subplots(2, len(FAMILIES) + 1, figsize=(3 * (len(FAMILIES) + 1), 7))
    ax[0, 0].imshow(axial(base_ct, z), cmap="gray", vmin=0, vmax=2000, aspect="equal")
    ax[0, 0].set_title("Original"); ax[0, 0].axis("off")
    ax[1, 0].axis("off"); ax[1, 0].set_title("Diff")
    for i, (fam, lab) in enumerate(FAMILIES):
        c = i + 1
        pv = load_csv_vol(f"{PERT}/{fam}/{LEVEL}/{PID}/ct.csv")
        ax[0, c].imshow(axial(pv, z), cmap="gray", vmin=0, vmax=2000, aspect="equal")
        ax[1, c].imshow(axial(pv, z) - axial(base_ct, z), cmap="RdBu_r", vmin=-200, vmax=200,
                        aspect="equal")
        ax[0, c].set_title(f"{lab} {LEVEL}"); ax[0, c].axis("off"); ax[1, c].axis("off")
    fig.suptitle(f"Example CT Slices — {PID} (axial slice {z})", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{OUT}/ct_slices_axial.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- Dose predictions ----
    dmax = float(np.percentile(base_dose[base_dose > 0], 99)) if base_dose.max() > 0 else 70
    fig, ax = plt.subplots(2, len(FAMILIES) + 1, figsize=(3 * (len(FAMILIES) + 1), 7))
    ax[0, 0].imshow(axial(base_dose, z), cmap="jet", vmin=0, vmax=dmax, aspect="equal")
    ax[0, 0].set_title("Baseline"); ax[0, 0].axis("off")
    ax[1, 0].axis("off"); ax[1, 0].set_title("Diff")
    for i, (fam, lab) in enumerate(FAMILIES):
        c = i + 1
        pd_ = load_csv_vol(f"{PRED}/{fam}/{LEVEL}/{PID}.csv")
        ax[0, c].imshow(axial(pd_, z), cmap="jet", vmin=0, vmax=dmax, aspect="equal")
        ax[1, c].imshow(axial(pd_, z) - axial(base_dose, z), cmap="RdBu_r", vmin=-5, vmax=5,
                        aspect="equal")
        ax[0, c].set_title(f"{lab} {LEVEL}"); ax[0, c].axis("off"); ax[1, c].axis("off")
    fig.suptitle(f"Dose Predictions — {PID} (axial slice {z})", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{OUT}/dose_difference_axial.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/ct_slices_axial.png and {OUT}/dose_difference_axial.png")


if __name__ == "__main__":
    main()
