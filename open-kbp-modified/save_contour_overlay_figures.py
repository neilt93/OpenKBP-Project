#!/usr/bin/env python3
"""Contour-overlay figures for the "is the perturbation contour-safe?" test.

For the subtle-perturbation threat model (see the certified-robustness study): the
clinically relevant question is whether an epsilon small enough to be a real threat is
ALSO small enough that a clinician would contour the CT identically. This script makes
the evidence for a human (Birjoo) to judge.

Per patient x epsilon x slice it saves an axial 3-panel figure:

    [ Original CT + contours ]   [ Adversarial CT + contours ]   [ Perturbation (HU) ]

The SAME structure contours (Brainstem, cord, parotids, PTVs, ...) are drawn on BOTH the
original and the adversarial CT. The masks are model INPUTS and do not change; the test
is whether the CT *under* each contour line has visibly moved — i.e. whether a clinician
re-contouring the right-hand image would draw different lines. At eps=0.02 (~82 HU) the
two images should look identical, which is the point: a threat this subtle is
contour-invariant, so the human-in-the-loop safety net cannot catch it and the
model-level defence / certificate is what matters.

Reuses the attack, float32-rebuild, slice-picking and windowing from
`save_adversarial_ct_figures.py` (identical rendering, just with contours added).

Run on the box (TF 2.18.0, model + data present):
    python save_contour_overlay_figures.py \
        --model models/epoch_100.keras --data-dir $DATA \
        --patient-ids pt_201 pt_205 pt_210 --epsilons 0.02 \
        --attack pgd --n-slices 3 --output contour_overlay_figures/
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model

from adversarial_eval import fgsm_attack, pgd_attack
from provided_code import DataLoader, get_paths
from provided_code.network_architectures import InstanceNormalization
from save_adversarial_ct_figures import (
    CT_MAX, WIN_LO, WIN_HI,
    norm_to_true_hu, to_float32, find_patient, axial,
)

# Distinct colour per ROI (OARs cool/varied, PTVs warm) for the contour overlay.
ROI_COLORS = {
    "Brainstem": "#00e5ff", "SpinalCord": "#18ffb0", "RightParotid": "#ffd400",
    "LeftParotid": "#ffa600", "Esophagus": "#b388ff", "Larynx": "#40c4ff",
    "Mandible": "#e0e0e0", "PTV56": "#ff8a80", "PTV63": "#ff5252", "PTV70": "#ff1744",
}


def ptv_top_slices(masks, roi_list, n):
    """The n S-I indices (W axis) with the largest PTV area — clinically relevant slices."""
    ptv_idx = [i for i, nm in enumerate(roi_list) if "PTV" in nm.upper()]
    if not ptv_idx:
        return [masks.shape[3] // 2]
    ptv = masks[0, ..., ptv_idx].sum(axis=-1)          # (D,H,W)
    area_per_w = ptv.sum(axis=(0, 1))                  # per S-I slice
    order = np.argsort(area_per_w)[::-1]
    return [int(w) for w in order[:n] if area_per_w[w] > 0] or [int(np.argmax(area_per_w))]


def draw_contours(ax, masks_np, roi_list, w, crop_box):
    """Overlay each ROI's boundary on the axial (D,H) slice at S-I index w.

    masks_np is (1,D,H,W,num_rois); crop_box is (d0,d1,h0,h1) matching the CT crop so the
    contour coordinates line up with the imshow'd (and .T'd) CT image.
    """
    d0, d1, h0, h1 = crop_box
    handles = []
    for i, roi in enumerate(roi_list):
        m = masks_np[0, :, :, w, i][d0:d1, h0:h1]      # cropped (D,H) boolean-ish
        if m.sum() == 0:
            continue                                   # ROI absent on this slice
        color = ROI_COLORS.get(roi, "#ffffff")
        # .T to match the CT imshow (L-R horizontal); contour at the 0.5 level = mask edge.
        ax.contour(m.T, levels=[0.5], colors=[color], linewidths=1.0, origin="lower")
        handles.append(plt.Line2D([0], [0], color=color, lw=1.5, label=roi))
    return handles


def main():
    p = argparse.ArgumentParser(description="Contour-overlay adversarial CT figures")
    p.add_argument("--model", required=True)
    p.add_argument("--patient-ids", nargs="+", default=["pt_201"])
    p.add_argument("--epsilons", default="0.02")
    p.add_argument("--attack", choices=["fgsm", "pgd"], default="pgd")
    p.add_argument("--pgd-steps", type=int, default=10)
    p.add_argument("--n-slices", type=int, default=1, help="Top-N PTV-area axial slices per patient")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--output", default="contour_overlay_figures")
    args = p.parse_args()

    epsilons = [float(e) for e in args.epsilons.split(",")]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).parent
    data_dir = Path(args.data_dir) if args.data_dir else script_dir / "provided-data" / "validation-pats"
    if not data_dir.exists():
        data_dir = script_dir.parent / "provided-data" / "validation-pats"

    print(f"Loading model {args.model}")
    tf.keras.mixed_precision.set_global_policy("float32")
    model = load_model(args.model, custom_objects={"InstanceNormalization": InstanceNormalization},
                       compile=False, safe_mode=False)
    model = to_float32(model)  # float32 rebuild -> real (non-NaN) perturbation on CPU too

    for pid in args.patient_ids:
        patient_path = find_patient(data_dir, pid)
        loader = DataLoader([patient_path], batch_size=1, normalize=True, cache_data=True)
        loader.set_mode("training_model")
        batch = next(iter(loader.get_batches()))
        ct = tf.constant(batch.ct, dtype=tf.float32)
        masks = tf.constant(batch.structure_masks, dtype=tf.float32)
        dose = tf.constant(batch.dose, dtype=tf.float32)
        ct_np = batch.ct
        roi_list = loader.full_roi_list

        for eps in epsilons:
            adv = (pgd_attack(model, ct, masks, dose, eps, args.pgd_steps) if args.attack == "pgd"
                   else fgsm_attack(model, ct, masks, dose, eps)).numpy()
            print(f"  {pid} eps={eps}: computed {args.attack} perturbation")

            for w in ptv_top_slices(batch.structure_masks, roi_list, args.n_slices):
                # Body bounding box on this slice -> crop so anatomy fills the frame.
                body = axial(ct_np, w) > 0.12
                ys, xs = np.where(body)
                mg = 6
                if len(ys):
                    d0 = max(int(ys.min()) - mg, 0); d1 = min(int(ys.max()) + mg + 1, body.shape[0])
                    h0 = max(int(xs.min()) - mg, 0); h1 = min(int(xs.max()) + mg + 1, body.shape[1])
                else:
                    d0, d1, h0, h1 = 0, body.shape[0], 0, body.shape[1]
                crop_box = (d0, d1, h0, h1)
                def crop(a): return a[d0:d1, h0:h1]

                orig_hu = crop(norm_to_true_hu(axial(ct_np, w)))
                adv_hu = crop(norm_to_true_hu(axial(adv, w)))
                delta_hu = crop((axial(adv, w) - axial(ct_np, w)) * CT_MAX) * crop(body).astype(np.float32)
                vlim = eps * CT_MAX

                fig, ax = plt.subplots(1, 3, figsize=(13, 4.4))
                # Panel 0 + 1: CT with identical contours; Panel 2: perturbation in HU.
                ax[0].imshow(orig_hu.T, origin="lower", cmap="gray", vmin=WIN_LO, vmax=WIN_HI, interpolation="bilinear")
                ax[0].set_title("Original CT + contours", fontsize=12)
                handles = draw_contours(ax[0], batch.structure_masks, roi_list, w, crop_box)

                ax[1].imshow(adv_hu.T, origin="lower", cmap="gray", vmin=WIN_LO, vmax=WIN_HI, interpolation="bilinear")
                ax[1].set_title(f"Adversarial CT + contours (eps={eps:g}, ~{eps*CT_MAX:.0f} HU)", fontsize=12)
                draw_contours(ax[1], batch.structure_masks, roi_list, w, crop_box)

                im = ax[2].imshow(delta_hu.T, origin="lower", cmap="seismic", vmin=-vlim, vmax=vlim, interpolation="bilinear")
                ax[2].set_title("Perturbation (HU)", fontsize=12)
                plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)

                for a in ax:
                    a.set_xticks([]); a.set_yticks([])
                if handles:
                    ax[0].legend(handles=handles, loc="upper left", fontsize=6.5,
                                 framealpha=0.4, labelcolor="white")
                fig.suptitle(f"{pid}  |  {args.attack.upper()} eps={eps:g}  |  axial S-I slice {w}  "
                             f"— same contours on both CTs; do they still fit the right image?", fontsize=11)
                fig.tight_layout()
                stem = f"{pid}_{args.attack}_eps{eps:g}_slice{w}_contours"
                fig.savefig(out / f"{stem}.png", dpi=200, bbox_inches="tight")
                plt.close(fig)
                print(f"    saved {stem}.png")

    print(f"\nSaved contour-overlay figures to {out}/")
    return 0


if __name__ == "__main__":
    exit(main())
