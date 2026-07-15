#!/usr/bin/env python3
"""Render CT slices with FGSM / PGD adversarial noise inserted — for the AAPM poster.

For a chosen patient it computes the real gradient-based FGSM and PGD perturbations on
the trained model, then saves, per (attack, epsilon), a 3-panel axial slice:

    [ Original CT ]   [ Perturbation (HU) ]   [ Adversarial CT ]

The perturbation panel (diverging colormap) is the money shot: it shows the actual
adversarial noise the attack adds. The adversarial CT looks near-identical to the
original — that is the point (small, near-imperceptible perturbation, large dose effect).
Also writes a combined grid PNG + PDF with all conditions.

Axes: the loader sample is (D,H,W,C) with D=A-P, H=L-R, W=S-I. The axial (radiologist)
view is the (D,H) plane at a fixed S-I index; we pick the slice with the most PTV.

Run on the box (TF 2.18.0, GPU), model already on the volume:
    python save_adversarial_ct_figures.py \
        --model models_in/epoch_100.keras \
        --patient-id pt_201 \
        --attacks fgsm pgd --epsilons 0.02,0.05 \
        --output adversarial_ct_figures/
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

CT_MAX = 4095.0      # normalised CT in [0,1] == stored HU / 4095
HU_OFFSET = 1024.0   # OpenKBP stores trueHU + 1024
# Soft-tissue display window (true HU): level 40, width 400.
WIN_LO, WIN_HI = 40.0 - 200.0, 40.0 + 200.0


def norm_to_true_hu(x):
    return x * CT_MAX - HU_OFFSET


def find_patient(data_dir: Path, pid: str) -> Path:
    for p in sorted(get_paths(data_dir)):
        if Path(p).name == pid:
            return Path(p)
    raise SystemExit(f"patient {pid} not found under {data_dir}")


def ptv_slice_index(masks, roi_list):
    """S-I index (W axis) with the largest PTV area, for a clinically relevant slice."""
    ptv_idx = [i for i, n in enumerate(roi_list) if "PTV" in n.upper()]
    if not ptv_idx:
        return masks.shape[3] // 2
    ptv = masks[0, ..., ptv_idx].sum(axis=-1)   # (D,H,W)
    area_per_w = ptv.sum(axis=(0, 1))           # over D,H -> per S-I slice
    return int(np.argmax(area_per_w))


def axial(vol, w):
    """(D,H) axial slice at S-I index w from a (1,D,H,W,1) or (D,H,W) array."""
    a = vol[0, ..., 0] if vol.ndim == 5 else vol
    return a[:, :, w]


def main():
    p = argparse.ArgumentParser(description="Save FGSM/PGD adversarial CT slice figures")
    p.add_argument("--model", required=True)
    p.add_argument("--patient-id", default="pt_201")
    p.add_argument("--attacks", nargs="+", choices=["fgsm", "pgd"], default=["fgsm", "pgd"])
    p.add_argument("--epsilons", default="0.02,0.05")
    p.add_argument("--pgd-steps", type=int, default=10)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--output", default="adversarial_ct_figures")
    args = p.parse_args()

    epsilons = [float(e) for e in args.epsilons.split(",")]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).parent
    data_dir = Path(args.data_dir) if args.data_dir else script_dir / "provided-data" / "validation-pats"
    if not data_dir.exists():
        data_dir = script_dir.parent / "provided-data" / "validation-pats"

    print(f"Loading model {args.model}")
    model = load_model(args.model, custom_objects={"InstanceNormalization": InstanceNormalization},
                       compile=False, safe_mode=False)

    patient_path = find_patient(data_dir, args.patient_id)
    loader = DataLoader([patient_path], batch_size=1, normalize=True, cache_data=True)
    loader.set_mode("training_model")
    batch = next(iter(loader.get_batches()))
    ct = tf.constant(batch.ct, dtype=tf.float32)
    masks = tf.constant(batch.structure_masks, dtype=tf.float32)
    dose = tf.constant(batch.dose, dtype=tf.float32)

    w = ptv_slice_index(batch.structure_masks, loader.full_roi_list)
    ct_np = batch.ct
    print(f"patient {args.patient_id}: axial S-I slice {w} (max PTV area)")

    # Compute every (attack, eps) adversarial CT.
    conds = []  # (label, attack, eps, adv_np)
    for atk in args.attacks:
        for eps in epsilons:
            adv = (pgd_attack(model, ct, masks, dose, eps, args.pgd_steps) if atk == "pgd"
                   else fgsm_attack(model, ct, masks, dose, eps)).numpy()
            conds.append((f"{atk.upper()} eps={eps:g} (~{eps*CT_MAX:.0f} HU)", atk, eps, adv))
            print(f"  computed {atk} eps={eps}")

    orig_hu = norm_to_true_hu(axial(ct_np, w))

    # Per-condition 3-panel figures + a combined grid.
    nrows = len(conds)
    fig, axes = plt.subplots(nrows, 3, figsize=(11, 3.4 * nrows), squeeze=False)
    for r, (label, atk, eps, adv) in enumerate(conds):
        adv_hu = norm_to_true_hu(axial(adv, w))
        delta_hu = (axial(adv, w) - axial(ct_np, w)) * CT_MAX  # perturbation in HU
        vlim = eps * CT_MAX
        panels = [
            ("Original CT", orig_hu, dict(cmap="gray", vmin=WIN_LO, vmax=WIN_HI)),
            ("Perturbation (HU)", delta_hu, dict(cmap="seismic", vmin=-vlim, vmax=vlim)),
            ("Adversarial CT", adv_hu, dict(cmap="gray", vmin=WIN_LO, vmax=WIN_HI)),
        ]
        for c, (title, img, kw) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(img.T, origin="lower", **kw)   # .T so L-R is horizontal
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=12)
            if c == 0:
                ax.set_ylabel(label, fontsize=11)
            if c == 1:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # also save the standalone perturbation + adversarial panel for this condition
        fig1, a1 = plt.subplots(1, 3, figsize=(11, 3.6))
        for c, (title, img, kw) in enumerate(panels):
            im = a1[c].imshow(img.T, origin="lower", **kw)
            a1[c].set_title(title, fontsize=12); a1[c].set_xticks([]); a1[c].set_yticks([])
            if c == 1:
                plt.colorbar(im, ax=a1[c], fraction=0.046, pad=0.04)
        fig1.suptitle(f"{args.patient_id}  |  {label}", fontsize=12)
        fig1.tight_layout()
        stem = f"{args.patient_id}_{atk}_eps{eps:g}"
        fig1.savefig(out / f"{stem}.png", dpi=200, bbox_inches="tight")
        plt.close(fig1)

    fig.suptitle(f"Adversarial CT perturbations — {args.patient_id} (axial slice {w})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / f"{args.patient_id}_adversarial_ct_grid.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{args.patient_id}_adversarial_ct_grid.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figures to {out}/  (per-condition PNGs + combined grid PNG/PDF)")
    return 0


if __name__ == "__main__":
    exit(main())
