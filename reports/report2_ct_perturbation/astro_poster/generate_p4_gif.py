#!/usr/bin/env python3
"""Panel 6 interactive element: step one patient through the P4 resolution sweep.

Renders, for ONE representative patient, an animated GIF of the predicted dose across
P4 L0 -> L4 (plus baseline): a dose wash over the CT slice on the left and the Larynx /
PTV70 cumulative DVH on the right (baseline dashed vs current solid). Also dumps the
individual frame PNGs so the sequence survives even without GIF assembly.

RUNS ON THE POD / GPU BOX -- needs TensorFlow, the trained model, and the validation
patient data. It reuses the exact verified inference path from run_inference.py
(DataLoader dose_prediction mode + denormalize * prescription) and the committed
perturbation classes, so the GIF is consistent with the sweep numbers in the poster.

Example:
  cd open-kbp-modified/openkbp_hn_robustness
  python ../../reports/report2_ct_perturbation/astro_poster/generate_p4_gif.py \
      --model /workspace/results/<best>/models/epoch_100.keras \
      --data-dir /tmp/okbp-data/provided-data/validation-pats \
      --patient pt_205 --fps 2
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

# --- make the project code importable regardless of CWD ------------------------
HERE = Path(__file__).resolve().parent
OKBP = HERE.parents[2] / "open-kbp-modified"          # .../open-kbp-modified
ROBUST = OKBP / "openkbp_hn_robustness"
for p in (str(OKBP), str(ROBUST)):
    if p not in sys.path:
        sys.path.insert(0, p)

from perturbations import ResolutionDegradation                      # noqa: E402
from perturbations.base import (                                     # noqa: E402
    load_ct_volume, load_structure_mask, create_perturbed_patient)

DOSE_PRESCRIPTION = 70.0
DVH_STRUCTURES = ["Larynx", "PTV70"]   # Larynx = the sentinel; PTV70 = target context


def predict_dose(model, patient_dir: Path):
    """Return (dose_gy, ct_norm, pdm) for one patient dir via the verified path."""
    from provided_code.data_loader import DataLoader
    loader = DataLoader([patient_dir], batch_size=1, normalize=True, cache_data=False)
    loader.set_mode("dose_prediction")
    batch = next(iter(loader.get_batches()))
    dose = model.predict([batch.ct, batch.structure_masks], verbose=0)
    dose = np.squeeze(dose * batch.possible_dose_mask) * DOSE_PRESCRIPTION
    ct = np.squeeze(batch.ct)
    pdm = np.squeeze(batch.possible_dose_mask)
    return dose, ct, pdm


def cumulative_dvh(dose_gy, mask, n=200):
    """Cumulative DVH: x = dose (Gy), y = % of structure volume receiving >= x."""
    vals = dose_gy[mask.astype(bool)]
    if vals.size == 0:
        return np.array([0.0]), np.array([0.0])
    x = np.linspace(0, max(vals.max(), 1.0), n)
    y = [(vals >= d).mean() * 100.0 for d in x]
    return x, np.array(y)


def main():
    ap = argparse.ArgumentParser(description="P4 severity-sweep GIF for one patient")
    ap.add_argument("--model", required=True, help="Path to trained .keras model")
    ap.add_argument("--data-dir", required=True, help="validation-pats dir")
    ap.add_argument("--patient", default=None, help="Patient id (default: first in data-dir)")
    ap.add_argument("--out", default=str(HERE / "p4_gif"), help="Output dir for frames + gif")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--slice", default="auto", help="Axial slice index or 'auto' (larynx centroid)")
    args = ap.parse_args()

    import tensorflow as tf
    from provided_code.network_architectures import InstanceNormalization

    data_dir = Path(args.data_dir)
    pdir = (data_dir / args.patient) if args.patient else sorted(data_dir.iterdir())[0]
    if not pdir.exists():
        sys.exit(f"patient dir not found: {pdir}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"patient={pdir.name}  model={args.model}")

    model = tf.keras.models.load_model(
        args.model, custom_objects={"InstanceNormalization": InstanceNormalization},
        compile=False, safe_mode=False)

    ct_hu, body_mask = load_ct_volume(pdir)
    masks = {s: load_structure_mask(pdir, s) for s in DVH_STRUCTURES}

    # slice through the larynx (the sentinel structure) unless overridden
    if args.slice == "auto":
        lm = masks["Larynx"]
        z = int(np.round(np.argwhere(lm).mean(0)[0])) if lm.any() else ct_hu.shape[0] // 2
    else:
        z = int(args.slice)

    pert = ResolutionDegradation()
    levels = ["baseline"] + list(pert.levels.keys())   # baseline, L0..L4
    frames = []
    base_dvh = None
    with tempfile.TemporaryDirectory() as td:
        for lvl in levels:
            if lvl == "baseline":
                ct_pert = ct_hu
            else:
                ct_pert = pert.apply(ct_hu.copy(), body_mask, lvl, np.random.default_rng(0))
            dst = Path(td) / f"{pdir.name}_{lvl}"
            create_perturbed_patient(pdir, dst, ct_pert)
            dose, ctn, pdm = predict_dose(model, dst)
            dvh = {s: cumulative_dvh(dose, masks[s]) for s in DVH_STRUCTURES}
            if lvl == "baseline":
                base_dvh = dvh
            frames.append((lvl, dose, ctn, dvh))
            print(f"  predicted {lvl}")

    # --- render -----------------------------------------------------------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5))
    writer = PillowWriter(fps=args.fps)
    gif_path = out / f"p4_sweep_{pdir.name}.gif"
    colors = {"Larynx": "#c0392b", "PTV70": "#2980b9"}
    with writer.saving(fig, str(gif_path), dpi=120):
        for i, (lvl, dose, ctn, dvh) in enumerate(frames):
            axL.clear(); axR.clear()
            axL.imshow(ctn[z], cmap="gray")
            m = np.ma.masked_where(dose[z] <= 0.5, dose[z])
            im = axL.imshow(m, cmap="jet", alpha=0.55, vmin=0, vmax=DOSE_PRESCRIPTION)
            axL.set_title(f"Predicted dose — P4 {lvl}"); axL.axis("off")
            for s in DVH_STRUCTURES:
                bx, by = base_dvh[s]; axR.plot(bx, by, "--", color=colors[s], alpha=0.5,
                                               label=f"{s} baseline")
                cx, cy = dvh[s]; axR.plot(cx, cy, "-", color=colors[s], lw=2.2, label=f"{s} {lvl}")
            axR.set_xlabel("Dose (Gy)"); axR.set_ylabel("% volume")
            axR.set_title("Cumulative DVH"); axR.set_ylim(0, 100); axR.grid(alpha=0.25)
            axR.legend(fontsize=8, loc="lower left")
            fig.tight_layout()
            fig.savefig(out / f"frame_{i}_{lvl}.png", dpi=120)
            writer.grab_frame()
    plt.close(fig)
    print(f"\nWrote GIF: {gif_path}\nFrames: {out}/frame_*.png")
    print("Caption: 'Loss of edge definition degrades dose prediction near structure boundaries.'")


if __name__ == "__main__":
    main()
