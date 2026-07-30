#!/usr/bin/env python3
"""Certified robustness for photon dose prediction via median randomised smoothing.

The principled completion of the empirical noise defence (`adversarial_defense.py`):
that defence averaged the prediction over Gaussian CT noise and RECOVERED attack
damage, but with NO guarantee — an adaptive attacker can chase it (see
`adversarial_adaptive.py`). This script turns the same noise into a PROVABLE
per-voxel certificate.

Median (percentile) smoothing, not mean smoothing, because dose prediction is
dense voxel-wise regression (Chiang et al. 2020). The certificate math and its
finite-sample order-statistic construction live in
`provided_code/smoothing_certify.py` (unit-tested off-GPU); this script owns only
the TF Monte-Carlo (draw n noisy CTs, run the model) and the clinical read-out.

What it produces, per L2 radius R and per patient:
  * per-voxel certified dose interval [lower, upper] in Gy: no CT perturbation with
    ||delta||_2 <= R can move the smoothed-median dose at any voxel outside it,
    with confidence 1 - alpha;
  * the clinical summary: fraction of in-body voxels certified within a tolerance
    (default 1 Gy), and the certified DVH intervals (guaranteed bounds on D95 /
    mean OAR dose / etc.), obtained by pushing the lower/upper dose volumes through
    the OpenKBP DVH metrics — valid because every metric is monotone in voxel dose.

Radius units. R is an L2 norm over the whole normalised-CT input vector (V voxels).
Reported alongside R: its per-voxel RMS equivalent R/sqrt(V) and the HU equivalent
(x CT_MAX=4095), so the guarantee is stated in terms a physicist can sanity-check
against scanner noise, not as an abstract ball. As in randomised smoothing
generally, a whole-volume L2 ball spreads thin per voxel — that honest tension is
part of the result, not hidden.

Base model. A first pass certifies the EXISTING best model directly (it already
saw intensity augmentation). The known lift is to fine-tune the base network on
Gaussian-noised CTs first (SmoothAdv / Salman et al.) so it predicts well UNDER
the smoothing noise, widening the certified radius — that is the phase-2 training
run, driven from `runpod_train.py`, then re-certified here.

Run on the box (TF 2.18.0, GPU):
    python certify_smoothing.py \
        --model results/.../models/epoch_100.keras \
        --sigma 0.05 --n-samples 100 --radii 0.5,1.0,2.0 \
        --tol-gy 1.0 --n-patients 10 --output certify_results/
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from provided_code import DataLoader, DoseEvaluator, get_paths
from provided_code.network_architectures import InstanceNormalization
from provided_code.smoothing_certify import (
    certify_from_samples,
    per_voxel_rms_equivalent,
)


def draw_smoothed_samples(model, ct, masks, sigma, n_samples, batch_draws, rng):
    """Return (n_samples, V) normalised-dose predictions on Gaussian-noised CTs.

    Noise std `sigma` is in normalised CT units (x4095 ~ HU). Noisy CTs are clipped
    to [0, 1] to stay in the model's input range (standard smoothing practice;
    the clip is a mild, universally-used deviation from the exact Gaussian model).
    Draws are batched (`batch_draws` at a time) to keep the forward pass efficient.
    """
    ct0 = ct.numpy()[0]                       # (D, H, W, 1)
    V = ct0.size
    out = np.empty((n_samples, V), dtype=np.float32)
    done = 0
    while done < n_samples:
        b = min(batch_draws, n_samples - done)
        noise = rng.normal(0.0, sigma, size=(b, *ct0.shape)).astype(np.float32)
        noisy = np.clip(ct0[None, ...] + noise, 0.0, 1.0)
        masks_b = tf.repeat(masks, b, axis=0)
        pred = model([tf.constant(noisy), masks_b], training=False).numpy()  # (b,D,H,W,1)
        out[done:done + b] = pred.reshape(b, V)
        done += b
    return out


def certified_dvh_intervals(evaluator, batch, lower_gy_flat, upper_gy_flat):
    """Certified [lower, upper] for every DVH metric, via monotone push-through.

    Each OpenKBP DVH metric is monotonically non-decreasing in per-voxel dose, so
    the per-voxel lower-bound volume yields a lower bound on every metric and the
    upper-bound volume an upper bound — no extra confidence budget. Returns a dict
    {"metric|roi": [lo, hi]}.
    """
    evaluator.reference_batch = batch
    lo_df = evaluator._calculate_dvh_metrics(evaluator.reference_dvh_metrics_df.copy(), lower_gy_flat)
    hi_df = evaluator._calculate_dvh_metrics(evaluator.prediction_dvh_metrics_df.copy(), upper_gy_flat)
    pid = batch.patient_list[0]
    out = {}
    for (metric, roi) in lo_df.columns:
        lo = lo_df.at[pid, (metric, roi)]
        hi = hi_df.at[pid, (metric, roi)]
        if lo is None or hi is None or (isinstance(lo, float) and np.isnan(lo)):
            continue
        out[f"{metric}|{roi}"] = [float(lo), float(hi)]
    return out


def run(model, loader, sigma, n_samples, radii, tol_gy, alpha, batch_draws, seed):
    presc = loader.DOSE_PRESCRIPTION
    per_patient = []
    evaluator = DoseEvaluator(loader)  # reused for the monotone DVH push-through

    n_done = 0
    for batch in loader.get_batches():
        n_done += 1
        ct_t = tf.constant(batch.ct, dtype=tf.float32)
        masks_t = tf.constant(batch.structure_masks, dtype=tf.float32)
        pdm = batch.possible_dose_mask.astype(bool).flatten()
        rng = np.random.default_rng(seed)  # per-patient reset -> reproducible

        samples = draw_smoothed_samples(model, ct_t, masks_t, sigma, n_samples, batch_draws, rng)
        V = samples.shape[1]

        rec = {"patient": batch.patient_list[0], "radii": {}}
        for R in radii:
            cert = certify_from_samples(samples, radius=R, sigma=sigma, p=0.5, alpha=alpha)
            width_gy = (cert.upper - cert.lower) * presc          # per-voxel interval width, Gy
            body = pdm                                            # score inside the possible-dose region
            width_body = width_gy[body]
            frac_within = float(np.mean(width_body <= tol_gy)) if body.any() else float("nan")
            frac_certified = float(np.mean(cert.certified_low and cert.certified_high))  # scalar flags per (n,R)

            lower_gy = (cert.lower * presc)
            upper_gy = (cert.upper * presc)
            dvh_intervals = certified_dvh_intervals(evaluator, batch, lower_gy, upper_gy)
            dvh_widths = {k: v[1] - v[0] for k, v in dvh_intervals.items()}

            rec["radii"][f"{R:g}"] = {
                "radius_l2": R,
                "per_voxel_rms": per_voxel_rms_equivalent(R, V),
                "per_voxel_rms_hu": per_voxel_rms_equivalent(R, V) * 4095.0,
                "median_width_gy": float(np.median(width_body)),
                "p95_width_gy": float(np.percentile(width_body, 95)),
                "max_width_gy": float(np.max(width_body)),
                "frac_within_tol": frac_within,
                "certified_low": bool(cert.certified_low),
                "certified_high": bool(cert.certified_high),
                "j": cert.j, "k": cert.k, "p_lo": cert.p_lo, "p_hi": cert.p_hi,
                "dvh_interval_widths_gy": dvh_widths,
            }
        per_patient.append(rec)
        first_R = f"{radii[0]:g}"
        print(f"  patient {n_done}: {rec['patient']}  "
              f"R={first_R} median±/max width = "
              f"{rec['radii'][first_R]['median_width_gy']:.3f}/"
              f"{rec['radii'][first_R]['max_width_gy']:.3f} Gy  "
              f"frac<= {tol_gy}Gy = {rec['radii'][first_R]['frac_within_tol']:.3f}")
    return per_patient, n_done


def aggregate(per_patient, radii, tol_gy):
    """Mean over patients of the headline numbers, per radius."""
    agg = {}
    for R in radii:
        key = f"{R:g}"
        rows = [p["radii"][key] for p in per_patient if key in p["radii"]]
        if not rows:
            continue
        agg[key] = {
            "radius_l2": R,
            "per_voxel_rms_hu": rows[0]["per_voxel_rms_hu"],
            "mean_median_width_gy": float(np.mean([r["median_width_gy"] for r in rows])),
            "mean_p95_width_gy": float(np.mean([r["p95_width_gy"] for r in rows])),
            "mean_max_width_gy": float(np.mean([r["max_width_gy"] for r in rows])),
            "mean_frac_within_tol": float(np.mean([r["frac_within_tol"] for r in rows])),
            "all_certified": all(r["certified_low"] and r["certified_high"] for r in rows),
        }
    return agg


def print_table(agg, tol_gy):
    print(f"\n{'R(L2)':>7}{'~HU/vox':>9}  {'medW':>8}{'p95W':>8}{'maxW':>8}  {'frac<=%gGy' % tol_gy:>12}")
    print("-" * 60)
    for key, a in agg.items():
        print(f"{a['radius_l2']:>7.3g}{a['per_voxel_rms_hu']:>9.2f}  "
              f"{a['mean_median_width_gy']:>8.3f}{a['mean_p95_width_gy']:>8.3f}"
              f"{a['mean_max_width_gy']:>8.3f}  {a['mean_frac_within_tol']:>12.3f}")


def main():
    p = argparse.ArgumentParser(description="Certified robustness via median randomised smoothing")
    p.add_argument("--model", required=True, help="Path to trained .keras model")
    p.add_argument("--sigma", type=float, default=0.05,
                   help="Smoothing noise std in normalised CT units (x4095 ~ HU). Default 0.05 ~ 205 HU.")
    p.add_argument("--n-samples", type=int, default=100, help="Monte-Carlo noisy-CT draws per patient")
    p.add_argument("--batch-draws", type=int, default=8, help="Noisy CTs per forward pass")
    p.add_argument("--radii", default="0.5,1.0,2.0", help="Comma-separated L2 certified radii")
    p.add_argument("--tol-gy", type=float, default=1.0, help="Clinical tolerance for 'certified within' fraction")
    p.add_argument("--alpha", type=float, default=0.001, help="1 - confidence of the certificate")
    p.add_argument("--n-patients", type=int, default=None, help="Subset for a fast first pass")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--output", default="certify_results")
    args = p.parse_args()

    radii = [float(r) for r in args.radii.split(",")]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        script_dir = Path(__file__).parent
        data_dir = script_dir / "provided-data" / "validation-pats"
        if not data_dir.exists():
            data_dir = script_dir.parent / "provided-data" / "validation-pats"
    if not data_dir.exists():
        print(f"ERROR: validation data not found at {data_dir}")
        return 1

    print(f"Loading model: {args.model}")
    model = load_model(args.model,
                       custom_objects={"InstanceNormalization": InstanceNormalization},
                       compile=False, safe_mode=False)

    paths = sorted(get_paths(data_dir))
    if args.n_patients:
        paths = paths[:args.n_patients]
    print(f"Validation patients: {len(paths)}  sigma={args.sigma}  n_samples={args.n_samples}  "
          f"radii={radii}  alpha={args.alpha}")

    loader = DataLoader(paths, batch_size=1, normalize=True, cache_data=True)
    loader.set_mode("training_model")

    per_patient, n_done = run(model, loader, args.sigma, args.n_samples, radii,
                              args.tol_gy, args.alpha, args.batch_draws, args.seed)
    agg = aggregate(per_patient, radii, args.tol_gy)
    print_table(agg, args.tol_gy)

    summary = {
        "model": str(args.model),
        "timestamp": datetime.now().isoformat(),
        "method": "median randomised smoothing (Chiang et al. 2020) for voxel-wise dose regression",
        "sigma": args.sigma, "n_samples": args.n_samples, "alpha": args.alpha,
        "radii": radii, "tol_gy": args.tol_gy, "n_patients": n_done,
        "aggregate": agg,
        "per_patient": per_patient,
        "note": ("Certified per-voxel dose interval + certified DVH intervals under any "
                 "L2 CT perturbation of radius R, confidence 1-alpha. Base model NOT noise-"
                 "finetuned in this pass; SmoothAdv fine-tuning is the phase-2 lift."),
    }
    out_file = out_dir / "certify_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_file}")
    return 0


if __name__ == "__main__":
    exit(main())
