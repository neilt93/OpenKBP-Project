#!/usr/bin/env python3
"""Why is P4 (resolution) the failure mode? Frequency-domain sensitivity of the dose predictor.

Hypothesis: the model leans on HIGH-frequency edge content of the CT, so anisotropic blur (P4)
— which removes exactly that — degrades dose prediction, while intensity perturbations (P1/P2/P3/P5)
that preserve edges are nearly free. This script tests it mechanistically: band-stop narrow radial
spectral bands of the CT, run the model, and measure the dose response per band. A response curve
rising toward high frequencies explains P4, predicts which future perturbations will hurt, and
explains why perturbed-CT retraining helps.

RUNS ON THE POD / GPU BOX (needs TF + model + validation data). Reuses the verified
run_inference.py path. `--self-test` validates the band-decomposition math locally (no pod):
the radial bands partition the spectrum and the band-isolated components sum back to the original.

Outputs: freq_sensitivity.csv (band, freq range, mean |Δdose| Gy) + fig_freq_sensitivity.png.
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OKBP = os.path.normpath(os.path.join(HERE, "..", "..", "open-kbp-modified"))
for p in (OKBP, os.path.join(OKBP, "openkbp_hn_robustness")):
    if p not in sys.path:
        sys.path.insert(0, p)

DOSE_PRESCRIPTION = 70.0
HU_MIN, HU_MAX = 1.0, 4095.0


def radial_band_masks(shape, n_bands):
    """Partition the (fftshifted) 3D spectrum into n_bands equal radial-frequency shells.
    Returns a list of boolean masks over the shifted-FFT grid; they tile the whole spectrum."""
    zz, yy, xx = np.indices(shape)
    c = (np.array(shape) - 1) / 2.0
    r = np.sqrt(((zz - c[0]) / c[0]) ** 2 + ((yy - c[1]) / c[1]) ** 2 + ((xx - c[2]) / c[2]) ** 2)
    r = r / r.max()                                    # normalized radial frequency in [0,1]
    edges = np.linspace(0.0, 1.0 + 1e-9, n_bands + 1)
    return [(r >= edges[i]) & (r < edges[i + 1]) for i in range(n_bands)], edges


def bandstop_ct(ct, band_mask):
    """Remove one radial frequency band from the CT (real inverse)."""
    F = np.fft.fftshift(np.fft.fftn(ct))
    F[band_mask] = 0.0
    return np.real(np.fft.ifftn(np.fft.ifftshift(F)))


def band_isolate(ct, band_mask):
    """Keep only one radial frequency band."""
    F = np.fft.fftshift(np.fft.fftn(ct))
    F[~band_mask] = 0.0
    return np.real(np.fft.ifftn(np.fft.ifftshift(F)))


def self_test():
    rng = np.random.default_rng(0)
    ct = rng.normal(0, 1, size=(24, 24, 24))
    masks, edges = radial_band_masks(ct.shape, 6)
    # (1) masks partition the spectrum exactly (each cell in exactly one band)
    stacked = np.sum(masks, axis=0)
    assert stacked.min() == 1 and stacked.max() == 1, "bands do not partition the spectrum"
    # (2) band-isolated components sum back to the original (linearity of the DFT)
    recon = sum(band_isolate(ct, m) for m in masks)
    assert np.allclose(recon, ct, atol=1e-8), "band components do not reconstruct the CT"
    # (3) band-stop == original minus that band's isolate
    m = masks[3]
    assert np.allclose(bandstop_ct(ct, m), ct - band_isolate(ct, m), atol=1e-8)
    print(f"[self-test] PASS (6 bands partition spectrum; reconstruct to atol 1e-8; edges={np.round(edges,2)})")
    raise SystemExit(0)


def main():
    ap = argparse.ArgumentParser(description="Frequency-domain sensitivity of the dose predictor")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model", help="trained .keras model")
    ap.add_argument("--data-dir", help="validation-pats dir")
    ap.add_argument("--n-patients", type=int, default=10)
    ap.add_argument("--n-bands", type=int, default=8)
    ap.add_argument("--out", default=HERE)
    args = ap.parse_args()
    if args.self_test:
        self_test()
    if not (args.model and args.data_dir):
        raise SystemExit("need --model and --data-dir on the pod (or use --self-test locally)")

    import tensorflow as tf
    from provided_code.data_loader import DataLoader
    from provided_code.network_architectures import InstanceNormalization
    from perturbations.base import load_ct_volume
    from pathlib import Path

    model = tf.keras.models.load_model(
        args.model, custom_objects={"InstanceNormalization": InstanceNormalization},
        compile=False, safe_mode=False)

    def predict(patient_dir):
        loader = DataLoader([patient_dir], batch_size=1, normalize=True, cache_data=False)
        loader.set_mode("dose_prediction")
        b = next(iter(loader.get_batches()))
        dose = np.squeeze(model.predict([b.ct, b.structure_masks], verbose=0)
                          * b.possible_dose_mask) * DOSE_PRESCRIPTION
        return dose, np.squeeze(b.possible_dose_mask)

    import tempfile
    from perturbations.base import create_perturbed_patient
    data_dir = Path(args.data_dir)
    patients = sorted(data_dir.iterdir())[:args.n_patients]
    masks, edges = radial_band_masks((128, 128, 128), args.n_bands)
    band_resp = np.zeros(args.n_bands)

    with tempfile.TemporaryDirectory() as td:
        for pdir in patients:
            ct, body = load_ct_volume(pdir)
            base_dose, pdm = predict(pdir)
            for bi, m in enumerate(masks):
                bs = np.clip(bandstop_ct(ct, m), 0, HU_MAX)
                bs = np.where(body, np.clip(bs, HU_MIN, HU_MAX), 0.0)
                dst = Path(td) / f"{pdir.name}_b{bi}"
                create_perturbed_patient(pdir, dst, bs)
                dose, _ = predict(dst)
                band_resp[bi] += np.mean(np.abs(dose - base_dose)[pdm > 0])
    band_resp /= len(patients)

    rows = [{"band": i, "freq_lo": round(edges[i], 3), "freq_hi": round(edges[i + 1], 3),
             "mean_abs_dose_change_gy": round(float(band_resp[i]), 4)} for i in range(args.n_bands)]
    with open(os.path.join(args.out, "freq_sensitivity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(args.n_bands)]
    plt.figure(figsize=(7, 4.3))
    plt.bar(centers, band_resp, width=(1.0 / args.n_bands) * 0.9, color="#c0392b")
    plt.xlabel("Normalized radial spatial frequency (0=low, 1=Nyquist)")
    plt.ylabel("Dose response to band removal — mean |Δdose| (Gy)")
    plt.title("Which CT frequencies the dose model relies on")
    plt.grid(alpha=0.25, axis="y"); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "fig_freq_sensitivity.png"), dpi=200)
    print(f"Wrote freq_sensitivity.csv + fig_freq_sensitivity.png\nband responses (Gy): "
          f"{np.round(band_resp,3)}")
    print("Prediction: response rises toward high frequency -> explains P4 (blur removes it).")


if __name__ == "__main__":
    main()
