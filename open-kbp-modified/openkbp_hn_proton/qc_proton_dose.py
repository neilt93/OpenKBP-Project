#!/usr/bin/env python3
"""Phase-1 GATE: validate generated proton dose on one patient before scaling.

Two independent checks, because a plan can score well yet be geometrically wrong:

  1. SPATIAL (no TensorFlow): overlay CT + PTV70 + proton dose on three orthogonal
     planes. Confirm the high-dose region sits ON the target and the falloff is
     anatomically sane. This is what catches a slice-axis / beam-geometry mismatch
     that DVH numbers alone hide (D95 is computed against the same mask matRad used).

  2. DVH drop-in compat (needs TensorFlow / provided_code): load the proton dose
     through the REAL DataLoader(evaluation) + DoseEvaluator and print PTV/OAR DVH
     metrics. Proves the exported CSV is drop-in for the training/scoring stack.

Run from open-kbp-modified/ (warehouse must be mounted):
    python openkbp_hn_proton/qc_proton_dose.py \
        --patient provided-data/validation-pats/pt_201 \
        --dose    openkbp_hn_proton/proton_dose/pt_201/dose.csv \
        --out-png openkbp_hn_proton/qc/pt_201_overlay.png

Before proton dose exists you can still validate the spatial machinery on the
ORIGINAL photon dose:  --dose provided-data/validation-pats/pt_201/dose.csv
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from openkbp_hn_proton import config as C          # noqa: E402
from openkbp_hn_proton import ct_to_matrad as bridge  # noqa: E402
from openkbp_hn_proton import dose_io               # noqa: E402


def _self_contained_dvh(patient_dir: Path, dose: np.ndarray) -> None:
    """Quick DVH landmarks without provided_code/TF (sanity, not the official metric)."""
    print("  --- DVH (self-contained, Gy) ---")
    for t in C.TARGETS:
        m = bridge.load_structure_mask(patient_dir, t)
        if m.any():
            d = dose[m]
            print(f"  {t:10s} D95={np.percentile(d,5):5.1f}  Dmean={d.mean():5.1f}  "
                  f"(presc {C.PRESCRIPTIONS[t]})")
    for o in C.OARS:
        m = bridge.load_structure_mask(patient_dir, o)
        if m.any():
            d = dose[m]
            print(f"  {o:12s} Dmean={d.mean():5.1f}  Dmax={d.max():5.1f}")


def spatial_overlay(patient_dir: Path, dose: np.ndarray, out_png: Path) -> None:
    """Render CT + PTV70 contour + dose colorwash on 3 orthogonal planes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stored, scanned = bridge.load_ct_cube(patient_dir)
    ct = bridge.correct_hu(stored, scanned)
    ptv = bridge.load_structure_mask(patient_dir, "PTV70")
    if not ptv.any():
        ptv = bridge.load_structure_mask(patient_dir, "PTV63")
    ref = ptv if ptv.any() else (dose > 0)
    cz, cy, cx = np.argwhere(ref).mean(axis=0).astype(int)  # centroid (axis0,1,2)

    planes = [("axis0", cz), ("axis1", cy), ("axis2", cx)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    dmax = max(dose.max(), 1e-6)
    for ax, (label, idx) in zip(axes, planes):
        if label == "axis0":
            ct_s, dose_s, ptv_s = ct[idx], dose[idx], ptv[idx]
        elif label == "axis1":
            ct_s, dose_s, ptv_s = ct[:, idx], dose[:, idx], ptv[:, idx]
        else:
            ct_s, dose_s, ptv_s = ct[:, :, idx], dose[:, :, idx], ptv[:, :, idx]
        ax.imshow(ct_s, cmap="gray", vmin=-1000, vmax=1000)
        masked = np.ma.masked_where(dose_s <= 0.05 * dmax, dose_s)
        ax.imshow(masked, cmap="jet", alpha=0.45, vmin=0, vmax=dmax)
        if ptv_s.any():
            ax.contour(ptv_s, levels=[0.5], colors="cyan", linewidths=1.2)
        ax.set_title(f"{label} = {idx}")
        ax.axis("off")
    fig.suptitle(f"{patient_dir.stem}: CT (gray) + dose (jet) + PTV (cyan).  "
                 f"High dose should sit ON the PTV.", fontsize=11)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"  spatial overlay -> {out_png}")
    print("  INSPECT: which axis is the true through-slice axis? Does dose hug the PTV?")


def dvh_via_provided_code(patient_dir: Path, dose_csv: Path) -> None:
    """Drop-in compat proof through the REAL evaluation stack (needs TensorFlow)."""
    try:
        from provided_code.data_loader import DataLoader
        from provided_code.dose_evaluation_class import DoseEvaluator
    except Exception as e:  # noqa: BLE001
        print(f"  [skip] provided_code/TF not importable here ({type(e).__name__}); "
              f"run this part on the cluster. Spatial + self-contained DVH still ran.")
        return

    # Build an evaluation dir: proton dose as dose.csv + symlinks to the real masks etc.
    with tempfile.TemporaryDirectory() as t:
        ev_dir = Path(t) / patient_dir.stem
        ev_dir.mkdir()
        for f in patient_dir.iterdir():
            if f.name == "dose.csv":
                continue
            (ev_dir / f.name).symlink_to(f.resolve())
        (ev_dir / "dose.csv").symlink_to(Path(dose_csv).resolve())

        loader = DataLoader([ev_dir], batch_size=1, normalize=False, cache_data=False)
        ev = DoseEvaluator(loader)  # reference-only: computes DVH metrics for this dose
        ev.evaluate()
        row = ev.reference_dvh_metrics_df.loc[patient_dir.stem]
        print("  --- DVH via provided_code DoseEvaluator (official metric machinery) ---")
        for (metric, roi), val in row.dropna().items():
            print(f"  {roi:12s} {metric:8s} = {val:5.1f}")
        print("  PASS if these load without error and PTV70 D_95 ~>= 95% of 70.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-1 proton-dose QC gate")
    ap.add_argument("--patient", required=True, help="Original OpenKBP patient dir")
    ap.add_argument("--dose", required=True, help="Dose CSV to validate (proton or photon)")
    ap.add_argument("--out-png", default=None, help="Overlay PNG path")
    ap.add_argument("--no-eval", action="store_true", help="Skip the provided_code/TF DVH check")
    args = ap.parse_args()

    patient_dir = Path(args.patient)
    dose_csv = Path(args.dose)
    if not (patient_dir / "ct.csv").exists():
        ap.error(f"{patient_dir}/ct.csv missing — is the SanDisk warehouse mounted?")
    out_png = Path(args.out_png) if args.out_png else script_dir / "qc" / f"{patient_dir.stem}_overlay.png"

    dose = dose_io.load_sparse_cube(dose_csv)
    print(f"\n=== QC {patient_dir.stem}  (dose: {dose_csv}) ===")
    print(f"  dose: nonzero voxels={int((dose>0).sum())}  max={dose.max():.1f} Gy")
    _self_contained_dvh(patient_dir, dose)
    spatial_overlay(patient_dir, dose, out_png)
    if not args.no_eval:
        dvh_via_provided_code(patient_dir, dose_csv)


if __name__ == "__main__":
    main()
