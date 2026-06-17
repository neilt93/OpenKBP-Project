"""Bridge OpenKBP patient data into a matRad-ingestible .mat file.

Geometry safety: we hand matRad the full 3D cube and 3D structure masks, NOT raw
linear voxel indices. OpenKBP linear indices are C-order (row-major) and 0-based;
MATLAB is F-order (column-major) and 1-based. By passing whole cubes (scipy.io
preserves logical indexing across the order difference) and letting matRad compute
find(mask) itself, we never cross-contaminate the two index conventions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.io import savemat

from openkbp_hn_proton import config as C


# --- OpenKBP sparse-CSV loaders (self-contained, match provided_code format) --

def load_ct_cube(patient_dir: Path) -> Tuple[NDArray, NDArray]:
    """Load ct.csv -> (stored_cube float64 (128^3), scanned_mask bool).

    scanned_mask marks voxels OpenKBP actually stored (value > 0); everything else
    is outside the field of view and will become air.
    """
    df = pd.read_csv(patient_dir / "ct.csv", index_col=0)
    flat = np.zeros(int(np.prod(C.VOLUME_SHAPE)), dtype=np.float64)
    flat[df.index.values] = df["data"].values
    cube = flat.reshape(C.VOLUME_SHAPE)  # C-order, matching OpenKBP
    return cube, cube > 0


def load_structure_mask(patient_dir: Path, name: str) -> NDArray:
    """Load a structure mask CSV -> (128^3) bool array. Missing file -> all False."""
    path = patient_dir / f"{name}.csv"
    if not path.exists():
        return np.zeros(C.VOLUME_SHAPE, dtype=bool)
    df = pd.read_csv(path, index_col=0)  # mask CSVs have NaN data; indices are voxels
    indices = np.array(df.index).squeeze()
    flat = np.zeros(int(np.prod(C.VOLUME_SHAPE)), dtype=bool)
    flat[indices] = True
    return flat.reshape(C.VOLUME_SHAPE)


def load_voxel_dimensions(patient_dir: Path) -> NDArray:
    """Load voxel_dimensions.csv -> (3,) mm spacing [x, y, z]."""
    return np.loadtxt(patient_dir / "voxel_dimensions.csv")


# --- HU correction -----------------------------------------------------------

def correct_hu(stored_cube: NDArray, scanned_mask: NDArray) -> NDArray:
    """Convert OpenKBP rescaled HU to standard HU for matRad.

    trueHU = clip(stored - HU_OFFSET, HU_MIN, HU_MAX) inside the scanned region,
    AIR_HU outside it.
    """
    true_hu = np.full(stored_cube.shape, C.AIR_HU, dtype=np.float64)
    shifted = np.clip(stored_cube[scanned_mask] - C.HU_OFFSET, C.HU_MIN, C.HU_MAX)
    true_hu[scanned_mask] = shifted
    return true_hu


def qc_hu(true_hu: NDArray, masks: Dict[str, NDArray]) -> dict:
    """Sanity-check the HU correction. Soft tissue should sit near 0, air near -1000.

    Returns a dict of landmarks and prints a short report. Use this on pt_201 the
    first time the SanDisk warehouse is mounted to confirm HU_OFFSET is right.
    """
    body = true_hu > -200  # rough soft-tissue+ selection
    report = {
        "hu_min": float(true_hu.min()),
        "hu_max": float(true_hu.max()),
        "soft_tissue_median": float(np.median(true_hu[body])) if body.any() else None,
        "air_fraction_near_-1000": float(np.mean(np.abs(true_hu + 1000) < 50)),
    }
    # If a parotid (soft tissue) is contoured, its median is a clean landmark.
    for soft in ("LeftParotid", "RightParotid", "Brainstem"):
        m = masks.get(soft)
        if m is not None and m.any():
            report[f"{soft}_median_HU"] = float(np.median(true_hu[m]))
    print("HU QC:")
    for k, v in report.items():
        print(f"  {k:28s} {v}")
    print("  EXPECT: soft tissue / parotid median near 0 HU (+/- ~60); "
          "if it sits near +1000, HU_OFFSET is wrong.")
    return report


# --- matRad input assembly ---------------------------------------------------

def build_matrad_input(patient_dir: Path, run_qc: bool = True) -> dict:
    """Assemble the dict that gets written to a matRad input .mat for one patient."""
    patient_dir = Path(patient_dir)
    pid = patient_dir.stem

    stored, scanned = load_ct_cube(patient_dir)
    true_hu = correct_hu(stored, scanned)
    vox = load_voxel_dimensions(patient_dir)  # [x, y, z] mm; z is the slice axis

    structures = C.TARGETS + C.OARS
    masks = {n: load_structure_mask(patient_dir, n) for n in structures}
    present = {n: m for n, m in masks.items() if m.any()}

    if run_qc:
        qc_hu(true_hu, present)
        missing = [n for n in structures if n not in present]
        if missing:
            print(f"  NOTE: structures not contoured for {pid}: {missing}")

    mat = {
        "patientID": pid,
        "cubeHU": true_hu,                                   # 3D, standard HU
        "resolution": np.asarray(vox, dtype=np.float64),     # [x, y, z] mm
        "cubeDim": np.asarray(C.VOLUME_SHAPE, dtype=np.float64),
        # masks as a struct: data.masks.<Name> -> 3D uint8
        "masks": {n: m.astype(np.uint8) for n, m in present.items()},
        "structureType": {n: C.structure_type(n) for n in present},
        "prescription": {n: C.PRESCRIPTIONS[n] for n in present if n in C.PRESCRIPTIONS},
        "oarMaxDose": {n: C.OAR_MAX_DOSE[n] for n in present if n in C.OAR_MAX_DOSE},
        # fixed planning protocol
        "gantryAngles": np.asarray(C.GANTRY_ANGLES, dtype=np.float64),
        "couchAngles": np.asarray(C.COUCH_ANGLES, dtype=np.float64),
        "bixelWidth": float(C.BIXEL_WIDTH),
        "targetPenalty": float(C.TARGET_PENALTY),
        "oarPenalty": float(C.OAR_PENALTY),
        "radiationMode": C.RADIATION_MODE,
        "machine": C.MACHINE,
        "rbe": float(C.RBE),
    }
    return mat


def write_matrad_input(patient_dir: Path, out_path: Path, run_qc: bool = True) -> Path:
    """Build and save the matRad input .mat for one patient. Returns out_path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mat = build_matrad_input(patient_dir, run_qc=run_qc)
    # -v7 so MATLAB and Octave both read it; long_field_names for safety.
    savemat(str(out_path), mat, format="5", long_field_names=True, do_compression=True)
    return out_path
