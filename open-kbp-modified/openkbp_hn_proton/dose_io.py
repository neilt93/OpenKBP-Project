"""Dose <-> OpenKBP sparse-CSV conversion and round-trip / QC helpers.

The sparse CSV format (CT, dose, predictions) is: a header row ",data" then one
row per non-zero voxel as "<flat_index>,<value>", where flat_index is the C-order
(row-major) ravel of the 128^3 cube. Only voxels with value > 0 are stored.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.io import loadmat

from openkbp_hn_proton import config as C


def save_sparse_csv(cube: NDArray, out_path: Path) -> Path:
    """Save a 3D cube to OpenKBP sparse CSV (only value > 0). Returns out_path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flat = cube.ravel(order="C")
    mask = flat > 0
    df = pd.DataFrame(data=flat[mask], index=np.where(mask)[0], columns=["data"])
    df.index.name = None
    df.to_csv(out_path)
    return out_path


def load_sparse_cube(csv_path: Path, shape=C.VOLUME_SHAPE) -> NDArray:
    """Reconstruct a dense 3D cube from an OpenKBP sparse dose/CT CSV."""
    df = pd.read_csv(csv_path, index_col=0)
    flat = np.zeros(int(np.prod(shape)), dtype=np.float64)
    flat[df.index.values] = df["data"].values
    return flat.reshape(shape)


def matrad_dose_to_cube(mat_path: Path, field: Optional[str] = None) -> NDArray:
    """Read a matRad result .mat -> 3D dose cube (Gy(RBE)).

    Looks for, in order: an explicit `field`, then 'rbeDose', 'RBExDose',
    'physicalDose' (scaled by config.RBE), then 'dose'.
    """
    md = loadmat(str(mat_path))
    candidates = [field] if field else ["rbeDose", "RBExDose", "physicalDose", "dose"]
    for key in candidates:
        if key and key in md:
            cube = np.asarray(md[key], dtype=np.float64)
            if key == "physicalDose":  # constant-RBE fallback
                cube = cube * C.RBE
            if cube.shape != tuple(C.VOLUME_SHAPE):
                raise ValueError(f"Dose cube shape {cube.shape} != {C.VOLUME_SHAPE}")
            return cube
    raise KeyError(f"No dose field found in {mat_path}. Keys: {list(md.keys())}")


def import_matrad_dose(mat_path: Path, out_csv: Path, field: Optional[str] = None) -> Path:
    """matRad result .mat -> OpenKBP sparse dose CSV (ground-truth proton dose)."""
    cube = matrad_dose_to_cube(mat_path, field=field)
    return save_sparse_csv(cube, out_csv)


def roundtrip_ok(cube: NDArray, tmp_csv: Path, rtol: float = 0.0, atol: float = 1e-9) -> bool:
    """Save a cube to sparse CSV, reload, and confirm it is recovered exactly.

    Note: only voxels > 0 survive the sparse format, so we compare on max(cube, 0).
    """
    save_sparse_csv(cube, tmp_csv)
    recovered = load_sparse_cube(tmp_csv)
    expected = np.where(cube > 0, cube, 0.0)
    return np.allclose(recovered, expected, rtol=rtol, atol=atol)
