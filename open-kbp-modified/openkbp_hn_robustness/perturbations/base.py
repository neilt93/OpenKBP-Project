"""Base I/O utilities and abstract perturbation class for CT robustness pipeline."""
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Volume shape for OpenKBP dataset
VOLUME_SHAPE = (128, 128, 128)
# Minimum clip value: 1 HU (not 0) to preserve sparse representation
# sparse_vector_function stores x[x > 0], so values must be > 0 to survive
HU_CLIP_MIN = 1.0
HU_CLIP_MAX = 4095.0


def load_ct_volume(patient_dir: Path) -> Tuple[NDArray, NDArray]:
    """Load ct.csv from a patient directory and reconstruct the 128x128x128 volume.

    Args:
        patient_dir: Path to patient directory containing ct.csv

    Returns:
        volume: (128,128,128) float64 array of HU values
        body_mask: (128,128,128) bool array where CT > 0 (body voxels)
    """
    ct_path = patient_dir / "ct.csv"
    df = pd.read_csv(ct_path, index_col=0)
    indices = df.index.values
    data = df["data"].values

    volume = np.zeros(np.prod(VOLUME_SHAPE), dtype=np.float64)
    volume[indices] = data
    volume = volume.reshape(VOLUME_SHAPE)

    body_mask = volume > 0
    return volume, body_mask


def load_structure_mask(patient_dir: Path, structure_name: str) -> NDArray:
    """Load a structure mask CSV and reconstruct the 128x128x128 binary mask.

    Args:
        patient_dir: Path to patient directory
        structure_name: Name of the structure (e.g., 'Mandible')

    Returns:
        mask: (128,128,128) bool array
    """
    mask_path = patient_dir / f"{structure_name}.csv"
    if not mask_path.exists():
        return np.zeros(VOLUME_SHAPE, dtype=bool)

    df = pd.read_csv(mask_path, index_col=0)
    # Structure masks have NaN in the data column; indices are the voxel locations
    indices = np.array(df.index).squeeze()

    mask = np.zeros(np.prod(VOLUME_SHAPE), dtype=bool)
    mask[indices] = True
    return mask.reshape(VOLUME_SHAPE)


def save_ct_volume(volume: NDArray, output_path: Path) -> None:
    """Save a perturbed CT volume to sparse CSV matching OpenKBP format.

    Only voxels with value > 0 are saved (sparse representation).

    Args:
        volume: (128,128,128) array of HU values (already clipped)
        output_path: Path to save the ct.csv file
    """
    flat = volume.flatten()
    mask = flat > 0
    indices = np.where(mask)[0]
    data = flat[mask]

    df = pd.DataFrame(data=data, index=indices, columns=["data"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)


def create_perturbed_patient(src_dir: Path, dst_dir: Path, perturbed_volume: NDArray) -> None:
    """Save perturbed ct.csv and symlink all other files from source patient.

    Args:
        src_dir: Original patient directory
        dst_dir: Destination directory for perturbed patient
        perturbed_volume: (128,128,128) perturbed CT array (already clipped)
    """
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Save perturbed CT
    save_ct_volume(perturbed_volume, dst_dir / "ct.csv")

    # Symlink all other files
    for src_file in src_dir.iterdir():
        if src_file.name == "ct.csv":
            continue
        dst_file = dst_dir / src_file.name
        if not dst_file.exists():
            os.symlink(src_file.resolve(), dst_file)


class BasePerturbation(ABC):
    """Abstract base class for CT perturbations.

    Subclasses must define:
        name: Short identifier (e.g., 'P1_noise')
        levels: Dict mapping level names to parameter dicts
        apply(): Method that perturbs a CT volume
    """
    name: str
    levels: Dict[str, dict]

    @abstractmethod
    def apply(self, ct_volume: NDArray, body_mask: NDArray, level: str,
              rng: np.random.Generator, **kwargs) -> NDArray:
        """Apply perturbation to a CT volume.

        Args:
            ct_volume: (128,128,128) float64 array in HU
            body_mask: (128,128,128) bool array of body voxels
            level: Level name (e.g., 'L1', 'L2')
            rng: NumPy random Generator for reproducibility
            **kwargs: Additional data (e.g., mandible_mask for P5)

        Returns:
            Perturbed volume clipped to [HU_CLIP_MIN, HU_CLIP_MAX] within body_mask,
            0 outside body_mask.
        """
        ...

    def clip_and_mask(self, volume: NDArray, body_mask: NDArray) -> NDArray:
        """Clip values within body mask and zero out air voxels."""
        result = np.where(body_mask, np.clip(volume, HU_CLIP_MIN, HU_CLIP_MAX), 0.0)
        return result
