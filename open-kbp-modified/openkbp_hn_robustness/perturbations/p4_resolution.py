"""P4: Resolution degradation via anisotropic Gaussian blur.

Simulates reduced spatial resolution from thicker CT slices or reconstruction
kernel differences. More blur is applied along the Z (slice) axis to model
increased slice thickness, with mild in-plane blur.
"""
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from .base import BasePerturbation


class ResolutionDegradation(BasePerturbation):
    name = "P4_resolution"
    levels = {
        "L0": {"sigma_z": 0.5, "sigma_xy": 0.25},
        "L1": {"sigma_z": 1.0, "sigma_xy": 0.5},
        "L2": {"sigma_z": 2.0, "sigma_xy": 1.0},
        "L3": {"sigma_z": 3.0, "sigma_xy": 1.5},
        "L4": {"sigma_z": 4.0, "sigma_xy": 2.0},
    }

    def apply(self, ct_volume: NDArray, body_mask: NDArray, level: str,
              rng: np.random.Generator, **kwargs) -> NDArray:
        params = self.levels[level]
        sigma = (params["sigma_z"], params["sigma_xy"], params["sigma_xy"])

        # Apply Gaussian blur to the full volume
        perturbed = gaussian_filter(ct_volume, sigma=sigma, mode='constant', cval=0.0)

        # Restore body mask boundary (blur may spread values into air)
        return self.clip_and_mask(perturbed, body_mask)
