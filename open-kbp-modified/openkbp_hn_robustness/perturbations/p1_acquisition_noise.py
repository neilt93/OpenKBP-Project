"""P1: Signal-dependent acquisition noise.

Simulates CT scanner quantum noise with higher variance in bone (higher attenuation)
than in soft tissue. This models the heteroscedastic noise characteristic of
photon-counting and energy-integrating CT detectors.
"""
import numpy as np
from numpy.typing import NDArray

from .base import BasePerturbation

# OpenKBP stores CT as positive values with air=0 (roughly standard HU + 1024).
# Water ≈ 1024, bone starts at ≈ 1400+ (≈ 400 standard HU).
# We use 1500 to target cortical bone (≈ 476 standard HU).
BONE_THRESHOLD = 1500.0


class AcquisitionNoise(BasePerturbation):
    name = "P1_noise"
    levels = {
        "L1": {"sigma_soft": 8.0, "sigma_bone": 12.0},
        "L2": {"sigma_soft": 15.0, "sigma_bone": 25.0},
        "L3": {"sigma_soft": 30.0, "sigma_bone": 50.0},
        "L4": {"sigma_soft": 60.0, "sigma_bone": 100.0},
        "L5": {"sigma_soft": 100.0, "sigma_bone": 160.0},
    }

    def apply(self, ct_volume: NDArray, body_mask: NDArray, level: str,
              rng: np.random.Generator, **kwargs) -> NDArray:
        params = self.levels[level]
        sigma_soft = params["sigma_soft"]
        sigma_bone = params["sigma_bone"]

        # Build per-voxel sigma map: higher noise for bone
        bone_mask = ct_volume > BONE_THRESHOLD
        sigma_map = np.where(bone_mask, sigma_bone, sigma_soft)
        # Zero sigma outside body
        sigma_map = np.where(body_mask, sigma_map, 0.0)

        noise = rng.normal(0.0, 1.0, size=ct_volume.shape) * sigma_map
        perturbed = ct_volume + noise

        return self.clip_and_mask(perturbed, body_mask)
