"""P7: CBCT ring artifact.

Detector-element gain miscalibration in cone-beam CT produces concentric ring artifacts on axial
slices: a roughly sinusoidal HU modulation as a function of in-plane radius from the rotation
centre, repeating across all slices. Modelled as amp * sin(2*pi * n_rings * r/r_max + phase) added
within the body. CBCT-characteristic degradation for the commissioning-tolerances study.
"""
import numpy as np
from numpy.typing import NDArray

from .base import BasePerturbation


class RingArtifact(BasePerturbation):
    name = "P7_ring"
    levels = {
        "L1": {"amplitude_hu": 20.0, "n_rings": 12},
        "L2": {"amplitude_hu": 40.0, "n_rings": 12},
        "L3": {"amplitude_hu": 80.0, "n_rings": 12},
        "L4": {"amplitude_hu": 120.0, "n_rings": 12},
        "L5": {"amplitude_hu": 200.0, "n_rings": 12},
    }

    def apply(self, ct_volume: NDArray, body_mask: NDArray, level: str,
              rng: np.random.Generator, **kwargs) -> NDArray:
        amp = self.levels[level]["amplitude_hu"]
        n_rings = self.levels[level]["n_rings"]
        _, ny, nx = ct_volume.shape
        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
        yy, xx = np.meshgrid(np.arange(ny) - cy, np.arange(nx) - cx, indexing="ij")
        r = np.sqrt(yy ** 2 + xx ** 2)
        r_max = np.sqrt(cy ** 2 + cx ** 2)
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        rings = (amp * np.sin(2.0 * np.pi * n_rings * (r / r_max) + phase))[None, :, :]
        return self.clip_and_mask(ct_volume + rings, body_mask)
