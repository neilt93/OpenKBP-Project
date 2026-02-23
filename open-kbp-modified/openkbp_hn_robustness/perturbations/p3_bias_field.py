"""P3: Low-frequency bias field.

Simulates slowly varying intensity inhomogeneity caused by RF coil non-uniformity
or scatter artifacts. Implemented as a sum of low-order separable 3D cosine waves.
"""
import numpy as np
from numpy.typing import NDArray

from .base import BasePerturbation


class BiasField(BasePerturbation):
    name = "P3_bias_field"
    levels = {
        "L1": {"amplitude": 10.0, "n_harmonics": 3},
        "L2": {"amplitude": 20.0, "n_harmonics": 3},
    }

    def apply(self, ct_volume: NDArray, body_mask: NDArray, level: str,
              rng: np.random.Generator, **kwargs) -> NDArray:
        params = self.levels[level]
        amplitude = params["amplitude"]
        n_harmonics = params["n_harmonics"]

        shape = ct_volume.shape
        # Coordinate grids normalized to [0, 1]
        coords = [np.linspace(0, 1, s) for s in shape]

        # Build bias field as sum of separable 3D cosine harmonics
        field = np.zeros(shape, dtype=np.float64)
        for _ in range(n_harmonics):
            # Random frequency between 0.5 and 2.5 cycles per FOV
            freq = rng.uniform(0.5, 2.5, size=3)
            # Random phase
            phase = rng.uniform(0, 2 * np.pi, size=3)
            # Random amplitude weight
            weight = rng.uniform(0.3, 1.0)

            # Separable 3D cosine: product of 1D cosines along each axis
            cos_x = np.cos(2 * np.pi * freq[0] * coords[0] + phase[0])
            cos_y = np.cos(2 * np.pi * freq[1] * coords[1] + phase[1])
            cos_z = np.cos(2 * np.pi * freq[2] * coords[2] + phase[2])

            harmonic = weight * cos_x[:, None, None] * cos_y[None, :, None] * cos_z[None, None, :]
            field += harmonic

        # Normalize field to [-1, 1] then scale by amplitude
        field_max = np.abs(field).max()
        if field_max > 0:
            field = field / field_max
        field = field * amplitude

        # Apply only within body
        field = np.where(body_mask, field, 0.0)
        perturbed = ct_volume + field

        return self.clip_and_mask(perturbed, body_mask)
