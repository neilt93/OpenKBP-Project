"""P2: Bone-weighted HU calibration shift.

Simulates scanner calibration drift that affects bone and soft tissue differently.
Uses a sigmoid blend for smooth transition at the bone/soft-tissue boundary,
avoiding sharp artifacts at the threshold.
"""
import numpy as np
from numpy.typing import NDArray

from .base import BasePerturbation

# OpenKBP stores CT as positive values with air=0 (roughly standard HU + 1024).
# Water ≈ 1024, cortical bone ≈ 1500+ (≈ 476 standard HU).
# Sigmoid center at 1500 with width 200 gives smooth bone/soft-tissue transition.
TRANSITION_CENTER = 1500.0
TRANSITION_WIDTH = 200.0


class BoneWeightedShift(BasePerturbation):
    name = "P2_bone_shift"
    levels = {
        "L1": {"shift_soft": 5.0, "shift_bone": 50.0},
        "L2": {"shift_soft": 10.0, "shift_bone": 100.0},
    }

    def apply(self, ct_volume: NDArray, body_mask: NDArray, level: str,
              rng: np.random.Generator, **kwargs) -> NDArray:
        params = self.levels[level]
        shift_soft = params["shift_soft"]
        shift_bone = params["shift_bone"]

        # Random sign: simulates direction of calibration error
        sign = rng.choice([-1.0, 1.0])

        # Sigmoid blend: 0 at soft tissue, 1 at bone
        bone_fraction = 1.0 / (1.0 + np.exp(-(ct_volume - TRANSITION_CENTER) / TRANSITION_WIDTH))

        # Interpolate shift: soft tissue gets small shift, bone gets large shift
        shift = sign * (shift_soft + (shift_bone - shift_soft) * bone_fraction)

        # Only apply within body
        shift = np.where(body_mask, shift, 0.0)
        perturbed = ct_volume + shift

        return self.clip_and_mask(perturbed, body_mask)
