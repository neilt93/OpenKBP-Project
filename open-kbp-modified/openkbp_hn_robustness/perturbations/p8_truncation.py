"""P8: CBCT limited-FOV truncation.

Cone-beam CT has a smaller reconstructed field of view than planning CT, so lateral anatomy
(shoulders in H&N) can fall outside the FOV and be truncated, with a characteristic bright rim at
the FOV boundary. Modelled by (a) adding a bright rim just inside the FOV edge and (b) zeroing
body voxels outside the FOV radius (missing anatomy). Severity increases as the FOV shrinks.
CBCT-characteristic degradation for the commissioning-tolerances study.
"""
import numpy as np
from numpy.typing import NDArray

from .base import BasePerturbation


class Truncation(BasePerturbation):
    name = "P8_truncation"
    # fov_frac = FOV radius as a fraction of the in-plane half-width (smaller = more truncation).
    levels = {
        "L1": {"fov_frac": 0.95, "rim_hu": 40.0},
        "L2": {"fov_frac": 0.90, "rim_hu": 80.0},
        "L3": {"fov_frac": 0.85, "rim_hu": 120.0},
        "L4": {"fov_frac": 0.80, "rim_hu": 160.0},
        "L5": {"fov_frac": 0.75, "rim_hu": 200.0},
    }

    def apply(self, ct_volume: NDArray, body_mask: NDArray, level: str,
              rng: np.random.Generator, **kwargs) -> NDArray:
        fov_frac = self.levels[level]["fov_frac"]
        rim_hu = self.levels[level]["rim_hu"]
        _, ny, nx = ct_volume.shape
        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
        yy, xx = np.meshgrid(np.arange(ny) - cy, np.arange(nx) - cx, indexing="ij")
        r = np.sqrt(yy ** 2 + xx ** 2)
        half_width = min(cy, cx)
        fov_radius = fov_frac * half_width

        perturbed = ct_volume.copy()
        # bright rim in the 3-voxel band just inside the FOV boundary
        rim = (r >= fov_radius - 3.0) & (r <= fov_radius)
        perturbed[:, rim] += rim_hu

        # zero body voxels outside the FOV (truncated / missing anatomy). Pass an effective mask
        # (body AND inside FOV) to clip_and_mask so those voxels become air (0), not HU_CLIP_MIN.
        fov_mask = (r <= fov_radius)[None, :, :]
        effective_mask = body_mask & fov_mask
        return self.clip_and_mask(perturbed, effective_mask)
