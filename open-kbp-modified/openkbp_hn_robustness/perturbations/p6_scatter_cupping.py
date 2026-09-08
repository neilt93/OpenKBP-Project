"""P6: CBCT scatter / cupping artifact.

Cone-beam CT scatter produces a low-frequency "cupping" bias: HU are depressed toward the centre
of the field of view relative to the periphery. Modelled as a smooth radial bowl subtracted from
the CT within the body, constant across slices (in-plane radial dependence). This is a
CBCT-characteristic degradation for the online-adaptive-RT (Ethos) commissioning-tolerances study;
image-quality metrics famously fail to predict its dosimetric impact, which is exactly what the
1 Gy DVH criterion measures.
"""
import numpy as np
from numpy.typing import NDArray

from .base import BasePerturbation


class ScatterCupping(BasePerturbation):
    name = "P6_scatter_cupping"
    levels = {
        "L1": {"amplitude_hu": 20.0},
        "L2": {"amplitude_hu": 40.0},
        "L3": {"amplitude_hu": 80.0},
        "L4": {"amplitude_hu": 120.0},
        "L5": {"amplitude_hu": 200.0},
    }

    def apply(self, ct_volume: NDArray, body_mask: NDArray, level: str,
              rng: np.random.Generator, **kwargs) -> NDArray:
        amp = self.levels[level]["amplitude_hu"]
        _, ny, nx = ct_volume.shape
        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
        yy, xx = np.meshgrid(np.arange(ny) - cy, np.arange(nx) - cx, indexing="ij")
        r = np.sqrt(yy ** 2 + xx ** 2)
        r_max = np.sqrt(cy ** 2 + cx ** 2)
        bowl = 1.0 - (r / r_max) ** 2                 # 1 at centre, 0 at corner
        bias = (-amp * bowl)[None, :, :]              # depress centre, broadcast over slices
        return self.clip_and_mask(ct_volume + bias, body_mask)
