"""P5: Dental metal artifact simulation.

Simulates streak artifacts from dental fillings/implants. Generates radial
streak patterns anchored to the mandible centroid, with alternating bright/dark
streaks that decay with distance and Z-offset from the seed slice.
"""
import numpy as np
from numpy.typing import NDArray

from .base import BasePerturbation, load_structure_mask


class DentalArtifact(BasePerturbation):
    name = "P5_dental"
    levels = {
        "L1": {"amplitude": 150.0, "n_streaks": 8, "metal_spot_hu": 3000.0},
        "L2": {"amplitude": 300.0, "n_streaks": 12, "metal_spot_hu": 3500.0},
    }

    def apply(self, ct_volume: NDArray, body_mask: NDArray, level: str,
              rng: np.random.Generator, **kwargs) -> NDArray:
        params = self.levels[level]
        amplitude = params["amplitude"]
        n_streaks = params["n_streaks"]
        metal_spot_hu = params["metal_spot_hu"]

        # Get mandible mask (passed via kwargs or loaded from patient dir)
        mandible_mask = kwargs.get("mandible_mask")
        if mandible_mask is None:
            patient_dir = kwargs.get("patient_dir")
            if patient_dir is not None:
                mandible_mask = load_structure_mask(patient_dir, "Mandible")
            else:
                # No mandible info available, return unchanged
                return ct_volume.copy()

        if not mandible_mask.any():
            return ct_volume.copy()

        # Find mandible centroid
        coords = np.argwhere(mandible_mask)
        centroid = coords.mean(axis=0)  # (z, y, x)
        z_center = int(round(centroid[0]))

        shape = ct_volume.shape  # (128, 128, 128)

        # Build 2D streak pattern in the axial (y, x) plane
        y_grid, x_grid = np.mgrid[0:shape[1], 0:shape[2]]
        dy = y_grid - centroid[1]
        dx = x_grid - centroid[2]
        angles = np.arctan2(dy, dx)  # (-pi, pi)
        radial_dist = np.sqrt(dy**2 + dx**2) + 1e-6  # avoid div by zero

        # Generate streak angles with random offsets
        streak_angles = rng.uniform(0, 2 * np.pi, size=n_streaks)
        streak_width = np.pi / (n_streaks * 1.5)  # angular width of each streak

        # Build 2D streak pattern
        streak_2d = np.zeros((shape[1], shape[2]), dtype=np.float64)
        for i, angle in enumerate(streak_angles):
            # Angular proximity to this streak
            ang_diff = np.abs(np.arctan2(np.sin(angles - angle), np.cos(angles - angle)))
            # Gaussian angular profile
            profile = np.exp(-0.5 * (ang_diff / streak_width)**2)
            # Alternating sign for bright/dark streaks
            sign = 1.0 if i % 2 == 0 else -1.0
            streak_2d += sign * profile

        # Radial decay: streaks fade with distance from center
        radial_decay = np.exp(-radial_dist / 40.0)
        streak_2d *= radial_decay

        # Normalize 2D pattern to [-1, 1]
        s_max = np.abs(streak_2d).max()
        if s_max > 0:
            streak_2d /= s_max

        # Extend to 3D with Z-decay from seed slice
        artifact = np.zeros(shape, dtype=np.float64)
        z_coords = np.arange(shape[0])
        z_decay = np.exp(-0.5 * ((z_coords - z_center) / 3.0)**2)  # ~3 slice spread

        for z in range(shape[0]):
            artifact[z] = streak_2d * amplitude * z_decay[z]

        # Add metal saturation spot near mandible centroid (small sphere)
        z_indices, y_indices, x_indices = np.ogrid[0:shape[0], 0:shape[1], 0:shape[2]]
        metal_dist = np.sqrt(
            (z_indices - centroid[0])**2 +
            (y_indices - centroid[1])**2 +
            (x_indices - centroid[2])**2
        )
        metal_spot = np.where(metal_dist < 3.0, metal_spot_hu, 0.0)

        # Only apply artifacts within body mask
        artifact = np.where(body_mask, artifact, 0.0)
        perturbed = ct_volume + artifact
        # Add metal spot (overwrite, don't add)
        perturbed = np.where(metal_dist < 3.0, np.maximum(perturbed, metal_spot), perturbed)

        return self.clip_and_mask(perturbed, body_mask)
