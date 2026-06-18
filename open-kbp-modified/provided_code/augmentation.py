"""Geometric + perturbation data augmentation for adversarial / robustness retraining.

Pure numpy/scipy (NO TensorFlow) so it is unit-testable off-GPU. The training loop
(`network_functions`) calls `augment_batch` on the numpy batch *before* the tensor
conversion; XLA is off in practice (`--no-jit`), so CPU augmentation is fine.

Volume layout is OpenKBP BDHWC: (batch, D=S-I, H=A-P, W=L-R, channels). Two kinds of
augmentation, deliberately distinct:

  * GEOMETRIC (flips, translation, in-plane rotation, scaling, elastic): applied to the
    CT *and* the structure masks, dose, and possible_dose_mask with the SAME transform
    per sample so they stay registered. CT/dose interpolate linearly; masks /
    possible_dose_mask use nearest-neighbour so they stay binary. Out-of-volume fills 0
    (air / no-dose / outside-ROI).

  * INTENSITY / CT-only perturbations (intensity scale, Gaussian noise): applied to the
    CT only — the dose/masks ground truth is unchanged. This is the on-the-fly analogue
    of injecting the pre-generated perturbed CT sets (robustness families P1-P5): the
    model must predict the correct dose despite a corrupted CT.

All geometric ops are fused into a SINGLE coordinate resample per field (one
interpolation, not one-per-op) to avoid compounding blur.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, map_coordinates

# scipy's map_coordinates/gaussian_filter are C extensions that release the GIL, so the
# per-sample augmentation parallelizes across cores with threads (no pickling/copy cost
# like processes). Sized to the box; cheap to keep one persistent pool.
_POOL = ThreadPoolExecutor(max_workers=min((os.cpu_count() or 4), 16))


def _sampling_coords(
    shape: tuple,
    rng: np.random.Generator,
    translate_frac: float,
    rotate_deg: float,
    scale_range: float,
    elastic_alpha: float,
    elastic_sigma: float,
) -> Optional[NDArray]:
    """Build a (3, D, H, W) array of input coordinates to sample for each output voxel.

    Combines in-plane (axial = H,W) rotation, in-plane scaling, 3D translation and a
    smooth elastic field into one map. Returns None if no geometric op is active (so the
    caller can skip resampling entirely).
    """
    D, H, W = shape
    if rotate_deg <= 0 and scale_range <= 0 and translate_frac <= 0 and elastic_alpha <= 0:
        return None

    grid = np.indices((D, H, W), dtype=np.float32)            # (3, D, H, W)
    center = np.array([(D - 1) / 2, (H - 1) / 2, (W - 1) / 2], dtype=np.float32).reshape(3, 1, 1, 1)
    coords = grid - center

    # In-plane rotation (axial plane = axes H,W = vector components 1,2) + in-plane scale.
    theta = np.deg2rad(rng.uniform(-rotate_deg, rotate_deg)) if rotate_deg > 0 else 0.0
    c, s = np.cos(theta), np.sin(theta)
    sc = 1.0 + rng.uniform(-scale_range, scale_range) if scale_range > 0 else 1.0
    # M maps output offset -> input offset. Scale in-plane only (S-I kept at 1.0; coarse
    # 3 mm slices make through-plane scaling less meaningful).
    M = np.array([
        [1.0, 0.0, 0.0],
        [0.0, sc * c, -sc * s],
        [0.0, sc * s, sc * c],
    ], dtype=np.float32)
    coords = (M @ coords.reshape(3, -1)).reshape(3, D, H, W) + center

    if translate_frac > 0:
        trans = np.array([
            rng.uniform(-translate_frac, translate_frac) * D,
            rng.uniform(-translate_frac, translate_frac) * H,
            rng.uniform(-translate_frac, translate_frac) * W,
        ], dtype=np.float32).reshape(3, 1, 1, 1)
        coords = coords + trans

    if elastic_alpha > 0:
        for ax in range(3):
            disp = gaussian_filter(
                rng.uniform(-1, 1, size=(D, H, W)).astype(np.float32), elastic_sigma
            )
            coords[ax] += disp * elastic_alpha

    return coords


def _resample(field: NDArray, coords: NDArray, order: int) -> NDArray:
    """Resample a (D, H, W, C) field at `coords` (3, D, H, W), per channel. cval=0."""
    out = np.empty_like(field)
    for ch in range(field.shape[-1]):
        out[..., ch] = map_coordinates(
            field[..., ch], coords, order=order, mode="constant", cval=0.0
        )
    return out


def augment_sample(
    ct: NDArray,
    structure_masks: NDArray,
    dose: NDArray,
    possible_dose_mask: NDArray,
    rng: np.random.Generator,
    *,
    flip_prob: float = 0.5,
    intensity_scale: float = 0.1,
    noise_std: float = 0.0,
    translate_frac: float = 0.0,
    rotate_deg: float = 0.0,
    scale_range: float = 0.0,
    elastic_alpha: float = 0.0,
    elastic_sigma: float = 4.0,
) -> tuple:
    """Augment ONE sample. Each field is (D, H, W, C). Returns new arrays (no mutation)."""
    ct = ct.astype(np.float32, copy=True)
    structure_masks = structure_masks.astype(np.float32, copy=True)
    dose = dose.astype(np.float32, copy=True)
    possible_dose_mask = possible_dose_mask.astype(np.float32, copy=True)

    # --- flips (cheap, exact) -------------------------------------------------
    # Sample axes are (D=0, H=1, W=2, C=3). Matches augment_batch_tf's BDHWC flips:
    # its axis 3 (W, L-R) -> sample axis 2; its axis 2 (H, A-P) -> sample axis 1.
    if rng.random() < flip_prob:  # L-R flip (W = sample axis 2)
        ct = ct[:, :, ::-1]; structure_masks = structure_masks[:, :, ::-1]
        dose = dose[:, :, ::-1]; possible_dose_mask = possible_dose_mask[:, :, ::-1]
    if rng.random() < flip_prob:  # A-P flip (H = sample axis 1)
        ct = ct[:, ::-1]; structure_masks = structure_masks[:, ::-1]
        dose = dose[:, ::-1]; possible_dose_mask = possible_dose_mask[:, ::-1]

    # --- fused geometric resample (translation/rotation/scale/elastic) --------
    coords = _sampling_coords(
        ct.shape[:3], rng, translate_frac, rotate_deg, scale_range, elastic_alpha, elastic_sigma
    )
    if coords is not None:
        ct = _resample(ct, coords, order=1)
        dose = _resample(dose, coords, order=1)
        structure_masks = _resample(structure_masks, coords, order=0)
        possible_dose_mask = _resample(possible_dose_mask, coords, order=0)

    # --- CT-only intensity perturbations (dose/masks unchanged) ---------------
    if intensity_scale > 0:
        ct = ct * (1.0 + rng.uniform(-intensity_scale, intensity_scale))
    if noise_std > 0:
        ct = ct + rng.normal(0.0, noise_std, size=ct.shape).astype(np.float32)
    if intensity_scale > 0 or noise_std > 0:
        ct = np.clip(ct, 0.0, 1.0)  # keep normalized CT in range

    return ct, structure_masks, dose, possible_dose_mask


def augment_batch(
    ct: NDArray,
    structure_masks: NDArray,
    dose: NDArray,
    possible_dose_mask: NDArray,
    rng: Optional[np.random.Generator] = None,
    **kwargs,
) -> tuple:
    """Augment a (B, D, H, W, C) batch sample-by-sample (independent transforms).

    Drop-in for the training loop: call on the numpy batch before tensor conversion.
    `kwargs` are the per-sample augmentation strengths (see `augment_sample`).

    Samples are augmented in parallel across CPU threads (the heavy scipy ops release the
    GIL). Each sample gets its OWN rng seeded from `rng` — numpy Generators are not
    thread-safe to share, and this also makes augmentation reproducible given a seeded rng.
    """
    rng = rng or np.random.default_rng()
    out_ct = np.empty_like(ct, dtype=np.float32)
    out_sm = np.empty_like(structure_masks, dtype=np.float32)
    out_dose = np.empty_like(dose, dtype=np.float32)
    out_pdm = np.empty_like(possible_dose_mask, dtype=np.float32)
    B = ct.shape[0]
    seeds = rng.integers(0, 2**31 - 1, size=B)  # per-sample, parent-derived (thread-safe)

    def _work(b):
        out_ct[b], out_sm[b], out_dose[b], out_pdm[b] = augment_sample(
            ct[b], structure_masks[b], dose[b], possible_dose_mask[b],
            np.random.default_rng(seeds[b]), **kwargs
        )

    list(_POOL.map(_work, range(B)))  # writes to disjoint b-slices -> no data race
    return out_ct, out_sm, out_dose, out_pdm
