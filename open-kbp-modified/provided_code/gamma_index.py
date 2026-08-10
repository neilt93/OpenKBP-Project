"""3D gamma-index passing rate — pure numpy (NO TensorFlow), unit-testable off-GPU.

Gamma analysis (Low et al. 1998) is the clinical standard for comparing two dose
distributions: it combines a Dose-Difference (DD) criterion with a Distance-to-Agreement
(DTA) criterion. For each reference voxel r,
    gamma(r) = min_e sqrt( (D(e)-D_ref(r))^2 / dd^2  +  dist(e,r)^2 / dta^2 )
over evaluation voxels e in a neighbourhood; a voxel PASSES if gamma <= 1. The gamma
passing rate (GPR) is the % of evaluated voxels that pass, over voxels above a low-dose
threshold. AAPM TG-218 uses 3%/2mm (95% tolerance / 90% action) and 3%/3mm is common.

This is used here as a robustness metric: GPR between the CLEAN prediction and the
perturbed/defended prediction answers "does the perturbed plan still agree with the clean
plan within 3%/3mm?" — more clinically legible than voxel MAE.

Implementation notes (honest):
  * GLOBAL gamma: DD is normalised by a single global dose (default = reference max), the
    common convention for plan comparison.
  * Integer-voxel DTA search (offsets on the grid, not interpolated). This slightly
    UNDER-credits DTA, so the reported GPR is conservative (a lower bound on the
    interpolated GPR). Documented rather than hidden.
  * COARSE-GRID CAVEAT: OpenKBP dose voxels are large (~3 mm S-I, ~5 mm in-plane), so a
    3 mm DTA is sub-voxel in-plane — gamma there reduces largely to the dose-difference
    test. Report GPR with the voxel spacing stated so this is transparent.

Certified vs empirical: this computes EMPIRICAL gamma (two concrete dose volumes). A
*certified* gamma bound is harder (gamma's DTA min is spatial / non-monotone, so the
monotone per-voxel dose-interval push-through used for certified DVH does not apply) and is
left as future work.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def gamma_passing_rate(
    reference: NDArray,
    evaluation: NDArray,
    spacing_mm,
    dd_percent: float = 3.0,
    dta_mm: float = 3.0,
    dose_threshold_frac: float = 0.10,
    search_factor: float = 1.5,
    global_norm_dose: float | None = None,
) -> float:
    """Global 3D gamma passing rate (fraction in [0,1]) of `evaluation` vs `reference`.

    reference/evaluation: same-shape 3D dose arrays (Gy). spacing_mm: per-axis voxel size
    (mm), length 3, aligned to the array axes. dd_percent/dta_mm: gamma criteria. Voxels
    below dose_threshold_frac * global_norm_dose are excluded from the denominator.
    search_factor: DTA search window as a multiple of dta (1.5 is a safe default).
    global_norm_dose: DD normalisation dose; default = reference.max().
    """
    ref = np.asarray(reference, dtype=np.float32)
    ev = np.asarray(evaluation, dtype=np.float32)
    if ref.shape != ev.shape or ref.ndim != 3:
        raise ValueError(f"reference/evaluation must be equal-shape 3D; got {ref.shape}, {ev.shape}")
    spacing = np.asarray(spacing_mm, dtype=np.float64)
    if spacing.shape != (3,):
        raise ValueError(f"spacing_mm must have length 3; got {spacing}")

    norm = float(global_norm_dose) if global_norm_dose is not None else float(ref.max())
    if norm <= 0:
        return float("nan")
    dd = (dd_percent / 100.0) * norm  # absolute DD criterion in Gy

    # Voxels evaluated: reference above the low-dose threshold.
    mask = ref >= dose_threshold_frac * norm
    if not mask.any():
        return float("nan")

    # DTA search window per axis (integer voxels).
    reach = [int(np.ceil(search_factor * dta_mm / s)) if s > 0 else 0 for s in spacing]

    gamma2_min = np.full(ref.shape, np.inf, dtype=np.float32)
    for dz in range(-reach[0], reach[0] + 1):
        for dy in range(-reach[1], reach[1] + 1):
            for dx in range(-reach[2], reach[2] + 1):
                dist2 = ((dz * spacing[0]) ** 2 + (dy * spacing[1]) ** 2 + (dx * spacing[2]) ** 2)
                if dist2 > (search_factor * dta_mm) ** 2:
                    continue  # outside the spherical search radius
                shifted = _shift_fill(ev, (dz, dy, dx), fill=np.inf)
                g2 = ((ref - shifted) ** 2) / (dd ** 2) + np.float32(dist2 / (dta_mm ** 2))
                np.minimum(gamma2_min, g2, out=gamma2_min)

    passed = (gamma2_min[mask] <= 1.0)
    return float(np.mean(passed))


def _shift_fill(a: NDArray, offset, fill=np.inf) -> NDArray:
    """Shift 3D array by integer (dz,dy,dx); positions with no source voxel set to `fill`
    (so out-of-range neighbours never produce a spurious low gamma)."""
    dz, dy, dx = offset
    out = np.full_like(a, fill)
    zs_src, zs_dst = _slices(dz, a.shape[0])
    ys_src, ys_dst = _slices(dy, a.shape[1])
    xs_src, xs_dst = _slices(dx, a.shape[2])
    out[zs_dst, ys_dst, xs_dst] = a[zs_src, ys_src, xs_src]
    return out


def _slices(d: int, n: int):
    """Source/destination slice pair for shifting an axis of length n by d.
    out[dst] = a[src], i.e. out[i] = a[i+d]."""
    if d >= 0:
        return slice(d, n), slice(0, n - d)
    return slice(0, n + d), slice(-d, n)
