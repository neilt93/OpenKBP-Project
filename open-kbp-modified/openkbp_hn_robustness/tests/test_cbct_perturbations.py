#!/usr/bin/env python3
"""Off-GPU tests for the CBCT-characteristic perturbations (P6 scatter/cupping, P7 ring,
P8 truncation). Pure numpy/scipy — no model, no data, no TF.

Run: python tests/test_cbct_perturbations.py   (also pytest-compatible)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))

from perturbations import ScatterCupping, RingArtifact, Truncation  # noqa: E402
from perturbations.base import HU_CLIP_MIN, HU_CLIP_MAX, VOLUME_SHAPE  # noqa: E402


def _synthetic():
    """A blobby body inside air: central ellipsoid of soft tissue + a bone core."""
    z, y, x = np.indices(VOLUME_SHAPE)
    cz, cy, cx = (np.array(VOLUME_SHAPE) - 1) / 2.0
    # In-plane extent (63) fills the FOV so truncation (FOV radius 48–61) bites at every level,
    # like real H&N anatomy/shoulders reaching the periphery.
    rr = ((z - cz) / 55) ** 2 + ((y - cy) / 63) ** 2 + ((x - cx) / 63) ** 2
    ct = np.where(rr <= 1.0, 1000.0, 0.0)                       # soft tissue ~1000 HU (OpenKBP coords)
    bone_core = (np.sqrt((y - cy) ** 2 + (x - cx) ** 2) < 8) & (rr <= 1.0)   # 3D bone cylinder in body
    ct[bone_core] += 1800.0
    ct = np.clip(ct, 0, HU_CLIP_MAX)
    return ct, ct > 0


def _common_invariants(cls, name):
    ct, body = _synthetic()
    rng = np.random.default_rng(0)
    p = cls()
    changed_prev = -1.0
    for lvl in p.levels:
        out = p.apply(ct.copy(), body, lvl, rng)
        assert out.shape == VOLUME_SHAPE, f"{name} {lvl}: shape changed"
        assert np.all(out >= 0) and np.all(out <= HU_CLIP_MAX), f"{name} {lvl}: out of [0,{HU_CLIP_MAX}]"
        assert np.all(out[~body] == 0), f"{name} {lvl}: air voxels not zero"
        in_fov_body = (out > 0)
        assert np.all(out[in_fov_body] >= HU_CLIP_MIN), f"{name} {lvl}: kept voxels below HU_CLIP_MIN"
        assert not np.allclose(out, ct), f"{name} {lvl}: perturbation did nothing"
    return p, ct, body


def test_scatter_cupping_monotone():
    p, ct, body = _common_invariants(ScatterCupping, "P6")
    rng = np.random.default_rng(0)
    mags = [np.mean(np.abs(p.apply(ct.copy(), body, l, rng) - ct)[body]) for l in p.levels]
    assert all(b >= a - 1e-6 for a, b in zip(mags, mags[1:])), f"P6 not monotone in severity: {mags}"
    # cupping depresses the centre more than the periphery
    out = p.apply(ct.copy(), body, "L5", rng)
    cz, cy, cx = (np.array(VOLUME_SHAPE) - 1) // 2
    center_drop = ct[cz, cy, cx] - out[cz, cy, cx]
    assert center_drop > 0, "P6 L5 did not depress the centre"


def test_ring_zero_mean_oscillation():
    p, ct, body = _common_invariants(RingArtifact, "P7")
    rng = np.random.default_rng(0)
    out = p.apply(ct.copy(), body, "L3", rng)
    diff = (out - ct)[body]
    assert diff.min() < 0 and diff.max() > 0, "P7 rings should push HU both up and down"


def test_truncation_removes_peripheral_anatomy():
    p, ct, body = _common_invariants(Truncation, "P8")
    rng = np.random.default_rng(0)
    zeroed = [np.sum((ct > 0) & (p.apply(ct.copy(), body, l, rng) == 0)) for l in p.levels]
    assert zeroed[0] >= 0
    assert all(b >= a for a, b in zip(zeroed, zeroed[1:])), f"P8 truncation not monotone: {zeroed}"
    assert zeroed[-1] > zeroed[0], "P8 L5 should truncate more anatomy than L1"


if __name__ == "__main__":
    fns = [test_scatter_cupping_monotone, test_ring_zero_mean_oscillation,
           test_truncation_removes_peripheral_anatomy]
    ok = True
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}")
        except AssertionError as e:
            ok = False; print(f"FAIL {fn.__name__}: {e}")
    print("[cbct tests] PASS" if ok else "[cbct tests] FAIL")
    sys.exit(0 if ok else 1)
