"""Tests for provided_code.defense_transforms (numpy/scipy only — no TF/GPU/data).

These cover the mechanism each test-time defence relies on:
  * flip is an EXACT involution (so output dose can be flipped back losslessly),
  * smooth is a genuine low-pass (monotonically reduces local variation),
  * add_noise is clipped and UNBIASED (so averaging draws recovers the signal),
  * scale_intensity rescales and clips.

Run:  python tests/test_defense_transforms.py
"""
import importlib.util
from pathlib import Path

import numpy as np

# Load the module directly (numpy/scipy only) without importing the provided_code
# package, whose __init__ pulls in TensorFlow.
_mod_path = Path(__file__).resolve().parent.parent / "provided_code" / "defense_transforms.py"
_spec = importlib.util.spec_from_file_location("defense_transforms", _mod_path)
dt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dt)

D = H = W = 16


def _tv(x: np.ndarray) -> float:
    """Mean local variation (sum of |neighbour diffs| over the 3 spatial axes)."""
    return float(
        np.mean(np.abs(np.diff(x, axis=0)))
        + np.mean(np.abs(np.diff(x, axis=1)))
        + np.mean(np.abs(np.diff(x, axis=2)))
    )


def _ct(seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, (D, H, W, 1)).astype(np.float32)


def test_flip_is_exact_involution():
    x = _ct()
    once = dt.flip_lr(x)
    twice = dt.flip_lr(once)
    assert np.array_equal(twice, x), "flip_lr must be its own exact inverse"
    assert not np.array_equal(once, x), "flip of a random volume must change it"
    print("PASS test_flip_is_exact_involution")


def test_flip_axis_is_left_right():
    # Mark a single L-R slab; after flip it must land at the mirrored L-R index,
    # with D and W untouched.
    x = np.zeros((D, H, W, 1), dtype=np.float32)
    x[:, 2, :, :] = 1.0  # axis 1 == H == L-R
    f = dt.flip_lr(x)
    assert f[:, H - 1 - 2, :, :].all() and f[:, 2, :, :].sum() == 0
    print("PASS test_flip_axis_is_left_right")


def test_smooth_is_lowpass_and_monotonic():
    x = _ct(1)
    light = dt.smooth_ct(x, 0.5)
    heavy = dt.smooth_ct(x, 2.0)
    assert light.shape == x.shape and light.dtype == np.float32
    # More smoothing -> strictly less local variation than less smoothing -> than none.
    assert _tv(heavy) < _tv(light) < _tv(x)
    # A constant field is unchanged by smoothing.
    const = np.full((D, H, W, 1), 0.4, dtype=np.float32)
    assert np.allclose(dt.smooth_ct(const, 2.0), const, atol=1e-5)
    print("PASS test_smooth_is_lowpass_and_monotonic")


def test_smooth_zero_sigma_is_identity():
    x = _ct(2)
    assert np.array_equal(dt.smooth_ct(x, 0.0), x)
    print("PASS test_smooth_zero_sigma_is_identity")


def test_noise_clipped_and_unbiased():
    # Pick CT values away from the [0,1] clip boundaries so averaging is unbiased.
    x = np.full((D, H, W, 1), 0.5, dtype=np.float32)
    rng = np.random.default_rng(3)
    draws = [dt.add_noise(x, 0.05, rng) for _ in range(400)]
    stacked = np.stack(draws)
    assert stacked.min() >= 0.0 and stacked.max() <= 1.0, "noise must stay in [0,1]"
    assert np.allclose(stacked.mean(0), x, atol=0.01), "averaging draws must recover signal"
    # Reproducible given a seeded rng.
    a = dt.add_noise(x, 0.05, np.random.default_rng(7))
    b = dt.add_noise(x, 0.05, np.random.default_rng(7))
    assert np.array_equal(a, b)
    # std=0 is a no-op copy.
    assert np.array_equal(dt.add_noise(x, 0.0, rng), x)
    print("PASS test_noise_clipped_and_unbiased")


def test_scale_intensity_scales_and_clips():
    x = np.full((D, H, W, 1), 0.4, dtype=np.float32)
    assert np.allclose(dt.scale_intensity(x, 1.5), 0.6, atol=1e-6)
    assert np.allclose(dt.scale_intensity(x, 3.0), 1.0, atol=1e-6)  # clipped at 1
    assert np.array_equal(dt.scale_intensity(x, 1.0), x)            # no-op
    print("PASS test_scale_intensity_scales_and_clips")


if __name__ == "__main__":
    test_flip_is_exact_involution()
    test_flip_axis_is_left_right()
    test_smooth_is_lowpass_and_monotonic()
    test_smooth_zero_sigma_is_identity()
    test_noise_clipped_and_unbiased()
    test_scale_intensity_scales_and_clips()
    print("\nAll defense_transforms tests passed.")
