"""Tests for provided_code.augmentation (numpy/scipy only — no TF/GPU/data needed).

Run:  python tests/test_augmentation.py
"""
import importlib.util
from pathlib import Path

import numpy as np

# Load augmentation.py directly (it only needs numpy/scipy) without importing the
# provided_code package, whose __init__ pulls in TensorFlow / more_itertools.
_aug_path = Path(__file__).resolve().parent.parent / "provided_code" / "augmentation.py"
_spec = importlib.util.spec_from_file_location("augmentation", _aug_path)
_aug = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_aug)
augment_batch, augment_sample = _aug.augment_batch, _aug.augment_sample

D = H = W = 16  # small synthetic volume for speed


def _fake_batch(b=2):
    rng = np.random.default_rng(0)
    ct = rng.uniform(0, 1, (b, D, H, W, 1)).astype(np.float32)
    sm = (rng.uniform(0, 1, (b, D, H, W, 10)) > 0.5).astype(np.float32)  # binary masks
    dose = rng.uniform(0, 1, (b, D, H, W, 1)).astype(np.float32)
    pdm = (rng.uniform(0, 1, (b, D, H, W, 1)) > 0.3).astype(np.float32)
    return ct, sm, dose, pdm


def test_shapes_preserved():
    ct, sm, dose, pdm = _fake_batch()
    o = augment_batch(ct, sm, dose, pdm, rng=np.random.default_rng(1),
                      translate_frac=0.1, rotate_deg=10, scale_range=0.1,
                      elastic_alpha=2.0, noise_std=0.02)
    assert o[0].shape == ct.shape and o[1].shape == sm.shape
    assert o[2].shape == dose.shape and o[3].shape == pdm.shape
    print("PASS test_shapes_preserved")


def test_identity_when_off():
    ct, sm, dose, pdm = _fake_batch()
    # all strengths 0 and flip_prob 0 -> exact passthrough
    o = augment_batch(ct, sm, dose, pdm, rng=np.random.default_rng(2),
                      flip_prob=0.0, intensity_scale=0.0)
    assert np.allclose(o[0], ct) and np.allclose(o[1], sm)
    assert np.allclose(o[2], dose) and np.allclose(o[3], pdm)
    print("PASS test_identity_when_off")


def test_masks_stay_binary():
    ct, sm, dose, pdm = _fake_batch()
    o = augment_batch(ct, sm, dose, pdm, rng=np.random.default_rng(3),
                      flip_prob=0.0, intensity_scale=0.0,
                      rotate_deg=15, translate_frac=0.1, elastic_alpha=2.0)
    sm_vals = np.unique(o[1]); pdm_vals = np.unique(o[3])
    assert set(np.unique(sm_vals)).issubset({0.0, 1.0}), sm_vals
    assert set(np.unique(pdm_vals)).issubset({0.0, 1.0}), pdm_vals
    print("PASS test_masks_stay_binary (nearest interp keeps masks binary)")


def test_registration_preserved():
    # A single bright voxel in CT and dose at the SAME location must move together.
    ct = np.zeros((1, D, H, W, 1), np.float32); dose = np.zeros_like(ct)
    sm = np.zeros((1, D, H, W, 10), np.float32); pdm = np.zeros((1, D, H, W, 1), np.float32)
    ct[0, 8, 5, 11, 0] = 1.0; dose[0, 8, 5, 11, 0] = 1.0
    sm[0, 8, 5, 11, :] = 1.0; pdm[0, 8, 5, 11, 0] = 1.0
    o = augment_sample(ct[0], sm[0], dose[0], pdm[0], np.random.default_rng(4),
                       flip_prob=0.0, intensity_scale=0.0, rotate_deg=20, translate_frac=0.1)
    ct_pos = np.unravel_index(np.argmax(o[0]), o[0].shape)
    dose_pos = np.unravel_index(np.argmax(o[2]), o[2].shape)
    assert ct_pos[:3] == dose_pos[:3], (ct_pos, dose_pos)
    print(f"PASS test_registration_preserved (CT & dose marker co-located at {ct_pos[:3]})")


def test_geometric_actually_changes():
    ct, sm, dose, pdm = _fake_batch(1)
    o = augment_sample(ct[0], sm[0], dose[0], pdm[0], np.random.default_rng(5),
                       flip_prob=0.0, intensity_scale=0.0, rotate_deg=20)
    assert not np.allclose(o[0], ct[0]), "rotation should change the CT"
    print("PASS test_geometric_actually_changes")


def test_ct_only_perturbation():
    # noise/intensity must touch CT only, leave dose & masks identical
    ct, sm, dose, pdm = _fake_batch(1)
    o = augment_sample(ct[0], sm[0], dose[0], pdm[0], np.random.default_rng(6),
                       flip_prob=0.0, intensity_scale=0.2, noise_std=0.05)
    assert not np.allclose(o[0], ct[0]), "CT should change"
    assert np.allclose(o[1], sm[0]) and np.allclose(o[2], dose[0]), "dose/masks must be untouched"
    assert o[0].min() >= 0.0 and o[0].max() <= 1.0, "CT must stay in [0,1]"
    print("PASS test_ct_only_perturbation")


if __name__ == "__main__":
    test_shapes_preserved()
    test_identity_when_off()
    test_masks_stay_binary()
    test_registration_preserved()
    test_geometric_actually_changes()
    test_ct_only_perturbation()
    print("\nALL AUGMENTATION TESTS PASSED")
