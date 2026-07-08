"""Off-GPU tests for the bug-pass fixes.

Fix C (precomputed-NPZ order guard, `data_loader._load_from_precomputed`) is fully
testable here — `data_loader.py` has no TensorFlow import. Fix A (drop the spurious S-I
head-to-toe flip in `network_functions.augment_batch_tf`, keep only the L-R flip) is
checked two ways: a pure-numpy mirror of the flip axis that runs anywhere, and the REAL
`augment_batch_tf` when TensorFlow is installed (the box).

Run:  python tests/test_bugfixes.py
"""
import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # open-kbp-modified

# Fake provided_code package so submodules import without running __init__ (which pulls TF).
if "provided_code" not in sys.modules:
    _pkg = types.ModuleType("provided_code")
    _pkg.__path__ = [str(ROOT / "provided_code")]
    sys.modules["provided_code"] = _pkg


def _load(mod):
    spec = importlib.util.spec_from_file_location(mod, ROOT / "provided_code" / f"{mod}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


DataLoader = _load("data_loader").DataLoader


def _make_npz(tmp: Path, ids, with_ids=True) -> Path:
    """NPZ whose ct row i is marked with the scalar value i, in the order `ids`."""
    n = len(ids)
    arrs = {"ct": np.arange(n, dtype=np.float32).reshape(n, 1, 1, 1, 1)}
    if with_ids:
        arrs["patient_ids"] = np.array(ids)
    p = tmp / "pc.npz"
    np.savez(p, **arrs)
    return p


def _loader(paths, npz):
    return DataLoader([Path(f"/x/{p}") for p in paths], precomputed_path=npz, cache_data=False)


def test_npz_order_guard_maps_to_true_saved_row(tmp):
    # Saved order pt_1,pt_2,pt_3 -> rows 0,1,2. Request a DIFFERENT order.
    npz = _make_npz(tmp, ["pt_1", "pt_2", "pt_3"])
    loader = _loader(["pt_3", "pt_1", "pt_2"], npz)
    loader._load_from_precomputed()
    assert loader._patient_to_idx == {"pt_3": 2, "pt_1": 0, "pt_2": 1}, loader._patient_to_idx
    # The row served for pt_3 must be its true saved row (value 2), not positional (would be 0).
    assert loader._stacked_data["ct"][loader._patient_to_idx["pt_3"]].flatten()[0] == 2.0
    print("PASS test_npz_order_guard_maps_to_true_saved_row")


def test_npz_without_patient_ids_raises(tmp):
    npz = _make_npz(tmp, ["pt_1", "pt_2"], with_ids=False)
    loader = _loader(["pt_1", "pt_2"], npz)
    try:
        loader._load_from_precomputed()
        assert False, "expected a raise for an NPZ with no patient_ids"
    except ValueError as e:
        assert "patient_ids" in str(e), e
    print("PASS test_npz_without_patient_ids_raises")


def test_npz_missing_requested_patient_raises(tmp):
    npz = _make_npz(tmp, ["pt_1", "pt_2"])
    loader = _loader(["pt_1", "pt_99"], npz)  # pt_99 not in the NPZ
    try:
        loader._load_from_precomputed()
        assert False, "expected a raise when a requested patient is absent from the NPZ"
    except ValueError as e:
        assert "missing" in str(e).lower(), e
    print("PASS test_npz_missing_requested_patient_raises")


def _encode_hw(D, H, W):
    """Volume whose voxel value encodes (h*10 + w): h = L-R (axis 2), w = S-I (axis 3)."""
    x = np.zeros((1, D, H, W, 1), np.float32)
    for h in range(H):
        for w in range(W):
            x[0, :, h, w, 0] = h * 10 + w
    return x


def test_flip_axis_numpy_mirror():
    # BDHWC: axis 2 = H = L-R (the valid flip), axis 3 = W = S-I (must NOT be flipped).
    D = H = W = 4
    x = _encode_hw(D, H, W)
    f = np.flip(x, axis=2)   # the fixed augment_batch_tf reverses axis 2 only
    for h in range(H):
        for w in range(W):
            # axis 2 (L-R) reversed -> h maps to H-1-h; axis 3 (S-I) unchanged -> w stays.
            assert f[0, 0, h, w, 0] == (H - 1 - h) * 10 + w, (h, w, f[0, 0, h, w, 0])
    print("PASS test_flip_axis_numpy_mirror")


def _tf_augment_test():
    if importlib.util.find_spec("tensorflow") is None:
        print("SKIP tf augment_batch_tf test (no tensorflow) — runs on the box")
        return
    import tensorflow as tf
    nf = _load("network_functions")
    B, D, H, W = 1, 4, 4, 4
    ct = _encode_hw(D, H, W)  # voxel value = h*10 + w
    sm = np.zeros((B, D, H, W, 10), np.float32)
    dose = np.zeros((B, D, H, W, 1), np.float32)
    pdm = np.ones((B, D, H, W, 1), np.float32)
    # flip_prob=1 forces the flip; intensity_scale=0 isolates the geometry.
    out_ct, _, _, _ = nf.augment_batch_tf(
        tf.constant(ct), tf.constant(sm), tf.constant(dose), tf.constant(pdm),
        flip_prob=1.0, intensity_scale=0.0)
    o = out_ct.numpy()
    for h in range(H):
        for w in range(W):
            # After the fix: L-R (axis 2) flipped, S-I (axis 3) untouched.
            assert o[0, 0, h, w, 0] == (H - 1 - h) * 10 + w, (h, w, o[0, 0, h, w, 0])
    print("PASS tf augment_batch_tf flips only L-R (axis 2), not S-I")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_npz_order_guard_maps_to_true_saved_row(tmp)
        test_npz_without_patient_ids_raises(tmp)
        test_npz_missing_requested_patient_raises(tmp)
    test_flip_axis_numpy_mirror()
    _tf_augment_test()
    print("\nAll bugfix tests passed (numpy + data_loader proven; TF flip proven if installed).")
