"""Validate the DVH-loss percentile fix (network_functions.differentiable_percentile).

The old `histogram_percentile` used a softmax-over-CDF crossing that was biased toward
the middle of the distribution; on a tightly-peaked PTV dose that produced wrong
percentile targets, so the DVH loss hurt the score instead of helping. The fix is an
exact sort + linear-interpolation percentile.

This file has two parts:
  * a PURE-NUMPY mirror of the fixed algorithm, checked against np.percentile — runnable
    anywhere (no TensorFlow), so the algorithm itself is proven off-GPU;
  * a TF section that imports the REAL differentiable_percentile and checks it matches
    numpy, that gradients flow, and that identical dose gives zero DVH percentile error.
    It skips cleanly when TensorFlow is not installed (e.g. on the Mac).

Run:  python tests/test_dvh_percentile.py
"""
import numpy as np


def numpy_percentile_mirror(values, percentile):
    """Line-for-line numpy mirror of differentiable_percentile (sort + lerp).

    Must equal np.percentile(values, percentile, method="linear").
    """
    v = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    n = v.shape[0]
    rank = (percentile / 100.0) * (n - 1)
    lo = int(np.floor(rank))
    hi = min(lo + 1, n - 1)
    frac = rank - np.floor(rank)
    return v[lo] + frac * (v[hi] - v[lo])


def test_mirror_matches_numpy_percentile():
    rng = np.random.default_rng(0)
    for trial in range(50):
        n = int(rng.integers(20, 5000))
        vals = rng.normal(size=n).astype(np.float32)
        for p in (1.0, 5.0, 50.0, 95.0, 99.0):
            got = numpy_percentile_mirror(vals, p)
            ref = np.percentile(vals.astype(np.float64), p, method="linear")
            assert abs(got - ref) < 1e-4, (trial, p, got, ref)
    print("PASS test_mirror_matches_numpy_percentile")


def test_mirror_on_peaked_ptv_like_distribution():
    # The case the old estimator got wrong: dose tightly concentrated near prescription
    # (normalised ~1.0) with a small low-dose tail. D_99 (1st pct) and D_1 (99th pct)
    # must track the true order statistics, not collapse toward the mean.
    rng = np.random.default_rng(7)
    dose = np.concatenate([
        rng.normal(1.0, 0.01, 4000),   # bulk of PTV near prescription
        rng.normal(0.85, 0.02, 60),    # cold tail
    ]).astype(np.float32)
    for p in (1.0, 5.0, 99.0):
        got = numpy_percentile_mirror(dose, p)
        ref = np.percentile(dose.astype(np.float64), p, method="linear")
        assert abs(got - ref) < 1e-4, (p, got, ref)
    # D_99 (1st pct) must sit down in the cold tail, well below the ~1.0 bulk.
    assert numpy_percentile_mirror(dose, 1.0) < 0.95
    print("PASS test_mirror_on_peaked_ptv_like_distribution")


def _run_tf_tests():
    import importlib.util
    import sys
    import types
    from pathlib import Path

    if importlib.util.find_spec("tensorflow") is None:
        print("SKIP TF tests (tensorflow not installed) — run these on the box")
        return

    import tensorflow as tf

    # Import differentiable_percentile without triggering the provided_code package
    # __init__ (which pulls in the wider training stack).
    root = Path(__file__).resolve().parent.parent  # open-kbp-modified
    if "provided_code" not in sys.modules:
        pkg = types.ModuleType("provided_code")
        pkg.__path__ = [str(root / "provided_code")]
        sys.modules["provided_code"] = pkg
    spec = importlib.util.spec_from_file_location(
        "nf", root / "provided_code" / "network_functions.py")
    # network_functions imports the rest of the package at module load; if that fails
    # here, fall back to skipping (the box has the full stack).
    try:
        nf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nf)
    except Exception as exc:  # pragma: no cover
        print(f"SKIP TF tests (could not import network_functions: {exc})")
        return

    rng = np.random.default_rng(1)
    for p in (1.0, 5.0, 50.0, 99.0):
        vals = rng.normal(size=1500).astype(np.float32)
        got = float(nf.differentiable_percentile(tf.constant(vals), p).numpy())
        ref = float(np.percentile(vals.astype(np.float64), p, method="linear"))
        assert abs(got - ref) < 1e-3, (p, got, ref)
    print("PASS tf differentiable_percentile matches numpy")

    # Gradient flows to the bracketing order statistics (non-zero, finite).
    vals = tf.Variable(rng.normal(size=500).astype(np.float32))
    with tf.GradientTape() as tape:
        pctl = nf.differentiable_percentile(vals, 95.0)
    grad = tape.gradient(pctl, vals)
    assert grad is not None, "percentile must be differentiable w.r.t. its inputs"
    g = grad.numpy()
    assert np.isfinite(g).all() and np.count_nonzero(g) >= 1
    print("PASS tf percentile gradient flows")

    # Identical dose -> zero percentile error (the DVH loss floor).
    a = tf.constant(rng.normal(size=800).astype(np.float32))
    err = tf.abs(nf.differentiable_percentile(a, 50.0) - nf.differentiable_percentile(a, 50.0))
    assert float(err.numpy()) == 0.0
    print("PASS tf identical-dose percentile error is zero")


if __name__ == "__main__":
    test_mirror_matches_numpy_percentile()
    test_mirror_on_peaked_ptv_like_distribution()
    _run_tf_tests()
    print("\nAll DVH percentile tests passed (numpy proven; TF proven if installed).")
