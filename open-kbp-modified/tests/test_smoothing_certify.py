"""Tests for provided_code.smoothing_certify (pure numpy/scipy, NO TF).

These pin the certificate math — the novel core — before any paid GPU run:
  * the percentile-shift formula matches the Chiang et al. median-smoothing bound;
  * the order-statistic ranks give the intended coverage (a Monte-Carlo check that
    the certified interval actually traps the shifted-percentile truth at the
    advertised confidence);
  * monotonicity / clamping edge cases behave.

Run:  python tests/test_smoothing_certify.py
"""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent

if "provided_code" not in sys.modules:
    _pkg = types.ModuleType("provided_code")
    _pkg.__path__ = [str(ROOT / "provided_code")]
    sys.modules["provided_code"] = _pkg

_spec = importlib.util.spec_from_file_location(
    "smoothing_certify", ROOT / "provided_code" / "smoothing_certify.py")
sc = importlib.util.module_from_spec(_spec)
sys.modules["smoothing_certify"] = sc  # so @dataclass can resolve its own module
_spec.loader.exec_module(sc)


def test_certified_percentiles_median():
    # p=0.5 => symmetric bracket Phi(-R/sigma), Phi(+R/sigma).
    p_lo, p_hi = sc.certified_percentiles(radius=1.0, sigma=1.0, p=0.5)
    assert abs(p_lo - norm.cdf(-1.0)) < 1e-12
    assert abs(p_hi - norm.cdf(+1.0)) < 1e-12
    # radius 0 => degenerate point interval.
    assert sc.certified_percentiles(0.0, 1.0, 0.5) == (0.5, 0.5)


def test_certified_percentiles_scaling():
    # Larger sigma (more smoothing) => tighter percentile shift for the same radius
    # => the bracket sits closer to the median. Verify p_lo rises toward 0.5.
    lo_small, _ = sc.certified_percentiles(1.0, 0.5, 0.5)
    lo_big, _ = sc.certified_percentiles(1.0, 4.0, 0.5)
    assert lo_small < lo_big < 0.5


def test_ranks_monotone_and_bracketing():
    # Ranks must satisfy 0 <= j < k <= n+1 and widen (j down, k up) as radius grows.
    n = 2000
    j1, k1, _, _ = sc.certified_ranks(n, radius=0.5, sigma=1.0, alpha=0.001)
    j2, k2, _, _ = sc.certified_ranks(n, radius=2.0, sigma=1.0, alpha=0.001)
    assert 0 <= j1 < k1 <= n + 1
    assert j2 <= j1 and k2 >= k1  # bigger radius => wider (more conservative) interval


def test_ranks_fall_off_end_when_underpowered():
    # Tiny n at a large radius: the shifted percentile is so extreme the bound runs
    # off the sample; j should hit 0 (or k hit n+1), flagging "uncertified".
    j, k, p_lo, p_hi = sc.certified_ranks(n=20, radius=5.0, sigma=1.0, alpha=0.001)
    assert j == 0 or k == 21


def test_certify_from_samples_shapes_and_ordering():
    rng = np.random.default_rng(0)
    n, V = 500, 37
    samples = rng.normal(size=(n, V))
    cert = sc.certify_from_samples(samples, radius=0.5, sigma=1.0, alpha=0.05)
    assert cert.lower.shape == (V,) and cert.upper.shape == (V,)
    assert np.all(cert.lower <= cert.upper + 1e-9)          # interval well-formed
    assert np.all(cert.lower <= cert.median + 1e-9)
    assert np.all(cert.median <= cert.upper + 1e-9)


def test_coverage_monte_carlo():
    """The certified interval must trap the TRUE shifted-percentile prediction at
    >= 1 - alpha. Simulate the smoothed regressor as a fixed 1-D standard normal
    (the noise distribution of f(x+delta) at one voxel). The true p_lo/p_hi
    percentiles of that distribution are the quantities the interval must bracket;
    check the empirical order statistics do so across many resamplings.
    """
    rng = np.random.default_rng(1)
    n = 1000
    radius, sigma, alpha = 1.0, 1.0, 0.05
    j, k, p_lo, p_hi = sc.certified_ranks(n, radius, sigma, alpha=alpha)
    true_lo = norm.ppf(p_lo)   # true p_lo-quantile of N(0,1)
    true_hi = norm.ppf(p_hi)
    trials = 2000
    lo_ok = hi_ok = 0
    for _ in range(trials):
        s = np.sort(rng.normal(size=n))
        if s[j - 1] <= true_lo:        # lower order stat is a valid lower bound
            lo_ok += 1
        if s[k - 1] >= true_hi:        # upper order stat is a valid upper bound
            hi_ok += 1
    # Each side is a (1 - alpha/2) bound; allow Monte-Carlo slack.
    assert lo_ok / trials >= 1 - alpha / 2 - 0.02, lo_ok / trials
    assert hi_ok / trials >= 1 - alpha / 2 - 0.02, hi_ok / trials


def test_per_voxel_rms_equivalent():
    # R over V voxels => per-voxel RMS R/sqrt(V).
    assert abs(sc.per_voxel_rms_equivalent(10.0, 100) - 1.0) < 1e-12


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nAll {len(fns)} smoothing-certificate tests passed.")
