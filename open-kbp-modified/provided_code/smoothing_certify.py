"""Median (percentile) randomised-smoothing certificate for voxel-wise dose regression.

Pure numpy/scipy (NO TensorFlow) so the certificate arithmetic — the part most
likely to be wrong, and the novel core of the project — is unit-testable off-GPU
before any paid RunPod run. The TF orchestration (drawing noisy CTs, running the
model) lives in `certify_smoothing.py`.

Why median, not mean, smoothing
-------------------------------
Cohen et al. (2019) certify *classification* by smoothing the class vote. Dose
prediction is dense voxel-wise *regression*, so the object we smooth is a real
number per voxel and the mean has no robustness certificate. The result that
does carry over is percentile smoothing (Chiang et al. 2020, "Detection as
Regression"):

    Define the smoothed p-percentile prediction at a voxel
        h_p(x) = p-th percentile of  f(x + delta),  delta ~ N(0, sigma^2 I).
    Then for every perturbation ||eps||_2 <= R,
        h_{p_lo}(x)  <=  h_p(x + eps)  <=  h_{p_hi}(x)
    with
        p_lo = Phi(Phi^{-1}(p) - R/sigma),   p_hi = Phi(Phi^{-1}(p) + R/sigma).

For the median predictor p = 0.5 (Phi^{-1}(0.5) = 0) this collapses to the clean
statement used here: under any L2 CT perturbation of radius R, the smoothed dose
at each voxel is provably trapped in
        [ h_{Phi(-R/sigma)}(x) ,  h_{Phi(+R/sigma)}(x) ].

Finite samples -> a high-probability certificate
------------------------------------------------
We cannot evaluate a percentile exactly; we draw n noisy CTs and use order
statistics. `certified_ranks` picks two ranks (j, k) so that, with total
confidence >= 1 - alpha, the j-th smallest sample lower-bounds the true p_lo
percentile and the k-th smallest upper-bounds the true p_hi percentile
(Clopper-Pearson-style, binomial-exact, alpha split two ways). The resulting
per-voxel interval [y_(j), y_(k)] then holds jointly at confidence 1 - alpha.

Certified DVH intervals
-----------------------
Every OpenKBP DVH metric (mean, D_0.1_cc, D_99, D_95, D_1) is monotonically
non-decreasing in each voxel's dose. So feeding the per-voxel LOWER-bound volume
through the metric yields a certified lower bound on the metric, and the
per-voxel UPPER-bound volume a certified upper bound — no extra probability
budget spent. That is what turns a per-voxel guarantee into the clinically
meaningful statement ("no CT perturbation of radius R can push D95 outside
[a, b] Gy"). The monotone push-through is done by the orchestrator, which owns
the evaluator; this module only produces the per-voxel dose bounds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binom, norm


def certified_percentiles(radius: float, sigma: float, p: float = 0.5) -> tuple[float, float]:
    """(p_lo, p_hi): the two percentile levels whose true values bracket the
    smoothed p-percentile prediction under any L2 perturbation of size `radius`.

    radius == 0 returns (p, p) (no perturbation -> the interval is the point
    estimate). sigma must be > 0.
    """
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if radius < 0:
        raise ValueError("radius must be >= 0")
    z = norm.ppf(p)
    ratio = radius / sigma
    return float(norm.cdf(z - ratio)), float(norm.cdf(z + ratio))


def certified_ranks(n: int, radius: float, sigma: float, p: float = 0.5,
                    alpha: float = 0.001) -> tuple[int, int, float, float]:
    """Order-statistic ranks (j, k), 1-indexed, for an n-sample two-sided
    certificate at confidence 1 - alpha.

    With samples sorted ascending y_(1) <= ... <= y_(n):
      * y_(j) is a (1 - alpha/2) confidence LOWER bound on the true p_lo percentile,
      * y_(k) is a (1 - alpha/2) confidence UPPER bound on the true p_hi percentile.
    Together the interval [y_(j), y_(k)] holds at confidence >= 1 - alpha (union bound).

    Returns (j, k, p_lo, p_hi). j may be 0 and k may be n+1 to signal "not enough
    samples to certify at this radius/confidence" — i.e. the bound falls off the
    end of the sample and the caller must treat that voxel as UNCERTIFIED (clamp
    to the data range) rather than pretend a bound exists.

    Construction (binomial-exact, no normal approximation):
      Lower bound on quantile q_{p_lo}: the largest j with
        P(Bin(n, p_lo) <= j - 1) <= alpha/2   =>  P(y_(j) <= q_{p_lo}) >= 1 - alpha/2.
      Upper bound on quantile q_{p_hi}: the smallest k with
        P(Bin(n, p_hi) <= k - 1) >= 1 - alpha/2 => P(y_(k) >= q_{p_hi}) >= 1 - alpha/2.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    p_lo, p_hi = certified_percentiles(radius, sigma, p)
    half = alpha / 2.0

    # Lower rank j: largest j in [1, n] with binom.cdf(j-1; n, p_lo) <= half.
    # binom.cdf is non-decreasing in its first arg, so find the boundary.
    j = 0
    for jj in range(1, n + 1):
        if binom.cdf(jj - 1, n, p_lo) <= half:
            j = jj
        else:
            break

    # Upper rank k: smallest k in [1, n] with binom.cdf(k-1; n, p_hi) >= 1 - half.
    k = n + 1
    for kk in range(1, n + 1):
        if binom.cdf(kk - 1, n, p_hi) >= 1.0 - half:
            k = kk
            break

    return j, k, p_lo, p_hi


@dataclass
class VoxelCertificate:
    """Per-voxel certified dose interval (in whatever units `samples` were in)."""
    lower: NDArray          # (V,) certified lower bound per voxel
    upper: NDArray          # (V,) certified upper bound per voxel
    median: NDArray         # (V,) smoothed median estimate (point prediction)
    j: int                  # lower order-statistic rank used (0 = uncertified low)
    k: int                  # upper order-statistic rank used (n+1 = uncertified high)
    p_lo: float
    p_hi: float
    certified_low: bool     # True if j >= 1 (a real lower bound exists)
    certified_high: bool    # True if k <= n


def certify_from_samples(samples: NDArray, radius: float, sigma: float,
                        p: float = 0.5, alpha: float = 0.001) -> VoxelCertificate:
    """Turn n Monte-Carlo prediction samples into per-voxel certified bounds.

    `samples` is (n, V): n noisy-CT predictions, V voxels each. Returns the
    per-voxel [lower, upper] interval guaranteed to contain the smoothed p-percentile
    prediction under any L2 CT perturbation of radius `radius`, at confidence 1-alpha.

    When a rank falls off the end of the sample (j == 0 or k == n+1) the bound is
    clamped to the empirical min/max of that voxel and the corresponding
    `certified_*` flag is set False, so the caller never mistakes "ran out of
    samples" for a real guarantee.
    """
    samples = np.asarray(samples)
    if samples.ndim != 2:
        raise ValueError(f"samples must be (n, V), got shape {samples.shape}")
    n = samples.shape[0]
    j, k, p_lo, p_hi = certified_ranks(n, radius, sigma, p, alpha)

    ordered = np.sort(samples, axis=0)            # ascending along the sample axis
    median = np.median(samples, axis=0)

    certified_low = j >= 1
    certified_high = k <= n
    # 1-indexed rank -> 0-indexed row; clamp to the data range when uncertified.
    lower = ordered[j - 1] if certified_low else ordered[0]
    upper = ordered[k - 1] if certified_high else ordered[-1]
    return VoxelCertificate(lower=lower, upper=upper, median=median,
                            j=j, k=k, p_lo=p_lo, p_hi=p_hi,
                            certified_low=certified_low, certified_high=certified_high)


def per_voxel_rms_equivalent(radius: float, n_voxels: int) -> float:
    """An L2 radius R over the whole volume corresponds to a per-voxel RMS
    perturbation of R / sqrt(V). Reported alongside R so the certificate can be
    stated honestly in per-voxel (HU) terms, not just as an abstract L2 ball."""
    if n_voxels <= 0:
        raise ValueError("n_voxels must be > 0")
    return radius / np.sqrt(n_voxels)
