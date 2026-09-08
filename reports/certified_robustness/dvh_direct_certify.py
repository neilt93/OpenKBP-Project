#!/usr/bin/env python3
"""Tighter certified DVH intervals: certify each DVH metric DIRECTLY as a scalar
median-smoothed functional, instead of the loose per-voxel monotone push-through.

Limitation #4 of the certified strand was that the DVH interval was built by pushing per-voxel
[lower, upper] dose bounds through the DVH metric — a worst-case that assumes every voxel hits its
bound simultaneously and ignores the spatial correlation of the smoothed prediction, overstating
the interval.

Fix: a DVH metric (D95, Dmean, D0.1cc, ...) is a deterministic real-valued function g(dose(x)) of
the input. Median randomised smoothing (Chiang et al. 2020) certifies ANY real-valued output, so we
compute g on each of the n noisy-CT dose draws -> n samples of the metric -> certify the metric
directly with the SAME order-statistic construction (certify_from_samples). This uses the metric's
own distribution under noise (correlation-aware) and is provably valid AND tighter than the
push-through.

This module needs only the n per-draw DVH-metric values (computable on the pod from the stored
draws). It reuses provided_code/smoothing_certify.py's certified order-statistics; --self-test
validates coverage + the tightness gain on synthetic correlated data with no pod.
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_SC_PATH = os.path.normpath(os.path.join(
    HERE, "..", "..", "open-kbp-modified", "provided_code", "smoothing_certify.py"))


def _load_smoothing_certify():
    """Import smoothing_certify.py directly (numpy/scipy only) without provided_code/__init__,
    which would pull in TensorFlow."""
    spec = importlib.util.spec_from_file_location("smoothing_certify", _SC_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["smoothing_certify"] = mod
    spec.loader.exec_module(mod)
    return mod


SC = _load_smoothing_certify()


def certify_dvh_direct(metric_samples, radius, sigma, alpha=0.001, p=0.5):
    """Certified [lower, upper] interval for ONE DVH metric, from its n per-draw values.

    metric_samples: 1-D array of the DVH metric computed on each of the n noisy-CT dose draws.
    Returns (lower, upper, certified_low, certified_high) at joint confidence 1-alpha for the
    smoothed p-percentile metric under any L2 CT perturbation of size `radius`.
    """
    s = np.asarray(metric_samples, dtype=float).reshape(-1, 1)   # (n, 1) -> one "voxel"
    c = SC.certify_from_samples(s, radius=radius, sigma=sigma, p=p, alpha=alpha)
    return float(c.lower[0]), float(c.upper[0]), bool(c.certified_low), bool(c.certified_high)


def pushthrough_interval(voxel_samples, volume_frac, radius, sigma, alpha=0.001, p=0.5):
    """The OLD per-voxel push-through, for comparison: certify every voxel's dose, then read the
    D_{volume_frac} (dose-to-volume percentile) off the lower- and upper-bound dose volumes.

    voxel_samples: (n, V) per-draw dose for the V structure voxels. Returns (lower, upper) for the
    D_{volume_frac} metric (e.g. volume_frac=0.95 for D95)."""
    c = SC.certify_from_samples(np.asarray(voxel_samples, float), radius=radius, sigma=sigma,
                                p=p, alpha=alpha)
    # D_{v}: dose received by at least v-fraction of the volume = (1-v) upper quantile of doses
    q = 100.0 * (1.0 - volume_frac)
    return float(np.percentile(c.lower, q)), float(np.percentile(c.upper, q))


def _dose_to_volume(dose_vol, volume_frac):
    """D_{volume_frac}: dose s.t. volume_frac of voxels receive >= it (per draw)."""
    return np.percentile(dose_vol, 100.0 * (1.0 - volume_frac), axis=-1)


def self_test():
    """Validate on synthetic correlated voxel data: (1) direct certification is empirically valid
    (>= 1-alpha coverage of the true smoothed metric), (2) it is TIGHTER than the push-through."""
    rng = np.random.default_rng(0)
    sigma, radius, alpha = 0.05, 0.025, 0.1     # R/sigma=0.5, well within R_max at n=200
    V, n, trials = 400, 200, 300
    vfrac = 0.95                                 # D95

    # Ground-truth smoothed D95 = the true p-percentile of D95 under the noise law. Approximate it
    # with a large reference draw from the same generative model.
    def draw(n_draws):
        base = rng.normal(60, 3, size=V)                      # per-voxel mean dose (Gy)
        shared = rng.normal(0, 1.5, size=(n_draws, 1))        # correlated (shared) noise component
        indep = rng.normal(0, 1.0, size=(n_draws, V))         # independent component
        return base[None, :] + shared + indep                # (n_draws, V)

    true_d95 = float(np.median(_dose_to_volume(draw(20000), vfrac)))

    cov_direct = cov_push = 0
    w_direct = w_push = 0.0
    for _ in range(trials):
        vox = draw(n)                                          # (n, V)
        d95_samples = _dose_to_volume(vox, vfrac)             # (n,)
        lo_d, hi_d, cl, ch = certify_dvh_direct(d95_samples, radius, sigma, alpha)
        lo_p, hi_p = pushthrough_interval(vox, vfrac, radius, sigma, alpha)
        cov_direct += (lo_d <= true_d95 <= hi_d)
        cov_push += (lo_p <= true_d95 <= hi_p)
        w_direct += hi_d - lo_d
        w_push += hi_p - lo_p
    cov_direct /= trials; cov_push /= trials
    w_direct /= trials; w_push /= trials

    ok = True
    print(f"target coverage {1-alpha:.2f}   true smoothed D95 = {true_d95:.2f} Gy")
    print(f"  DIRECT      : coverage={cov_direct:.3f}  mean width={w_direct:.2f} Gy")
    print(f"  push-through: coverage={cov_push:.3f}  mean width={w_push:.2f} Gy")
    print(f"  tightening: direct is {100*(1-w_direct/w_push):.0f}% narrower")
    if not (cov_direct >= 1 - alpha - 0.03):  ok = False; print("  FAIL: direct under-covers")
    if not (w_direct < w_push):               ok = False; print("  FAIL: direct not tighter")
    print("[self-test] PASS" if ok else "[self-test] FAIL")
    raise SystemExit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description="Direct (tighter) certified DVH-metric intervals")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
    print("Import certify_dvh_direct(metric_samples, radius, sigma, alpha) and feed it the n "
          "per-draw DVH-metric values (computed on the pod from the stored smoothing draws).")


if __name__ == "__main__":
    main()
