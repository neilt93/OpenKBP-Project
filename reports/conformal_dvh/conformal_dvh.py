#!/usr/bin/env python3
"""Conformal DVH intervals + coverage-vs-severity stress test.

Split-conformal (distribution-free, finite-sample) prediction intervals on DVH criteria,
calibrated on CLEAN held-out patients, then evaluated for empirical coverage as the test
distribution shifts across the committed P1-P5 CT-perturbation battery. The clean calibrator
is fixed; shift shows up as coverage DROPPING (interval widths are frozen from clean
calibration) -- that is the conformal failure signature under non-exchangeability.

Two coverage notions, both reported:
  * MARGINAL, per criterion: interval [pred +/- q_j * sigma_j] with per-criterion 90% coverage.
  * JOINT, per patient: one max-normalized score per patient -> a simultaneous band over ALL
    criteria at once (the honest analog of the collective-DVH certificate; wider, but a real
    per-patient guarantee across 23 correlated criteria).

DATA: consumes the `per_patient_dvh` field that evaluate_metrics.py now emits (per-patient,
per-criterion SIGNED residual pred-ref in Gy). If that field is absent (metrics predate the
enabler patch), run `--self-test` -- it validates the coverage math on synthetic data with a
known distribution, no pod required.

Usage:
  python conformal_dvh.py --self-test
  python conformal_dvh.py --alpha 0.1 --cal-frac 0.5        # real run once residuals exist
"""
import argparse
import csv
import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = os.path.normpath(os.path.join(
    HERE, "..", "..", "open-kbp-modified", "openkbp_hn_robustness", "metrics", "per_patient"))


# ------------------------------------------------------------------ core math ---
def conformal_quantile(cal_scores, alpha):
    """Finite-sample split-conformal quantile: the ceil((n+1)(1-alpha))/n empirical quantile
    of calibration scores, giving marginal coverage >= 1-alpha (Vovk; Lei et al.)."""
    n = len(cal_scores)
    if n == 0:
        return math.inf
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:                      # not enough calibration points to certify at this alpha
        return math.inf
    return float(np.sort(cal_scores)[k - 1])


def robust_scale(abs_resid):
    """Per-criterion scale for normalization: MAD (robust), floored to avoid divide-by-zero."""
    mad = np.median(np.abs(abs_resid - np.median(abs_resid)))
    s = 1.4826 * mad
    return float(s) if s > 1e-6 else max(float(np.std(abs_resid)), 1e-6)


def fit_conformal(cal_resid_by_crit, alpha):
    """Fit marginal per-criterion q_j and the joint max-score quantile Q on CLEAN calibration.

    cal_resid_by_crit: {criterion: np.array of SIGNED residuals over calibration patients}
    Returns (sigma[crit], q[crit], Q_joint).
    """
    crits = list(cal_resid_by_crit)
    sigma = {c: robust_scale(cal_resid_by_crit[c]) for c in crits}
    # marginal: score_j = |resid_j| / sigma_j
    q = {c: conformal_quantile(np.abs(cal_resid_by_crit[c]) / sigma[c], alpha) for c in crits}
    # joint: one score per calibration patient = max_c |resid_c|/sigma_c  (aligned patient order)
    P = np.column_stack([np.abs(cal_resid_by_crit[c]) / sigma[c] for c in crits])
    max_scores = P.max(axis=1)
    Q = conformal_quantile(max_scores, alpha)
    return sigma, q, Q


def coverage(test_resid_by_crit, sigma, q, Q):
    """Empirical coverage on a test condition. Returns (mean marginal coverage over criteria,
    joint per-patient coverage)."""
    crits = list(q)
    marg = []
    for c in crits:
        s = np.abs(test_resid_by_crit[c]) / sigma[c]
        marg.append(float(np.mean(s <= q[c])) if q[c] != math.inf else 0.0)
    P = np.column_stack([np.abs(test_resid_by_crit[c]) / sigma[c] for c in crits])
    joint = float(np.mean(P.max(axis=1) <= Q)) if Q != math.inf else 0.0
    return float(np.mean(marg)), joint


# --------------------------------------------------------------- data loading ---
def load_condition(cond):
    p = os.path.join(METRICS, f"{cond}.json")
    d = json.load(open(p))
    if "per_patient_dvh" not in d:
        return None
    # {criterion: {patient: signed resid}} -> {criterion: (patients, array)}
    out = {}
    for crit, perpt in d["per_patient_dvh"].items():
        pts = sorted(perpt)
        out[crit] = (pts, np.array([perpt[k] for k in pts], dtype=float))
    return out


# --------------------------------------------------------------------- self test
def self_test():
    """Validate the coverage math on synthetic residuals with a known distribution."""
    rng = np.random.default_rng(0)
    alpha = 0.1
    crits = [f"crit{i}" for i in range(23)]
    # heteroscedastic clean residuals; calibration and clean-test are exchangeable
    scales = {c: 0.5 + 2.0 * rng.random() for c in crits}
    cal = {c: rng.normal(0, scales[c], size=100) for c in crits}
    clean_test = {c: rng.normal(0, scales[c], size=3000) for c in crits}
    shift_test = {c: rng.normal(0, 2.2 * scales[c], size=3000) for c in crits}  # distribution shift

    sigma, q, Q = fit_conformal(cal, alpha)
    m_clean, j_clean = coverage(clean_test, sigma, q, Q)
    m_shift, j_shift = coverage(shift_test, sigma, q, Q)

    ok = True
    print(f"target marginal coverage = {1-alpha:.2f}")
    print(f"  clean : marginal={m_clean:.3f}  joint={j_clean:.3f}")
    print(f"  shift : marginal={m_shift:.3f}  joint={j_shift:.3f}")
    # clean marginal must sit near nominal; joint must ALSO be >= ~nominal (simultaneous)
    if not (0.86 <= m_clean <= 0.95):  ok = False; print("  FAIL: clean marginal off nominal")
    if not (j_clean >= 0.86):          ok = False; print("  FAIL: clean joint below nominal")
    # shift must visibly collapse coverage (the whole point)
    if not (m_shift < 0.80):           ok = False; print("  FAIL: shift did not drop marginal")
    if not (j_shift < j_clean):        ok = False; print("  FAIL: shift did not drop joint")
    print("[self-test] PASS" if ok else "[self-test] FAIL")
    raise SystemExit(0 if ok else 1)


# --------------------------------------------------------------------- real run
def main():
    ap = argparse.ArgumentParser(description="Conformal DVH coverage vs perturbation severity")
    ap.add_argument("--self-test", action="store_true", help="validate coverage math on synthetic data")
    ap.add_argument("--alpha", type=float, default=0.1, help="miscoverage (0.1 -> 90% target)")
    ap.add_argument("--cal-frac", type=float, default=0.5, help="fraction of patients used to calibrate")
    ap.add_argument("--out", default=HERE)
    args = ap.parse_args()
    if args.self_test:
        self_test()

    base = load_condition("baseline")
    if base is None:
        raise SystemExit(
            "baseline.json has no 'per_patient_dvh' field. Re-run evaluate_metrics.py with the "
            "enabler patch (persists per-patient per-criterion residuals) to produce it, then "
            "re-run this. Until then use --self-test to validate the coverage math.")

    crits = list(base)
    patients = base[crits[0]][0]
    n_cal = max(2, int(round(len(patients) * args.cal_frac)))
    cal_pts, test_pts = set(patients[:n_cal]), set(patients[n_cal:])
    print(f"{len(patients)} patients: {len(cal_pts)} calibration / {len(test_pts)} test  "
          f"target coverage {1-args.alpha:.2f}")

    def subset(cond_data, keep):
        return {c: cond_data[c][1][np.array([p in keep for p in cond_data[c][0]])] for c in cond_data}

    sigma, q, Q = fit_conformal(subset(base, cal_pts), args.alpha)

    conds = ["baseline"] + sorted(
        os.path.basename(f)[:-5] for f in glob.glob(os.path.join(METRICS, "P*_*.json")))
    rows = []
    for cond in conds:
        cd = load_condition(cond)
        if cd is None:
            continue
        m, j = coverage(subset(cd, test_pts), sigma, q, Q)
        rows.append({"condition": cond, "marginal_coverage": round(m, 4), "joint_coverage": round(j, 4)})
        print(f"  {cond:22} marginal={m:.3f}  joint={j:.3f}")

    with open(os.path.join(args.out, "coverage_vs_severity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "marginal_coverage", "joint_coverage"])
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {os.path.join(args.out, 'coverage_vs_severity.csv')}")


if __name__ == "__main__":
    main()
