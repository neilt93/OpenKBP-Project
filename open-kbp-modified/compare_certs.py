#!/usr/bin/env python
"""SmoothAdv vs baseline certified-interval comparison.

Reduction: the DVH-metric widths (D_95|PTV70, mean|Brainstem) are recomputed here by
meaning per_patient[i].radii[R].dvh_interval_widths_gy across patients (certify_summary.json's
`aggregate` block holds only voxel-wise stats, not per-metric DVH widths). The frac<=1Gy column,
by contrast, is read straight from each cert's stored `aggregate.mean_frac_within_tol`. Both
SIDES use the same code for a given column -- baseline and smoothadv go through dvh_means() and
the same aggregate lookup -- so the comparison is apples-to-apples, but the two COLUMNS come from
different sources (recomputed vs stored), which is why this note exists.

CAVEAT: the "baseline" here is the ORIGINAL committed model, trained on the tf-lightweight aug
path, while the smoothadv arms ran the numpy geometric path. That pipeline difference is the
confound that inflates these deltas (see reports/certified_robustness/FINDINGS.md). For a clean
noise-training test compare the smoothadv arm against its MATCHED control
(certify_control_s{sigma}), not against this baseline.

Run `python compare_certs.py --self-test` to validate the reduction on committed data (no GPU).
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
BASE = REPO.parent / "reports" / "certified_robustness" / "experiment_results"
# A certificate is read from the live run directory if present, else from the
# committed copy under reports/ -- sigma=0.02 only exists in the latter.
NEW_DIRS = [REPO, BASE]
METRICS = ["D_95|PTV70", "mean|Brainstem"]


def find_new(sigma):
    for d in NEW_DIRS:
        p = d / f"certify_smoothadv_s{sigma}" / "certify_summary.json"
        if p.exists():
            return p
    return None


def load(p):
    with open(p) as f:
        return json.load(f)


def dvh_means(doc, radius_key):
    """Mean DVH interval width (Gy) across patients, per metric."""
    out = {}
    for m in METRICS:
        vals = []
        for rec in doc["per_patient"]:
            r = rec["radii"].get(radius_key)
            if r and m in r.get("dvh_interval_widths_gy", {}):
                vals.append(r["dvh_interval_widths_gy"][m])
        out[m] = sum(vals) / len(vals) if vals else None
    return out


def fmt(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def pct(new, base):
    if new is None or base is None or base == 0:
        return "n/a"
    d = (new - base) / base * 100.0
    return f"{d:+.1f}%"


def self_test():
    """Validate the reduction pipeline on committed certificates -- no GPU, no live run.

    Loads the committed baseline + smoothadv certs for each sigma, recomputes DVH widths and
    reads frac at the first (informative) radius, and asserts every number is finite and the
    widths are positive. Exits non-zero on any failure so it can gate CI / a pre-run check.
    """
    ok = True
    for sigma in ["0.02", "0.05", "0.10"]:
        bpath = BASE / f"certify_s{sigma}_full" / "certify_summary.json"
        npath = find_new(sigma)
        if not bpath.exists():
            print(f"[self-test] MISSING baseline cert for sigma={sigma}: {bpath}"); ok = False; continue
        if npath is None:
            print(f"[self-test] MISSING smoothadv cert for sigma={sigma}"); ok = False; continue
        b, n = load(bpath), load(npath)
        keys = [k for k in b["aggregate"] if k in n["aggregate"]]
        if not keys:
            print(f"[self-test] sigma={sigma}: no shared radius keys"); ok = False; continue
        k0 = keys[0]
        for tag, doc in (("baseline", b), ("smoothadv", n)):
            dv = dvh_means(doc, k0)
            for m in METRICS:
                w = dv[m]
                if w is None or not (w == w) or w <= 0:  # None / NaN / non-positive
                    print(f"[self-test] sigma={sigma} {tag} radius={k0} {m}: bad width {w}"); ok = False
            frac = doc["aggregate"][k0].get("mean_frac_within_tol")
            if frac is None or not (0.0 <= frac <= 1.0):
                print(f"[self-test] sigma={sigma} {tag} radius={k0}: bad frac {frac}"); ok = False
        if ok:
            print(f"[self-test] sigma={sigma}: OK (radius={k0}, widths+frac finite)")
    print("[self-test] PASS" if ok else "[self-test] FAIL")
    sys.exit(0 if ok else 1)


if "--self-test" in sys.argv:
    self_test()

for sigma in [a for a in sys.argv[1:] if not a.startswith("-")] or ["0.02", "0.05", "0.10"]:
    bpath = BASE / f"certify_s{sigma}_full" / "certify_summary.json"
    npath = find_new(sigma)
    if npath is None:
        print(f"\n### sigma={sigma}: SmoothAdv certificate not present yet\n")
        continue
    b, n = load(bpath), load(npath)

    print(f"\n{'='*100}")
    print(f"sigma={sigma}   baseline n_pat={b['n_patients']} n_samples={b['n_samples']} "
          f"alpha={b['alpha']} tol={b['tol_gy']}   |   smoothadv n_pat={n['n_patients']} "
          f"n_samples={n['n_samples']} alpha={n['alpha']} tol={n['tol_gy']}")
    print(f"{'='*100}")

    keys = [k for k in b["aggregate"] if k in n["aggregate"]]
    hdr = (f"{'radius':>8} | {'D95 PTV70 width (Gy)':^34} | "
           f"{'mean Brainstem width (Gy)':^34} | {'frac voxels <=1Gy':^30}")
    print(hdr)
    print(f"{'':>8} | {'base':>10} {'smoothadv':>10} {'delta':>10} | "
          f"{'base':>10} {'smoothadv':>10} {'delta':>10} | {'base':>9} {'smoothadv':>9} {'delta':>9}")
    print("-" * len(hdr))

    for k in keys:
        bd, nd_ = dvh_means(b, k), dvh_means(n, k)
        ba, na = b["aggregate"][k], n["aggregate"][k]
        cert = "" if ba.get("all_certified") else "  (vacuous)"
        print(f"{k:>8} | {fmt(bd[METRICS[0]]):>10} {fmt(nd_[METRICS[0]]):>10} "
              f"{pct(nd_[METRICS[0]], bd[METRICS[0]]):>10} | "
              f"{fmt(bd[METRICS[1]]):>10} {fmt(nd_[METRICS[1]]):>10} "
              f"{pct(nd_[METRICS[1]], bd[METRICS[1]]):>10} | "
              f"{fmt(ba['mean_frac_within_tol'],4):>9} {fmt(na['mean_frac_within_tol'],4):>9} "
              f"{pct(na['mean_frac_within_tol'], ba['mean_frac_within_tol']):>9}{cert}")
    print("\nnegative delta on WIDTH = tighter certificate = better")
    print("positive delta on frac<=1Gy = more voxels certified = better")
