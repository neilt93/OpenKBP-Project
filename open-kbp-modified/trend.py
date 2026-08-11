#!/usr/bin/env python3
"""Across-sigma trend for the three SmoothAdv models + vs-baseline deltas.

Everything is re-reduced from per_patient, independent of compare_certs.py.
The three radii grids are sigma-scaled (0.5x, 1.0x, 1.4x sigma), so the arms are
compared at matched R/sigma rather than matched absolute R.
"""
import json
from pathlib import Path

ER = Path("/workspace/openkbp/reports/certified_robustness/experiment_results")
NEW = Path("/workspace/openkbp/open-kbp-modified")
M = ["D_95|PTV70", "mean|Brainstem"]

GRID = {  # sigma -> [(R/sigma label, radius key)]
    "0.02": [("0.5", "0.01"), ("1.0", "0.02"), ("1.4", "0.028")],
    "0.05": [("0.5", "0.025"), ("1.0", "0.05"), ("1.4", "0.07")],
    "0.10": [("0.5", "0.05"), ("1.0", "0.1"), ("1.4", "0.14")],
}


def red(doc):
    acc = {}
    for p in doc["per_patient"]:
        for k, r in p["radii"].items():
            s = acc.setdefault(k, {m: [] for m in M} | {"frac": []})
            for m in M:
                if m in r.get("dvh_interval_widths_gy", {}):
                    s[m].append(r["dvh_interval_widths_gy"][m])
            if "frac_within_tol" in r:
                s["frac"].append(r["frac_within_tol"])
    f = lambda x: sum(x) / len(x) if x else float("nan")
    return {k: {"d95": f(v[M[0]]), "bs": f(v[M[1]]), "frac": f(v["frac"]),
                "n": len(v["frac"])} for k, v in acc.items()}


def sadv_path(s):
    p = NEW / f"certify_smoothadv_s{s}" / "certify_summary.json"
    return p if p.exists() else ER / f"certify_smoothadv_s{s}" / "certify_summary.json"


S = {s: json.load(open(sadv_path(s))) for s in GRID}
B = {s: json.load(open(ER / f"certify_s{s}_full" / "certify_summary.json")) for s in GRID}
RS, RB = {s: red(d) for s, d in S.items()}, {s: red(d) for s, d in B.items()}

print("PROTOCOL CHECK (all must match across arms)")
for s in GRID:
    a, b = S[s], B[s]
    print(f"  sigma={s}: sadv n_pat={a['n_patients']} n={a['n_samples']} "
          f"alpha={a['alpha']} tol={a['tol_gy']} | base n_pat={b['n_patients']} "
          f"n={b['n_samples']} alpha={b['alpha']} tol={b['tol_gy']}")

print("\n\n=== PRIMARY: across-sigma trend, SmoothAdv models only (clean) ===")
print("absolute certified widths in Gy, at matched R/sigma\n")
print(f"{'R/sigma':>8} {'sigma':>6} {'R abs':>7} | {'D95 PTV70':>10} {'Brainstem':>10} "
      f"{'frac<=1Gy':>10} | {'vox med':>8} {'vox p95':>8} {'vox max':>8}")
print("-" * 92)
for lab in ["0.5", "1.0", "1.4"]:
    for s in ["0.02", "0.05", "0.10"]:
        rk = dict(GRID[s])[lab]
        v, agg = RS[s][rk], S[s]["aggregate"][rk]
        print(f"{lab:>8} {s:>6} {rk:>7} | {v['d95']:>10.4f} {v['bs']:>10.4f} "
              f"{v['frac']:>10.4f} | {agg['mean_median_width_gy']:>8.4f} "
              f"{agg['mean_p95_width_gy']:>8.4f} {agg['mean_max_width_gy']:>8.4f}")
    print()

print("\n=== SECONDARY: SmoothAdv vs baseline, delta by sigma (confounded) ===")
print("negative = tighter = better for widths; positive = better for frac\n")
print(f"{'R/sigma':>8} {'sigma':>6} | {'D95 delta':>12} {'Brainstem delta':>16} {'frac delta':>12}")
print("-" * 60)
for lab in ["0.5", "1.0", "1.4"]:
    for s in ["0.02", "0.05", "0.10"]:
        rk = dict(GRID[s])[lab]
        n_, b_ = RS[s][rk], RB[s][rk]
        p = lambda a, c: (a - c) / c * 100
        print(f"{lab:>8} {s:>6} | {p(n_['d95'], b_['d95']):>11.1f}% "
              f"{p(n_['bs'], b_['bs']):>15.1f}% {p(n_['frac'], b_['frac']):>11.1f}%")
    print()
