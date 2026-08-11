#!/usr/bin/env python3
"""Independent re-reduction of both certificate sides, purely from per_patient.

Deliberately does NOT read the stored `aggregate` block for anything, so it is a
genuine check on compare_certs.py rather than a restatement of it.
"""
import json
from pathlib import Path

ER = Path("/workspace/openkbp/reports/certified_robustness/experiment_results")
METRICS = ["D_95|PTV70", "mean|Brainstem"]


def load(p):
    with open(p) as f:
        return json.load(f)


def reduce_all(doc):
    acc = {}
    for pat in doc["per_patient"]:
        for rkey, rec in pat["radii"].items():
            slot = acc.setdefault(rkey, {m: [] for m in METRICS} | {"frac": []})
            dvh = rec.get("dvh_interval_widths_gy", {})
            for m in METRICS:
                if m in dvh:
                    slot[m].append(dvh[m])
            if "frac_within_tol" in rec:
                slot["frac"].append(rec["frac_within_tol"])
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    return {k: {"d95": mean(v[METRICS[0]]), "bs": mean(v[METRICS[1]]),
                "frac": mean(v["frac"]), "n": len(v["frac"])} for k, v in acc.items()}


b = load(ER / "certify_s0.02_full" / "certify_summary.json")
n = load(ER / "certify_smoothadv_s0.02" / "certify_summary.json")

print(f"baseline : n_pat={b['n_patients']} n_samples={b['n_samples']} "
      f"alpha={b['alpha']} tol={b['tol_gy']}")
print(f"smoothadv: n_pat={n['n_patients']} n_samples={n['n_samples']} "
      f"alpha={n['alpha']} tol={n['tol_gy']}")

B, N = reduce_all(b), reduce_all(n)

print(f"\n{'R':>7} {'npat':>5} | {'D95 base':>9} {'D95 sadv':>9} {'d%':>8} |"
      f" {'BS base':>9} {'BS sadv':>9} {'d%':>8} | {'frac base':>9} {'frac sadv':>9} {'d%':>8}")
print("-" * 104)
for k in sorted(set(B) & set(N), key=float):
    x, y = B[k], N[k]
    p = lambda a, c: f"{(a - c) / c * 100:+.1f}%" if c else "n/a"
    print(f"{k:>7} {y['n']:>5} | {x['d95']:>9.4f} {y['d95']:>9.4f} {p(y['d95'], x['d95']):>8} |"
          f" {x['bs']:>9.4f} {y['bs']:>9.4f} {p(y['bs'], x['bs']):>8} |"
          f" {x['frac']:>9.4f} {y['frac']:>9.4f} {p(y['frac'], x['frac']):>8}")

# cross-check: does per_patient reduction reproduce each side's stored aggregate?
print("\nper_patient frac vs stored aggregate.mean_frac_within_tol (must match):")
for tag, doc, red in (("base", b, B), ("sadv", n, N)):
    for k in sorted(red, key=float):
        stored = doc["aggregate"].get(k, {}).get("mean_frac_within_tol")
        if stored is not None:
            d = abs(stored - red[k]["frac"])
            print(f"  {tag} R={k:>6}: stored={stored:.6f} recomputed={red[k]['frac']:.6f} "
                  f"diff={d:.2e} {'OK' if d < 1e-9 else '** MISMATCH **'}")
