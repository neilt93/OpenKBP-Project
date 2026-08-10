#!/usr/bin/env python3
"""SmoothAdv (noise-trained) vs baseline certificates, per sigma and radius.

Both sides are reduced with IDENTICAL code from `per_patient`, rather than reading
one side's stored `aggregate` block, so the columns are guaranteed comparable.

DVH interval widths live at per_patient[i]["radii"][R]["dvh_interval_widths_gy"];
frac-voxels-within-tolerance at per_patient[i]["radii"][R]["frac_within_tol"].
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
BASELINE_DIR = REPO.parent / "reports" / "certified_robustness" / "experiment_results"

# the DVH metrics the brief singles out
METRICS = ["D_95|PTV70", "mean|Brainstem"]


def load(p):
    with open(p) as fh:
        return json.load(fh)


def reduce_summary(doc):
    """-> {radius_float: {"d95": mean, "brainstem": mean, "frac": mean, "n": count}}"""
    acc = {}
    for pat in doc.get("per_patient", []):
        for rkey, rec in pat.get("radii", {}).items():
            r = float(rkey)
            slot = acc.setdefault(r, {m: [] for m in METRICS} | {"frac": []})
            dvh = rec.get("dvh_interval_widths_gy", {})
            for m in METRICS:
                if m in dvh:
                    slot[m].append(dvh[m])
            if "frac_within_tol" in rec:
                slot["frac"].append(rec["frac_within_tol"])
    out = {}
    for r, slot in acc.items():
        mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
        out[r] = {
            "d95": mean(slot["D_95|PTV70"]),
            "brainstem": mean(slot["mean|Brainstem"]),
            "frac": mean(slot["frac"]),
            "n": max(len(slot["D_95|PTV70"]), len(slot["frac"])),
        }
    return out


def fmt(x, nd=4):
    return "n/a" if x != x else f"{x:.{nd}f}"


def delta(new, base, lower_is_better=True):
    """Signed change plus a plain-language verdict."""
    if new != new or base != base:
        return "n/a", ""
    d = new - base
    if abs(base) > 1e-12:
        pct = 100.0 * d / base
        s = f"{d:+.4f} ({pct:+.1f}%)"
    else:
        s = f"{d:+.4f}"
    good = (d < 0) if lower_is_better else (d > 0)
    if abs(d) < 1e-9:
        verdict = "flat"
    else:
        verdict = "tighter" if (good and lower_is_better) else \
                  "higher" if (good and not lower_is_better) else \
                  "wider" if lower_is_better else "lower"
    return s, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", default="0.02,0.05,0.10")
    ap.add_argument("--new-dir", default=".", help="dir holding certify_smoothadv_s<sigma>/")
    ap.add_argument("--baseline-dir", default=str(BASELINE_DIR))
    ap.add_argument("--self-test", action="store_true",
                    help="compare each baseline against itself; every delta must be 0.0000")
    args = ap.parse_args()

    any_found = False
    for S in args.sigmas.split(","):
        S = S.strip()
        base_p = Path(args.baseline_dir) / f"certify_s{S}_full" / "certify_summary.json"
        if args.self_test:
            new_p = base_p
        else:
            new_p = Path(args.new_dir) / f"certify_smoothadv_s{S}" / "certify_summary.json"

        if not base_p.exists():
            print(f"[sigma={S}] baseline missing: {base_p}", file=sys.stderr)
            continue
        if not new_p.exists():
            print(f"[sigma={S}] SmoothAdv certificate not present yet ({new_p}) — skipping")
            continue
        any_found = True

        base = reduce_summary(load(base_p))
        new = reduce_summary(load(new_p))

        bdoc, ndoc = load(base_p), load(new_p)
        print(f"\n{'='*100}")
        print(f"sigma = {S}   baseline n_pat={bdoc.get('n_patients')} n_samples={bdoc.get('n_samples')} "
              f"alpha={bdoc.get('alpha')} | smoothadv n_pat={ndoc.get('n_patients')} "
              f"n_samples={ndoc.get('n_samples')} alpha={ndoc.get('alpha')}")
        print(f"{'='*100}")
        hdr = (f"{'radius':>8} | {'metric':<16} | {'baseline':>10} | {'smoothadv':>10} | "
               f"{'delta':>20} | verdict")
        print(hdr); print("-" * len(hdr))

        for r in sorted(set(base) & set(new)):
            rows = [
                ("D95 PTV70 (Gy)", base[r]["d95"], new[r]["d95"], True),
                ("mean Brainstem", base[r]["brainstem"], new[r]["brainstem"], True),
                ("frac<=1Gy", base[r]["frac"], new[r]["frac"], False),
            ]
            for name, b, n, lower_better in rows:
                d, verdict = delta(n, b, lower_is_better=lower_better)
                print(f"{r:>8} | {name:<16} | {fmt(b):>10} | {fmt(n):>10} | {d:>20} | {verdict}")
            print("-" * len(hdr))

        only_base = sorted(set(base) - set(new))
        only_new = sorted(set(new) - set(base))
        if only_base or only_new:
            print(f"  NOTE non-overlapping radii — baseline-only: {only_base}  smoothadv-only: {only_new}")

    if not any_found:
        print("\nNo SmoothAdv certificates found yet.")


if __name__ == "__main__":
    main()
