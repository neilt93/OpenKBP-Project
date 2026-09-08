#!/usr/bin/env python3
"""Coverage-vs-severity figure + threshold table from conformal_dvh.py output.

Consumes coverage_vs_severity.csv (condition, marginal_coverage, joint_coverage). Produces:
  - fig_coverage_vs_severity.png : marginal coverage per family vs severity, with the 90%
                                   nominal line and an 80% "guarantee broken" line.
  - coverage_threshold_table.{md,csv} : severity at which coverage first drops below 80%, per family.

If the CSV is absent it renders a clearly-labelled DEMO from synthetic coverage (P4 collapsing)
so the figure shape can be checked before the pod run. Real data overwrites the DEMO.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "coverage_vs_severity.csv")
NOMINAL, BROKEN = 0.90, 0.80

FAMS = [
    ("P1_noise", "P1 noise", "#2980b9"),
    ("P2_bone_shift", "P2 HU shift", "#e67e22"),
    ("P3_bias_field", "P3 bias field", "#27ae60"),
    ("P4_resolution", "P4 resolution", "#c0392b"),
    ("P5_dental", "P5 dental streak", "#8e44ad"),
]


def load_or_demo():
    if os.path.exists(CSV):
        rows = list(csv.DictReader(open(CSV)))
        return {r["condition"]: float(r["marginal_coverage"]) for r in rows}, False
    # DEMO: clean ~0.90, P4 collapses, others hold. Illustrative only.
    demo = {"baseline": 0.90}
    prof = {"P1_noise": [.90, .89, .89, .88, .87], "P2_bone_shift": [.90, .89, .88, .84, .70],
            "P3_bias_field": [.90, .90, .89, .89, .88], "P4_resolution": [.88, .78, .64, .50, .38],
            "P5_dental": [.90, .90, .89, .89, .89]}
    for fam, ys in prof.items():
        lvls = range(0, 5) if fam == "P4_resolution" else range(1, 6)
        for lvl, y in zip(lvls, ys):
            demo[f"{fam}/L{lvl}"] = y
    return demo, True


def levels_for(fam):
    return [0, 1, 2, 3, 4] if fam == "P4_resolution" else [1, 2, 3, 4, 5]


def main():
    cov, is_demo = load_or_demo()
    base = cov.get("baseline", NOMINAL)

    # figure
    plt.figure(figsize=(7.5, 4.5))
    rows = []
    for fam, label, color in FAMS:
        lvls = levels_for(fam)
        xs, ys = [], []
        for lvl in lvls:
            key = f"{fam}/L{lvl}"
            if key in cov:
                xs.append(lvl); ys.append(cov[key])
        # anchor at baseline (x=0) for families that start at L1
        if xs and xs[0] != 0:
            xs = [0] + xs; ys = [base] + ys
        red = fam == "P4_resolution"
        plt.plot(xs, ys, marker="o", lw=2.6 if red else 1.6, color=color,
                 zorder=3 if red else 2, label=label)
        broke = next((l for l, y in zip(xs, ys) if y < BROKEN and l > 0), None)
        rows.append([label, f"L{broke}" if broke else "never (>=80% in range)",
                     "SENSITIVE" if broke else "Robust"])
    plt.axhline(NOMINAL, ls="--", color="k", lw=1.0); plt.text(4.02, NOMINAL + 0.005,
                "90% target", ha="right", fontsize=8)
    plt.axhline(BROKEN, ls=":", color="#c0392b", lw=1.0); plt.text(4.02, BROKEN + 0.005,
                "80% (guarantee broken)", ha="right", fontsize=8, color="#c0392b")
    plt.xlabel("Severity level"); plt.ylabel("Empirical marginal coverage (target 90%)")
    ttl = "Conformal DVH coverage vs CT-perturbation severity"
    plt.title(ttl + ("   [DEMO — synthetic]" if is_demo else ""))
    plt.ylim(0.3, 1.0); plt.xticks([0, 1, 2, 3, 4, 5]); plt.legend(fontsize=8, loc="lower left")
    plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(os.path.join(HERE, "fig_coverage_vs_severity.png"), dpi=200); plt.close()

    # table
    with open(os.path.join(HERE, "coverage_threshold_table.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["family", "coverage_drops_below_80pct_at", "verdict"]); w.writerows(rows)
    with open(os.path.join(HERE, "coverage_threshold_table.md"), "w") as f:
        f.write(("# Coverage threshold table" + (" (DEMO — synthetic)\n\n" if is_demo else "\n\n")))
        f.write("| Family | Coverage drops <80% at | Verdict |\n|---|---|---|\n")
        for r in rows:
            f.write(f"| {r[0]} | {r[1]} | {r[2]} |\n")
    print(("DEMO " if is_demo else "") + "wrote fig_coverage_vs_severity.png + coverage_threshold_table.{md,csv}")
    if is_demo:
        print("NOTE: synthetic placeholder. Run conformal_dvh.py on real per_patient_dvh, then re-run this.")


if __name__ == "__main__":
    main()
