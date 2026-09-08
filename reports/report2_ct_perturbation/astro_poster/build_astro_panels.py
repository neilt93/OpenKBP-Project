#!/usr/bin/env python3
"""Build ASTRO #79011 poster panels from the committed CT-perturbation sweep.

Reads the per-patient metric JSONs (baseline + P1-P5 x L*) via astro_common and emits:
  - panel4_threshold_table.{csv,md}  : the centerpiece threshold table
  - fig_panel5_maxcrit_gy.png        : max per-criterion DVH shift (Gy) vs severity,
                                       with the 1.0 Gy visibility line (the money figure)
  - fig_panel5_dvh_pct.png           : aggregate DVH-score degradation (%) vs severity

Config, stats, and the threshold criterion live in astro_common (shared with build_poster_pdf).
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import astro_common as C

HERE = C.HERE
rows = [C.threshold_row(fam) for fam in C.FAMILIES]

# --- Panel 4 threshold table ---------------------------------------------------
with open(os.path.join(HERE, "panel4_threshold_table.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["family", "sweep_range", "clinically_visible_at", "vs_clinical_range", "verdict"])
    for r in rows:
        w.writerow([r["label"], r["sweep_full"], r["visible"], r["clinical"], r["verdict"]])

with open(os.path.join(HERE, "panel4_threshold_table.md"), "w") as f:
    f.write("# Panel 4 — Threshold table (criterion: mean shift on any DVH criterion > 1.0 Gy)\n\n")
    f.write("| Family | Sweep range tested | Clinically visible at | vs. clinical range | Verdict |\n")
    f.write("|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r['label']} | {r['sweep_full']} | {r['visible']} | "
                f"{r['clinical']} | {r['verdict']} |\n")


def _plot(yfun, ylabel, title, fname, hline=None, hlabel=None):
    plt.figure(figsize=(7, 4.3))
    for r in rows:
        red = r["fam"] == "P4_resolution"
        plt.plot(r["levels"], [yfun(r["stats"][l]) for l in r["levels"]], marker="o",
                 lw=2.6 if red else 1.6, color=C.PALETTE[r["fam"]],
                 zorder=3 if red else 2, label=r["label"])
    if hline is not None:
        plt.axhline(hline, ls="--", color="k", lw=1.2)
        plt.text(4.05, hline + 0.05, hlabel, ha="right", va="bottom", fontsize=9)
    plt.xlabel("Severity level"); plt.ylabel(ylabel); plt.title(title)
    plt.xticks([0, 1, 2, 3, 4, 5]); plt.legend(fontsize=9, framealpha=0.9)
    plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(os.path.join(HERE, fname), dpi=200); plt.close()


# Panel 5a: max per-criterion shift (Gy) with the 1 Gy line (the money figure)
_plot(lambda s: abs(s["max_shift_gy"]), "Max per-criterion DVH shift from baseline (Gy)",
      "CT perturbation severity vs worst-case DVH-criterion error", "fig_panel5_maxcrit_gy.png",
      hline=C.THRESH_GY, hlabel="clinically visible (1.0 Gy)")
# Panel 5b: aggregate DVH-score degradation (%)
_plot(lambda s: s["dvh_pct"], "DVH-score degradation vs baseline (%)",
      "Aggregate DVH degradation across perturbation families", "fig_panel5_dvh_pct.png")

# --- console summary -----------------------------------------------------------
print("Threshold criterion: mean shift on any DVH criterion > 1.0 Gy\n")
print(f"{'family':16} {'visible at':10} {'max shift (Gy)':>15} {'sentinel criterion':24}")
for r in rows:
    mx = max(r["levels"], key=lambda l: abs(r["stats"][l]["max_shift_gy"]))
    vis = f"L{r['visible_level']}" if r["visible_level"] is not None else "never"
    print(f"{r['label']:16} {vis:10} {r['stats'][mx]['max_shift_gy']:>15.2f} "
          f"{r['stats'][mx]['max_crit']:24}")
print("\nWrote: panel4_threshold_table.{csv,md}, fig_panel5_maxcrit_gy.png, fig_panel5_dvh_pct.png")
