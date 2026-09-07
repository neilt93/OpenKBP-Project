#!/usr/bin/env python3
"""Build ASTRO #79011 poster panels from the committed CT-perturbation sweep.

Reads the per-patient metric JSONs (baseline + P1-P5 x L*), applies ONE threshold
criterion, and emits:
  - panel4_threshold_table.{csv,md}  : the centerpiece threshold table
  - fig_panel5_maxcrit_gy.png        : max per-criterion DVH shift (Gy) vs severity,
                                       with the 1.0 Gy visibility line (the money figure)
  - fig_panel5_dvh_pct.png           : aggregate DVH-score degradation (%) vs severity

THRESHOLD CRITERION (Panel 2, stated once, used everywhere):
  A severity level is "clinically visible" when the cohort-mean change in ANY standard
  DVH criterion, relative to each patient's own baseline prediction, exceeds 1.0 Gy.
  (Corroborated by per-patient dose-MAE consistency; reported but not the gate.)
"""
import csv
import json
import glob
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "open-kbp-modified", "openkbp_hn_robustness", "metrics", "per_patient"))
THRESH_GY = 1.0

FAMILIES = {
    "P1_noise":      {"label": "P1 noise",        "levels": [1, 2, 3, 4, 5], "color": "#2980b9"},
    "P2_bone_shift": {"label": "P2 HU shift",     "levels": [1, 2, 3, 4, 5], "color": "#e67e22"},
    "P3_bias_field": {"label": "P3 bias field",   "levels": [1, 2, 3, 4, 5], "color": "#27ae60"},
    "P4_resolution": {"label": "P4 resolution",   "levels": [0, 1, 2, 3, 4], "color": "#c0392b"},
    "P5_dental":     {"label": "P5 dental streak","levels": [1, 2, 3, 4, 5], "color": "#8e44ad"},
}
# Physical parameters per level (from the report's perturbation table).
PHYS = {
    "P1_noise":      {1: "8/12 HU", 2: "15/25", 3: "30/50", 4: "60/100", 5: "100/160 HU"},
    "P2_bone_shift": {1: "5/50 HU", 2: "10/100", 3: "25/250", 4: "50/500", 5: "100/1000 HU"},
    "P3_bias_field": {1: "10 HU", 2: "20", 3: "50", 4: "100", 5: "200 HU"},
    "P4_resolution": {0: "0.5/0.25 vox", 1: "1.0/0.5", 2: "2.0/1.0", 3: "3.0/1.5", 4: "4.0/2.0 vox"},
    "P5_dental":     {1: "150HU/8", 2: "300/12", 3: "500/16", 4: "800/20", 5: "1200/24"},
}
CLINICAL = {
    "P1_noise":      "typical scanner noise ≈10–50 HU (≈L1–L2)",
    "P2_bone_shift": "inter-scanner calibration drift ≈10–50 HU (≈L2)",
    "P3_bias_field": "RF/scatter non-uniformity, tens of HU",
    "P4_resolution": "cross-scanner slice/kernel variation ≈1–3 vox (≈L1–L3)",
    "P5_dental":     "dental streaks common in H&N (L2–L4 realistic)",
}

def load(cond_file):
    with open(cond_file) as f:
        return json.load(f)

base = load(os.path.join(METRICS, "baseline.json"))
bstruct, bdose = base["per_structure"], base["per_patient_dose"]
base_dvh = base["dvh_score"]

def cond_stats(fam, lvl):
    d = load(os.path.join(METRICS, f"{fam}_L{lvl}.json"))
    s = d["per_structure"]
    shifts = {k: s[k] - bstruct[k] for k in s if k in bstruct}
    max_k = max(shifts, key=lambda k: abs(shifts[k]))
    dvh_pct = (d["dvh_score"] - base_dvh) / base_dvh * 100.0
    pd = d["per_patient_dose"]
    ddose = [pd[k] - bdose[k] for k in pd if k in bdose]
    frac05 = sum(1 for x in ddose if x > 0.5) / len(ddose)
    return {
        "max_shift_gy": shifts[max_k], "max_crit": max_k,
        "dvh_pct": dvh_pct, "frac_pts_dose_over_0.5gy": frac05,
    }

# --- compute + threshold table -------------------------------------------------
rows, curves = [], {}
for fam, meta in FAMILIES.items():
    lvls = meta["levels"]
    stats = {lvl: cond_stats(fam, lvl) for lvl in lvls}
    curves[fam] = (lvls, stats)
    visible_lvl = next((lvl for lvl in lvls if abs(stats[lvl]["max_shift_gy"]) > THRESH_GY), None)
    lo, hi = lvls[0], lvls[-1]
    sweep = f"L{lo} ({PHYS[fam][lo]}) → L{hi} ({PHYS[fam][hi]})"
    if visible_lvl is None:
        visible = f"never in tested range (max {max(abs(stats[l]['max_shift_gy']) for l in lvls):.2f} Gy)"
        verdict = "Robust"
    else:
        s = stats[visible_lvl]
        visible = f"L{visible_lvl} ({PHYS[fam][visible_lvl]}): {s['max_shift_gy']:.2f} Gy on {s['max_crit']}"
        verdict = "SENSITIVE — the failure mode"
    rows.append({
        "family": meta["label"], "sweep_range": sweep,
        "clinically_visible_at": visible, "vs_clinical_range": CLINICAL[fam],
        "verdict": verdict,
    })

os.makedirs(HERE, exist_ok=True)
with open(os.path.join(HERE, "panel4_threshold_table.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

with open(os.path.join(HERE, "panel4_threshold_table.md"), "w") as f:
    f.write("# Panel 4 — Threshold table (criterion: mean shift on any DVH criterion > 1.0 Gy)\n\n")
    f.write("| Family | Sweep range tested | Clinically visible at | vs. clinical range | Verdict |\n")
    f.write("|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r['family']} | {r['sweep_range']} | {r['clinically_visible_at']} | "
                f"{r['vs_clinical_range']} | {r['verdict']} |\n")

# --- Panel 5a: max per-criterion shift (Gy) with the 1 Gy line (money figure) ---
plt.figure(figsize=(7, 4.3))
for fam, meta in FAMILIES.items():
    lvls, stats = curves[fam]
    x = lvls
    y = [abs(stats[l]["max_shift_gy"]) for l in lvls]
    red = fam == "P4_resolution"
    plt.plot(x, y, marker="o", lw=2.6 if red else 1.6,
             color=meta["color"], zorder=3 if red else 2,
             label=meta["label"])
plt.axhline(THRESH_GY, ls="--", color="k", lw=1.2)
plt.text(4.05, THRESH_GY + 0.05, "clinically visible (1.0 Gy)", ha="right", va="bottom", fontsize=9)
plt.xlabel("Severity level"); plt.ylabel("Max per-criterion DVH shift from baseline (Gy)")
plt.title("CT perturbation severity vs worst-case DVH-criterion error")
plt.xticks([0, 1, 2, 3, 4, 5]); plt.legend(fontsize=9, framealpha=0.9)
plt.grid(alpha=0.25); plt.tight_layout()
plt.savefig(os.path.join(HERE, "fig_panel5_maxcrit_gy.png"), dpi=200)
plt.close()

# --- Panel 5b: aggregate DVH-score degradation (%) -----------------------------
plt.figure(figsize=(7, 4.3))
for fam, meta in FAMILIES.items():
    lvls, stats = curves[fam]
    red = fam == "P4_resolution"
    plt.plot(lvls, [stats[l]["dvh_pct"] for l in lvls], marker="o",
             lw=2.6 if red else 1.6, color=meta["color"],
             zorder=3 if red else 2, label=meta["label"])
plt.xlabel("Severity level"); plt.ylabel("DVH-score degradation vs baseline (%)")
plt.title("Aggregate DVH degradation across perturbation families")
plt.xticks([0, 1, 2, 3, 4, 5]); plt.legend(fontsize=9, framealpha=0.9)
plt.grid(alpha=0.25); plt.tight_layout()
plt.savefig(os.path.join(HERE, "fig_panel5_dvh_pct.png"), dpi=200)
plt.close()

# --- console summary -----------------------------------------------------------
print("Threshold criterion: mean shift on any DVH criterion > 1.0 Gy\n")
print(f"{'family':16} {'visible at':10} {'max shift (Gy)':>15} {'sentinel criterion':24}")
for fam, meta in FAMILIES.items():
    lvls, stats = curves[fam]
    vis = next((l for l in lvls if abs(stats[l]['max_shift_gy']) > THRESH_GY), None)
    mx = max(lvls, key=lambda l: abs(stats[l]['max_shift_gy']))
    print(f"{meta['label']:16} {('L'+str(vis)) if vis is not None else 'never':10} "
          f"{stats[mx]['max_shift_gy']:>15.2f} {stats[mx]['max_crit']:24}")
print("\nWrote: panel4_threshold_table.{csv,md}, fig_panel5_maxcrit_gy.png, fig_panel5_dvh_pct.png")
