#!/usr/bin/env python3
"""Shared config + stats for the ASTRO #79011 poster builders.

Single source of truth for the perturbation families, their physical parameters, the clinical
reference ranges, the visibility threshold, and the per-criterion / aggregate statistics read
from the committed sweep JSONs. Imported by build_astro_panels.py (table + figures) and
build_poster_pdf.py (the showable PDF) so the two never drift.

THRESHOLD CRITERION (stated once, used everywhere): a severity level is "clinically visible"
when the cohort-mean change in ANY standard DVH criterion, vs each patient's own baseline
prediction, exceeds THRESH_GY.
"""
import functools
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "open-kbp-modified", "openkbp_hn_robustness", "metrics", "per_patient"))

THRESH_GY = 1.0
GREEN, RED = "#d4efdf", "#f5b7b1"          # verdict cell fills

FAMILIES = {
    "P1_noise":      {"label": "P1 noise",         "levels": [1, 2, 3, 4, 5], "color": "#2980b9"},
    "P2_bone_shift": {"label": "P2 HU shift",      "levels": [1, 2, 3, 4, 5], "color": "#e67e22"},
    "P3_bias_field": {"label": "P3 bias field",    "levels": [1, 2, 3, 4, 5], "color": "#27ae60"},
    "P4_resolution": {"label": "P4 resolution",    "levels": [0, 1, 2, 3, 4], "color": "#c0392b"},
    "P5_dental":     {"label": "P5 dental streak", "levels": [1, 2, 3, 4, 5], "color": "#8e44ad"},
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
PALETTE = {fam: m["color"] for fam, m in FAMILIES.items()}


def load(cond_file):
    with open(os.path.join(METRICS, cond_file)) as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def _baseline():
    b = load("baseline.json")
    return b["per_structure"], b["per_patient_dose"], b["dvh_score"]


def cond_stats(fam, lvl):
    """Per-condition summary vs baseline: worst per-criterion DVH shift (Gy) + which criterion,
    aggregate DVH-score degradation (%), and per-patient dose-MAE consistency."""
    bstruct, bdose, base_dvh = _baseline()
    d = load(f"{fam}_L{lvl}.json")
    s = d["per_structure"]
    shifts = {k: s[k] - bstruct[k] for k in s if k in bstruct}
    max_k = max(shifts, key=lambda k: abs(shifts[k]))
    pd = d["per_patient_dose"]
    ddose = [pd[k] - bdose[k] for k in pd if k in bdose]
    return {
        "max_shift_gy": shifts[max_k], "max_crit": max_k,
        "dvh_pct": (d["dvh_score"] - base_dvh) / base_dvh * 100.0,
        "frac_pts_dose_over_0.5gy": sum(1 for x in ddose if x > 0.5) / len(ddose),
    }


def family_curve(fam):
    """(levels, {level: cond_stats}) for one family."""
    lvls = FAMILIES[fam]["levels"]
    return lvls, {lvl: cond_stats(fam, lvl) for lvl in lvls}


def sweep_range(fam, short=False):
    lvls = FAMILIES[fam]["levels"]
    lo, hi = lvls[0], lvls[-1]
    if short:
        return f"{PHYS[fam][lo]} → {PHYS[fam][hi]}"
    return f"L{lo} ({PHYS[fam][lo]}) → L{hi} ({PHYS[fam][hi]})"


def threshold_row(fam):
    """Canonical threshold-table row for one family (shared by the md/csv table and the PDF)."""
    lvls, stats = family_curve(fam)
    vis_lvl = next((l for l in lvls if abs(stats[l]["max_shift_gy"]) > THRESH_GY), None)
    if vis_lvl is None:
        maxabs = max(abs(stats[l]["max_shift_gy"]) for l in lvls)
        visible = f"never in tested range (max {maxabs:.2f} Gy)"
        visible_short = f"never (max {maxabs:.2f} Gy)"
        verdict, sensitive = "Robust", False
    else:
        s = stats[vis_lvl]
        crit = s["max_crit"]
        visible = f"L{vis_lvl} ({PHYS[fam][vis_lvl]}): {s['max_shift_gy']:.2f} Gy on {crit}"
        visible_short = f"L{vis_lvl}: {s['max_shift_gy']:.2f} Gy ({crit.replace('_', ' ')})"
        verdict, sensitive = "SENSITIVE — the failure mode", True
    return {
        "fam": fam, "label": FAMILIES[fam]["label"], "levels": lvls, "stats": stats,
        "visible_level": vis_lvl, "visible": visible, "visible_short": visible_short,
        "verdict": verdict, "verdict_short": "SENSITIVE" if sensitive else "Robust",
        "sensitive": sensitive, "cell_color": RED if sensitive else GREEN,
        "sweep_full": sweep_range(fam), "sweep_short": sweep_range(fam, short=True),
        "clinical": CLINICAL[fam],
    }
