"""Canonical DISPLAY taxonomy for the CT perturbation families (P1–P5).

Shared across report strands (ASTRO characterization + conformal coverage) so a family is
always the same label, colour, and severity-level set in every figure and table — e.g. P4
resolution is the same red everywhere. This is presentation only; the authoritative severity
parameters live in the perturbation classes under
open-kbp-modified/openkbp_hn_robustness/perturbations/.
"""
FAMILIES = {
    "P1_noise":      {"label": "P1 noise",         "color": "#2980b9", "levels": [1, 2, 3, 4, 5]},
    "P2_bone_shift": {"label": "P2 HU shift",      "color": "#e67e22", "levels": [1, 2, 3, 4, 5]},
    "P3_bias_field": {"label": "P3 bias field",    "color": "#27ae60", "levels": [1, 2, 3, 4, 5]},
    "P4_resolution": {"label": "P4 resolution",    "color": "#c0392b", "levels": [0, 1, 2, 3, 4]},
    "P5_dental":     {"label": "P5 dental streak", "color": "#8e44ad", "levels": [1, 2, 3, 4, 5]},
}
PALETTE = {k: v["color"] for k, v in FAMILIES.items()}


def levels_for(fam):
    return FAMILIES[fam]["levels"]
