"""Pure scoring / bookkeeping for the adversarial-defence sweep (NO TensorFlow).

Split out from `adversarial_defense.py` so the parts most likely to be wrong, the
in-memory DVH scoring and the headline-number arithmetic (clean cost, fraction of
attack damage recovered), are unit-testable off-GPU before any paid run. The TF
orchestration (model, attacks, batch loop) stays in `adversarial_defense.py`.
"""
from __future__ import annotations

import numpy as np

# Per-defence strength grids. Units differ by defence:
#   smooth    -> Gaussian sigma in voxels
#   noise     -> additive-noise std in normalised CT units (x4095 ~ HU)
#   intensity -> max relative intensity jitter (+/-)
#   none/flip -> no strength
DEFAULT_STRENGTHS = {
    "none": [None],
    "smooth": [0.5, 1.0, 2.0],
    "noise": [0.02, 0.05, 0.10],
    "intensity": [0.05, 0.10, 0.20],
    "flip": [None],
}


def build_defense_list(defenses: list[str], quick: bool) -> list[tuple]:
    """Expand selected defences into (defense, strength) pairs. --quick keeps the
    middle strength of each multi-strength defence for a fast first pass."""
    pairs = []
    for d in defenses:
        strengths = DEFAULT_STRENGTHS[d]
        if quick and len(strengths) > 1:
            strengths = [strengths[len(strengths) // 2]]
        for s in strengths:
            pairs.append((d, s))
    return pairs


def cond_key(attack, eps, defense, strength) -> str:
    return f"{attack}|eps{eps:g}|{defense}|{'' if strength is None else f'{strength:g}'}"


def score_prediction(evaluator, batch, ref_dose_gy, pred_dose_gy) -> None:
    """Feed one patient's reference/prediction into a per-condition DoseEvaluator,
    reusing its official metric math in-memory (no prediction CSVs on disk).

    `evaluator` is a DoseEvaluator; `batch` a DataBatch (training_model mode);
    doses are flat arrays in Gy.
    """
    evaluator.reference_batch = batch
    pid = batch.patient_list[0]
    evaluator.reference_dvh_metrics_df = evaluator._calculate_dvh_metrics(
        evaluator.reference_dvh_metrics_df, ref_dose_gy)
    evaluator.prediction_dvh_metrics_df = evaluator._calculate_dvh_metrics(
        evaluator.prediction_dvh_metrics_df, pred_dose_gy)
    evaluator.dose_errors[pid] = (
        np.sum(np.abs(ref_dose_gy - pred_dose_gy)) / np.sum(batch.possible_dose_mask))


def add_derived(records: list[dict]) -> list[dict]:
    """Annotate each record with clean cost and (under attack) the fraction of
    attack damage recovered, relative to the clean-undefended and same-attack-
    undefended baselines. Mutates and returns `records`.

    clean_cost_*      : score(clean, this defence) - score(clean, undefended). >0 = the
                        defence sacrifices accuracy on clean input.
    attack_damage_*   : score(attack, undefended) - score(clean, undefended).
    remaining_damage_*: score(attack, this defence) - score(clean, undefended).
    recovered_frac_*  : 1 - remaining/damage. 1.0 = attack fully neutralised, 0 = no help,
                        negative = the defence made the attacked case worse.
    """
    def find(atk, eps, defense, strength):
        for r in records:
            if (r["attack"], r["epsilon"], r["defense"], r["strength"]) == (atk, eps, defense, strength):
                return r
        return None

    clean_none = find("clean", 0.0, "none", None)
    for r in records:
        clean_def = find("clean", 0.0, r["defense"], r["strength"])
        if clean_none and clean_def:
            r["clean_cost_dvh"] = clean_def["dvh_score"] - clean_none["dvh_score"]
            r["clean_cost_dose"] = clean_def["dose_score"] - clean_none["dose_score"]
        if r["attack"] != "clean" and clean_none:
            undef = find(r["attack"], r["epsilon"], "none", None)
            if undef:
                for m in ("dvh", "dose"):
                    dmg = undef[f"{m}_score"] - clean_none[f"{m}_score"]
                    rem = r[f"{m}_score"] - clean_none[f"{m}_score"]
                    r[f"attack_damage_{m}"] = dmg
                    r[f"remaining_damage_{m}"] = rem
                    r[f"recovered_frac_{m}"] = (1.0 - rem / dmg) if abs(dmg) > 1e-9 else None
    return records
