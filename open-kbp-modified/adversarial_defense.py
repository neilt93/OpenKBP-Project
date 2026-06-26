#!/usr/bin/env python3
"""Test-time augmentation as a training-free adversarial defence.

Question (set by the prof): does augmentation applied only at TEST time, with NO
retraining, blunt an adversarial attack on the CT input? This is the inference-
time counterpart of the retraining study: there we used augmentation during
training to buy robustness; here we apply the same kinds of transform to the
(possibly adversarial) CT just before prediction and measure what they recover.

Method, per patient x attack x epsilon:
  1. Attack the *bare* model (FGSM / PGD, reused from adversarial_eval.py) to get
     an adversarial CT. This is a NON-adaptive attacker: it does not know about
     the defence. That is the right scope for a quick "does aug help" answer, but
     input-transformation defences are known to be largely defeated by adaptive
     attacks (Athalye et al. 2018), so we do not overclaim. Adaptive (BPDA) is
     the rigorous follow-up.
  2. Compute the adversarial CT ONCE and reuse it across every defence, so the
     differences between defences are not confounded by attack randomness.
  3. Apply each defence (test-time transform) to that CT, predict, score.

Two controls that keep it honest:
  * CLEAN COST. Every defence is also evaluated on the clean (unattacked) CT, so
    we can see how much accuracy the defence itself sacrifices. A defence only
    counts if it recovers the attacked loss without wrecking clean accuracy.
  * STRENGTH SWEEP. Each defence is swept over a grid of strengths, because a
    single blur sigma / noise level can make any input transform look arbitrarily
    good or useless. The real object is the tradeoff curve.

Scores are the OpenKBP competition metrics (dose score in Gy, DVH score),
computed in-memory by reusing DoseEvaluator's metric math on the predicted dose
array (no prediction CSVs written to disk). Lead with DVH, since the whole
project narrative is in DVH terms.

Run on the box (TF 2.18.0, GPU):
    python adversarial_defense.py \
        --model results/.../models/epoch_100.keras \
        --attack fgsm pgd --epsilons 0.02,0.05 \
        --defenses none smooth noise intensity flip \
        --n-patients 10 --output adversarial_defense_results/
Run epoch_100 (baseline) first: the `smooth` defence is essentially the P4 blur
that epoch_125 was trained to absorb, so on the robust model a "free" smooth
result is partly trained-in, not a pure test-time effect. Interpret the two
models separately.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from adversarial_eval import fgsm_attack, pgd_attack
from provided_code import DataLoader, DoseEvaluator, get_paths
from provided_code.network_architectures import InstanceNormalization
from provided_code.defense_transforms import (
    LR_AXIS_SAMPLE,
    add_noise,
    flip_lr,
    scale_intensity,
    smooth_ct,
)
from provided_code.defense_scoring import (
    add_derived,
    build_defense_list,
    cond_key,
    score_prediction,
)


def defended_predict(model, ct, masks, defense, strength, n_samples, rng):
    """Predict normalised dose for `ct` (1,D,H,W,1) under a test-time defence.

    Stochastic defences (noise, intensity) average the prediction over n_samples
    draws. `flip` is an averaged ensemble of the original and the L-R-mirrored
    pass (masks flipped with the CT; the predicted dose flipped back exactly).
    """
    if defense == "none":
        return model([ct, masks], training=False).numpy()

    sample = ct.numpy()[0]  # (D,H,W,C)

    if defense == "smooth":
        ct_def = smooth_ct(sample, strength)[None, ...]
        return model([tf.constant(ct_def), masks], training=False).numpy()

    if defense == "noise":
        preds = [
            model([tf.constant(add_noise(sample, strength, rng)[None, ...]), masks],
                  training=False).numpy()
            for _ in range(n_samples)
        ]
        return np.mean(preds, axis=0)

    if defense == "intensity":
        preds = []
        for _ in range(n_samples):
            factor = 1.0 + rng.uniform(-strength, strength)
            ct_def = scale_intensity(sample, factor)[None, ...]
            preds.append(model([tf.constant(ct_def), masks], training=False).numpy())
        return np.mean(preds, axis=0)

    if defense == "flip":
        p0 = model([ct, masks], training=False).numpy()
        ct_f = flip_lr(sample, LR_AXIS_SAMPLE)[None, ...]
        masks_f = flip_lr(masks.numpy()[0], LR_AXIS_SAMPLE)[None, ...]
        pf = model([tf.constant(ct_f), tf.constant(masks_f)], training=False).numpy()
        pf_back = flip_lr(pf[0], LR_AXIS_SAMPLE)[None, ...]  # output L-R axis == 1 too
        return 0.5 * (p0 + pf_back)

    raise ValueError(f"Unknown defense: {defense}")


def run(model, loader, attacks, epsilons, pgd_steps, defense_pairs, n_samples, seed):
    """Evaluate the full grid. Returns a list of score records (one per condition)."""
    eps_pos = [e for e in epsilons if e > 0]
    # One DoseEvaluator per condition (tiny: it only holds scalar metric tables).
    conditions = [("clean", 0.0, d, s) for (d, s) in defense_pairs]
    for atk in attacks:
        for eps in eps_pos:
            conditions += [(atk, eps, d, s) for (d, s) in defense_pairs]
    evaluators = {cond_key(*c): DoseEvaluator(loader) for c in conditions}

    n_patients = 0
    for batch in loader.get_batches():
        n_patients += 1
        ct_t = tf.constant(batch.ct, dtype=tf.float32)
        masks_t = tf.constant(batch.structure_masks, dtype=tf.float32)
        dose_true_t = tf.constant(batch.dose, dtype=tf.float32)
        pdm = batch.possible_dose_mask
        ref_dose_gy = (batch.dose * loader.DOSE_PRESCRIPTION).flatten()
        rng = np.random.default_rng(seed)  # per-patient reset -> reproducible

        # Compute each adversarial CT ONCE, reuse across all defences.
        advs = {("clean", 0.0): ct_t}
        for atk in attacks:
            for eps in eps_pos:
                if atk == "pgd":
                    advs[(atk, eps)] = pgd_attack(model, ct_t, masks_t, dose_true_t, eps, pgd_steps)
                else:
                    advs[(atk, eps)] = fgsm_attack(model, ct_t, masks_t, dose_true_t, eps)

        for (atk, eps), adv_ct in advs.items():
            for (defense, strength) in defense_pairs:
                pred_norm = defended_predict(model, adv_ct, masks_t, defense, strength, n_samples, rng)
                pred_gy = (pred_norm * pdm * loader.DOSE_PRESCRIPTION).flatten()
                score_prediction(evaluators[cond_key(atk, eps, defense, strength)], batch, ref_dose_gy, pred_gy)
        print(f"  patient {n_patients}: {batch.patient_list[0]} done")

    records = []
    for (atk, eps, defense, strength) in conditions:
        dose_score, dvh_score = evaluators[cond_key(atk, eps, defense, strength)].get_scores()
        records.append({
            "attack": atk, "epsilon": eps, "defense": defense, "strength": strength,
            "dose_score": float(dose_score), "dvh_score": float(dvh_score),
        })
    return records, n_patients


def print_table(records):
    print(f"\n{'attack':<7}{'eps':>7}  {'defense':<10}{'str':>6}  {'DVH':>8}{'dose':>8}  "
          f"{'cleanΔdvh':>10}{'recov%dvh':>10}")
    print("-" * 74)
    for r in sorted(records, key=lambda r: (r["attack"], r["epsilon"], r["defense"], r["strength"] or 0)):
        s = "" if r["strength"] is None else f"{r['strength']:g}"
        rec = r.get("recovered_frac_dvh")
        rec_s = "" if rec is None else f"{100 * rec:+.0f}"
        cc = r.get("clean_cost_dvh")
        cc_s = "" if cc is None else f"{cc:+.3f}"
        print(f"{r['attack']:<7}{r['epsilon']:>7.3g}  {r['defense']:<10}{s:>6}  "
              f"{r['dvh_score']:>8.3f}{r['dose_score']:>8.3f}  {cc_s:>10}{rec_s:>10}")


def main():
    p = argparse.ArgumentParser(description="Test-time augmentation adversarial defence")
    p.add_argument("--model", required=True, help="Path to trained .keras model")
    p.add_argument("--attack", nargs="+", choices=["fgsm", "pgd"], default=["fgsm", "pgd"])
    p.add_argument("--epsilons", default="0.02,0.05", help="Comma-separated; 0 is implicit (clean)")
    p.add_argument("--pgd-steps", type=int, default=10)
    p.add_argument("--defenses", nargs="+",
                   choices=["none", "smooth", "noise", "intensity", "flip"],
                   default=["none", "smooth", "noise", "intensity", "flip"])
    p.add_argument("--noise-samples", type=int, default=4, help="Draws averaged for stochastic defences")
    p.add_argument("--n-patients", type=int, default=None, help="Subset for a fast first pass")
    p.add_argument("--quick", action="store_true", help="One (middle) strength per defence")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--output", default="adversarial_defense_results")
    args = p.parse_args()

    epsilons = [float(e) for e in args.epsilons.split(",")]
    defenses = args.defenses if "none" in args.defenses else ["none"] + args.defenses
    defense_pairs = build_defense_list(defenses, args.quick)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Locate validation data.
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        script_dir = Path(__file__).parent
        data_dir = script_dir / "provided-data" / "validation-pats"
        if not data_dir.exists():
            data_dir = script_dir.parent / "provided-data" / "validation-pats"
    if not data_dir.exists():
        print(f"ERROR: validation data not found at {data_dir}")
        return 1

    print(f"Loading model: {args.model}")
    model = load_model(args.model,
                       custom_objects={"InstanceNormalization": InstanceNormalization},
                       compile=False, safe_mode=False)

    paths = sorted(get_paths(data_dir))
    if args.n_patients:
        paths = paths[:args.n_patients]
    print(f"Validation patients: {len(paths)} (defences: {defense_pairs})")

    loader = DataLoader(paths, batch_size=1, normalize=True, cache_data=True)
    loader.set_mode("training_model")

    records, n_patients = run(model, loader, args.attack, epsilons, args.pgd_steps,
                              defense_pairs, args.noise_samples, args.seed)
    records = add_derived(records)
    print_table(records)

    summary = {
        "model": str(args.model),
        "timestamp": datetime.now().isoformat(),
        "n_patients": n_patients,
        "attacks": args.attack, "epsilons": epsilons, "pgd_steps": args.pgd_steps,
        "defenses": defense_pairs, "noise_samples": args.noise_samples,
        "records": records,
        "note": "Non-adaptive attacker (defence-unaware). DVH is the primary metric.",
    }
    out_file = out_dir / "defense_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_file}")
    return 0


if __name__ == "__main__":
    exit(main())
