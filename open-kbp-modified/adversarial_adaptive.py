#!/usr/bin/env python3
"""Adaptive (EOT) attack on the test-time noise defence — the honest next step.

`adversarial_defense.py` showed test-time Gaussian noise on the CT recovers
60-104% of FGSM/PGD damage — but against a NON-adaptive attacker that does not
know the defence is there. Input-transformation defences routinely collapse
under an adaptive attacker (Athalye et al. 2018, "Obfuscated Gradients"). This
script runs that adaptive attacker and measures how much of the recovery survives.

The noise defence averages the prediction over K Gaussian CT draws. That whole
pipeline is DIFFERENTIABLE (the noise is additive and input-independent), so the
correct adaptive attack is not BPDA but EOT (Expectation Over Transformation,
Athalye et al. 2018): at each PGD step, estimate the gradient of the loss of the
NOISE-AVERAGED prediction by averaging the gradient over n_eot fresh noise draws,
then take the sign step. The attacker is optimising exactly the quantity the
defence reports.

The four conditions that tell the story (per patient x epsilon):
  1. clean, undefended                — reference accuracy
  2. non-adaptive PGD, undefended     — raw attack damage
  3. non-adaptive PGD, noise-defended — the recovery `adversarial_defense.py` claims
  4. ADAPTIVE  EOT-PGD, noise-defended — what survives an attacker that knows

If (4) is much worse than (3), the empirical defence is an obfuscated-gradient
artefact, not real robustness — which is precisely the motivation for the
certified defence (`certify_smoothing.py`): a guarantee no adaptive attacker can
chase. Scores are the OpenKBP metrics (DVH primary), computed in-memory by
reusing DoseEvaluator (no prediction CSVs).

Run on the box (TF 2.18.0, GPU):
    python adversarial_adaptive.py \
        --model results/.../models/epoch_100.keras \
        --epsilons 0.02,0.05 --pgd-steps 10 --eot-samples 8 \
        --defense-sigma 0.1 --defense-samples 8 \
        --n-patients 10 --output adaptive_results/
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from adversarial_eval import pgd_attack
from provided_code import DataLoader, DoseEvaluator, get_paths
from provided_code.network_architectures import InstanceNormalization
from provided_code.defense_scoring import cond_key, score_prediction


def eot_pgd_attack(model, ct, masks, dose_true, epsilon, steps, sigma, n_eot, rng, alpha=None):
    """Adaptive PGD against the noise defence via Expectation Over Transformation.

    At each step the loss is the MAE of the noise-averaged prediction; its gradient
    is estimated by averaging over `n_eot` fresh Gaussian CT draws (std `sigma`,
    the SAME noise the defence uses). Perturbation is projected to the L-inf
    epsilon-ball and the CT kept in [0, 1] — same geometry as the base PGD, so the
    only difference from a non-adaptive attack is defence-awareness.
    """
    if epsilon == 0:
        return ct
    alpha = alpha or (2.0 * epsilon / steps)
    ct_orig = tf.constant(ct, dtype=tf.float32)
    ct_adv = tf.Variable(ct, dtype=tf.float32)

    for _ in range(steps):
        grad_accum = tf.zeros_like(ct_orig)
        for _ in range(n_eot):
            noise = tf.constant(
                rng.normal(0.0, sigma, size=ct_orig.shape).astype(np.float32))
            with tf.GradientTape() as tape:
                tape.watch(ct_adv)
                noisy = tf.clip_by_value(ct_adv + noise, 0.0, 1.0)
                dose_pred = model([noisy, masks], training=False)
                loss = tf.reduce_mean(tf.abs(dose_true - dose_pred))
            grad_accum += tape.gradient(loss, ct_adv)
        grad = grad_accum / float(n_eot)
        ct_adv.assign_add(alpha * tf.sign(grad))
        perturbation = tf.clip_by_value(ct_adv - ct_orig, -epsilon, epsilon)
        ct_adv.assign(tf.clip_by_value(ct_orig + perturbation, 0.0, 1.0))
    return ct_adv


def noise_defended_predict(model, ct, masks, sigma, n_samples, rng):
    """Randomised-smoothing-style defended prediction: mean over n noise draws.
    sigma <= 0 or n <= 1 falls back to a single clean forward pass (undefended)."""
    if sigma <= 0 or n_samples <= 1:
        return model([ct, masks], training=False).numpy()
    base = ct.numpy()
    preds = []
    for _ in range(n_samples):
        noisy = np.clip(base + rng.normal(0.0, sigma, size=base.shape).astype(np.float32), 0.0, 1.0)
        preds.append(model([tf.constant(noisy), masks], training=False).numpy())
    return np.mean(preds, axis=0)


def run(model, loader, epsilons, pgd_steps, eot_samples, defense_sigma, defense_samples, seed):
    eps_pos = [e for e in epsilons if e > 0]
    # Conditions: (label, attack, eps, defended?)
    conditions = [("clean_undef", "clean", 0.0, False),
                  ("clean_def", "clean", 0.0, True)]
    for eps in eps_pos:
        conditions += [
            ("nonadaptive_undef", "pgd", eps, False),
            ("nonadaptive_def", "pgd", eps, True),
            ("adaptive_def", "eot_pgd", eps, True),
        ]
    evaluators = {cond_key(lbl, eps, "d" if dfd else "u", None): DoseEvaluator(loader)
                  for (lbl, atk, eps, dfd) in conditions}

    n_done = 0
    for batch in loader.get_batches():
        n_done += 1
        ct_t = tf.constant(batch.ct, dtype=tf.float32)
        masks_t = tf.constant(batch.structure_masks, dtype=tf.float32)
        dose_true_t = tf.constant(batch.dose, dtype=tf.float32)
        pdm = batch.possible_dose_mask
        ref_dose_gy = (batch.dose * pdm * loader.DOSE_PRESCRIPTION).flatten()

        # Precompute adversarial CTs once per epsilon (non-adaptive shares across
        # its defended/undefended readout; adaptive is its own).
        advs = {}
        for eps in eps_pos:
            adv_na = pgd_attack(model, ct_t, masks_t, dose_true_t, eps, pgd_steps)
            rng_atk = np.random.default_rng(seed + 12345)  # attack noise stream
            adv_ad = eot_pgd_attack(model, ct_t, masks_t, dose_true_t, eps,
                                    pgd_steps, defense_sigma, eot_samples, rng_atk)
            advs[eps] = {"na": adv_na, "ad": adv_ad}

        for (lbl, atk, eps, dfd) in conditions:
            rng_def = np.random.default_rng(seed)  # defence noise stream (reproducible)
            if atk == "clean":
                ct_use = ct_t
            elif atk == "pgd":
                ct_use = advs[eps]["na"]
            else:
                ct_use = advs[eps]["ad"]
            if dfd:
                pred_norm = noise_defended_predict(model, ct_use, masks_t, defense_sigma, defense_samples, rng_def)
            else:
                pred_norm = model([ct_use, masks_t], training=False).numpy()
            pred_gy = (pred_norm * pdm * loader.DOSE_PRESCRIPTION).flatten()
            ev = evaluators[cond_key(lbl, eps, "d" if dfd else "u", None)]
            score_prediction(ev, batch, ref_dose_gy, pred_gy)
        print(f"  patient {n_done}: {batch.patient_list[0]} done")

    records = []
    for (lbl, atk, eps, dfd) in conditions:
        dose_score, dvh_score = evaluators[cond_key(lbl, eps, "d" if dfd else "u", None)].get_scores()
        records.append({"condition": lbl, "attack": atk, "epsilon": eps, "defended": dfd,
                        "dose_score": float(dose_score), "dvh_score": float(dvh_score)})
    return records, n_done


def summarize(records):
    """Per epsilon: attack damage, and fraction recovered by the noise defence under
    the NON-adaptive vs the ADAPTIVE attacker. The gap between the two recovery
    fractions is the headline — how much of the empirical defence is obfuscation."""
    def find(cond, eps):
        for r in records:
            if r["condition"] == cond and (eps is None or abs(r["epsilon"] - eps) < 1e-12):
                return r
        return None

    clean = find("clean_undef", 0.0)
    out = []
    epsilons = sorted({r["epsilon"] for r in records if r["epsilon"] > 0})
    for eps in epsilons:
        na_undef = find("nonadaptive_undef", eps)
        na_def = find("nonadaptive_def", eps)
        ad_def = find("adaptive_def", eps)
        row = {"epsilon": eps}
        for m in ("dvh", "dose"):
            base = clean[f"{m}_score"]
            dmg = na_undef[f"{m}_score"] - base  # non-adaptive raw damage (undefended)
            na_rem = na_def[f"{m}_score"] - base
            ad_rem = ad_def[f"{m}_score"] - base
            row[f"{m}_clean"] = base
            row[f"{m}_attack_damage"] = dmg
            row[f"{m}_nonadaptive_recovered"] = (1 - na_rem / dmg) if abs(dmg) > 1e-9 else None
            row[f"{m}_adaptive_recovered"] = (1 - ad_rem / dmg) if abs(dmg) > 1e-9 else None
            row[f"{m}_adaptive_def_score"] = ad_def[f"{m}_score"]
        out.append(row)
    return out


def print_table(records, summary):
    print(f"\n{'condition':<20}{'eps':>7}{'def':>5}  {'DVH':>8}{'dose':>8}")
    print("-" * 50)
    for r in sorted(records, key=lambda r: (r["epsilon"], r["condition"])):
        print(f"{r['condition']:<20}{r['epsilon']:>7.3g}{'Y' if r['defended'] else 'N':>5}  "
              f"{r['dvh_score']:>8.3f}{r['dose_score']:>8.3f}")
    print(f"\n{'eps':>7}  {'damage(DVH)':>12}{'nonadapt rec%':>15}{'ADAPTIVE rec%':>15}")
    print("-" * 52)
    for s in summary:
        na = s["dvh_nonadaptive_recovered"]
        ad = s["dvh_adaptive_recovered"]
        na_s = "" if na is None else f"{100 * na:+.0f}"
        ad_s = "" if ad is None else f"{100 * ad:+.0f}"
        print(f"{s['epsilon']:>7.3g}  {s['dvh_attack_damage']:>12.3f}{na_s:>15}{ad_s:>15}")


def main():
    p = argparse.ArgumentParser(description="Adaptive (EOT) attack on the noise defence")
    p.add_argument("--model", required=True)
    p.add_argument("--epsilons", default="0.02,0.05")
    p.add_argument("--pgd-steps", type=int, default=10)
    p.add_argument("--eot-samples", type=int, default=8, help="Noise draws averaged in the EOT gradient")
    p.add_argument("--defense-sigma", type=float, default=0.1, help="Defence noise std (matches adversarial_defense best)")
    p.add_argument("--defense-samples", type=int, default=8, help="Noise draws averaged by the defence at inference")
    p.add_argument("--n-patients", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--output", default="adaptive_results")
    args = p.parse_args()

    epsilons = [float(e) for e in args.epsilons.split(",")]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"Validation patients: {len(paths)}  eps={epsilons}  eot={args.eot_samples}  "
          f"defence sigma={args.defense_sigma} x{args.defense_samples}")

    loader = DataLoader(paths, batch_size=1, normalize=True, cache_data=True)
    loader.set_mode("training_model")

    records, n_done = run(model, loader, epsilons, args.pgd_steps, args.eot_samples,
                          args.defense_sigma, args.defense_samples, args.seed)
    summary = summarize(records)
    print_table(records, summary)

    out = {
        "model": str(args.model),
        "timestamp": datetime.now().isoformat(),
        "n_patients": n_done,
        "epsilons": epsilons, "pgd_steps": args.pgd_steps, "eot_samples": args.eot_samples,
        "defense_sigma": args.defense_sigma, "defense_samples": args.defense_samples,
        "records": records, "summary": summary,
        "note": ("EOT adaptive attack vs non-adaptive, both read out under the noise defence. "
                 "Gap in recovered% = obfuscated-gradient component of the empirical defence."),
    }
    out_file = out_dir / "adaptive_summary.json"
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_file}")
    return 0


if __name__ == "__main__":
    exit(main())
