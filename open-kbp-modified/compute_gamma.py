#!/usr/bin/env python3
"""Gamma passing rate (GPR) as a robustness metric — clean vs perturbed dose prediction.

For each patient: predict dose on the clean CT and on the adversarial CT (PGD), optionally
with the test-time noise defence, then compute the 3D gamma passing rate of the perturbed
prediction against the CLEAN prediction (does the perturbed plan still agree within 3%/3mm?).
Reports mean GPR over patients at 3%/3mm and 3%/2mm (TG-218). Gamma math (pure numpy,
unit-tested) is in provided_code/gamma_index.py; this script owns the TF inference.

GPR here is EMPIRICAL (two concrete dose volumes); a certified gamma bound is future work
(gamma's DTA min is non-monotone, unlike the certified-DVH push-through). Coarse OpenKBP
voxels (~3-5 mm) make a 3 mm DTA sub-voxel in-plane — GPR is reported with spacing stated.

Run on the box (TF 2.18.0):
    python compute_gamma.py --model models/epoch_100.keras --data-dir $DATA \
        --epsilons 0.02,0.05 --defense-sigma 0.1 --defense-samples 8 \
        --n-patients 10 --output gamma_results/
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from adversarial_eval import pgd_attack
from provided_code import DataLoader, get_paths
from provided_code.network_architectures import InstanceNormalization
from provided_code.gamma_index import gamma_passing_rate

CRITERIA = [(3.0, 3.0), (3.0, 2.0)]  # (dd%, dta mm): 3%/3mm and TG-218 3%/2mm


def defended(model, ct, masks, sigma, n, rng):
    if sigma <= 0 or n <= 1:
        return model([ct, masks], training=False).numpy()
    base = ct.numpy()
    preds = [model([tf.constant(np.clip(base + rng.normal(0, sigma, base.shape).astype(np.float32), 0, 1)), masks],
                   training=False).numpy() for _ in range(n)]
    return np.mean(preds, axis=0)


def main():
    p = argparse.ArgumentParser(description="Gamma passing rate: clean vs perturbed dose prediction")
    p.add_argument("--model", required=True)
    p.add_argument("--epsilons", default="0.02,0.05")
    p.add_argument("--pgd-steps", type=int, default=10)
    p.add_argument("--defense-sigma", type=float, default=0.0, help="0 = no defence; else noise-defended pred too")
    p.add_argument("--defense-samples", type=int, default=8)
    p.add_argument("--n-patients", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--output", default="gamma_results")
    args = p.parse_args()

    epsilons = [float(e) for e in args.epsilons.split(",")]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    script_dir = Path(__file__).parent
    data_dir = Path(args.data_dir) if args.data_dir else script_dir / "provided-data" / "validation-pats"
    if not data_dir.exists():
        data_dir = script_dir.parent / "provided-data" / "validation-pats"

    print(f"Loading model {args.model}")
    model = load_model(args.model, custom_objects={"InstanceNormalization": InstanceNormalization},
                       compile=False, safe_mode=False)
    paths = sorted(get_paths(data_dir))
    if args.n_patients:
        paths = paths[:args.n_patients]
    loader = DataLoader(paths, batch_size=1, normalize=True, cache_data=True)
    loader.set_mode("training_model")
    presc = loader.DOSE_PRESCRIPTION

    per_patient = []
    for batch in loader.get_batches():
        ct = tf.constant(batch.ct, dtype=tf.float32)
        masks = tf.constant(batch.structure_masks, dtype=tf.float32)
        dose_true = tf.constant(batch.dose, dtype=tf.float32)
        pdm = batch.possible_dose_mask[0, ..., 0].astype(bool)
        spacing = np.asarray(batch.voxel_dimensions).reshape(-1)[:3]
        rng = np.random.default_rng(args.seed)

        clean = model([ct, masks], training=False).numpy()[0, ..., 0] * pdm * presc
        rec = {"patient": batch.patient_list[0], "spacing_mm": [float(s) for s in spacing], "eps": {}}
        for eps in epsilons:
            adv = pgd_attack(model, ct, masks, dose_true, eps, args.pgd_steps)
            pert = model([adv, masks], training=False).numpy()[0, ..., 0] * pdm * presc
            entry = {}
            for (dd, dta) in CRITERIA:
                entry[f"undef_{dd:g}_{dta:g}mm"] = gamma_passing_rate(clean, pert, spacing, dd, dta)
            if args.defense_sigma > 0:
                dfd = defended(model, adv, masks, args.defense_sigma, args.defense_samples, rng)[0, ..., 0] * pdm * presc
                for (dd, dta) in CRITERIA:
                    entry[f"def_{dd:g}_{dta:g}mm"] = gamma_passing_rate(clean, dfd, spacing, dd, dta)
            rec["eps"][f"{eps:g}"] = entry
        per_patient.append(rec)
        print(f"  {rec['patient']}: " + "  ".join(
            f"eps{e} GPR3/3={rec['eps'][e]['undef_3_3mm']:.3f}" for e in rec["eps"]))

    # aggregate: mean GPR over patients
    agg = {}
    for eps in [f"{e:g}" for e in epsilons]:
        keys = per_patient[0]["eps"][eps].keys()
        agg[eps] = {k: float(np.nanmean([p["eps"][eps][k] for p in per_patient])) for k in keys}
    print("\n=== mean GPR over patients ===")
    for eps, d in agg.items():
        print(f"eps={eps}: " + "  ".join(f"{k}={v:.3f}" for k, v in d.items()))

    summary = {"model": str(args.model), "timestamp": datetime.now().isoformat(),
               "epsilons": epsilons, "criteria": CRITERIA, "defense_sigma": args.defense_sigma,
               "n_patients": len(per_patient), "aggregate": agg, "per_patient": per_patient,
               "note": "Empirical gamma passing rate, perturbed-vs-clean prediction. TG-218 tol/action = 95/90% at 3%/2mm."}
    with open(out / "gamma_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out/'gamma_summary.json'}")
    return 0


if __name__ == "__main__":
    exit(main())
