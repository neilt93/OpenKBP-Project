#!/usr/bin/env python3
"""
Adversarial Attack Evaluation for OpenKBP Dose Prediction

Evaluates model robustness using FGSM and PGD attacks on CT inputs.
Measures how prediction quality degrades as perturbation strength increases.

Usage:
    python adversarial_eval.py --model path/to/model.keras --attack fgsm pgd
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from provided_code import DataLoader, DoseEvaluator, get_paths
from provided_code.network_architectures import InstanceNormalization


# Epsilon to HU conversion (CT normalized by CT_MAX=4095):
#   epsilon 0.01 ≈  41 HU    (within typical CT noise)
#   epsilon 0.025 ≈ 102 HU
#   epsilon 0.05 ≈ 205 HU
#   epsilon 0.1  ≈ 410 HU    (large, visible perturbation)
# Typical CT scanner noise: 10-50 HU
# Formula: HU = epsilon * 4095


def fgsm_attack(
    model: tf.keras.Model,
    ct: tf.Tensor,
    structure_masks: tf.Tensor,
    dose_true: tf.Tensor,
    epsilon: float,
) -> tf.Tensor:
    """
    Fast Gradient Sign Method (FGSM) attack.

    Perturbs CT input in the direction that maximizes the loss.

    Args:
        model: Trained dose prediction model
        ct: Original CT tensor (batch, 128, 128, 128, 1)
        structure_masks: Structure masks (batch, 128, 128, 128, 10)
        dose_true: Ground truth dose (batch, 128, 128, 128, 1)
        epsilon: Perturbation magnitude in normalized [0,1] space.
                 Multiply by 4095 to get approximate HU equivalent.

    Returns:
        Adversarial CT tensor
    """
    if epsilon == 0:
        return ct

    ct_var = tf.Variable(ct, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(ct_var)
        dose_pred = model([ct_var, structure_masks], training=False)
        # Use MAE loss (same as training)
        loss = tf.reduce_mean(tf.abs(dose_true - dose_pred))

    grad = tape.gradient(loss, ct_var)
    # Perturb in direction that increases loss
    perturbation = epsilon * tf.sign(grad)
    ct_adv = ct_var + perturbation

    # Clip to valid range [0, 1]
    return tf.clip_by_value(ct_adv, 0.0, 1.0)


def pgd_attack(
    model: tf.keras.Model,
    ct: tf.Tensor,
    structure_masks: tf.Tensor,
    dose_true: tf.Tensor,
    epsilon: float,
    steps: int = 10,
    alpha: float = None,
) -> tf.Tensor:
    """
    Projected Gradient Descent (PGD) attack.

    Iteratively perturbs CT input, projecting back to epsilon-ball each step.

    Args:
        model: Trained dose prediction model
        ct: Original CT tensor
        structure_masks: Structure masks
        dose_true: Ground truth dose
        epsilon: Maximum perturbation magnitude
        steps: Number of PGD iterations
        alpha: Step size (default: 2*epsilon/steps)

    Returns:
        Adversarial CT tensor
    """
    if epsilon == 0:
        return ct

    alpha = alpha or (2.0 * epsilon / steps)
    ct_orig = tf.constant(ct, dtype=tf.float32)
    ct_adv = tf.Variable(ct, dtype=tf.float32)

    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(ct_adv)
            dose_pred = model([ct_adv, structure_masks], training=False)
            loss = tf.reduce_mean(tf.abs(dose_true - dose_pred))

        grad = tape.gradient(loss, ct_adv)
        # Take gradient step
        ct_adv.assign_add(alpha * tf.sign(grad))

        # Project back to epsilon-ball around original
        perturbation = ct_adv - ct_orig
        perturbation = tf.clip_by_value(perturbation, -epsilon, epsilon)
        ct_adv.assign(tf.clip_by_value(ct_orig + perturbation, 0.0, 1.0))

    return ct_adv


def compute_dose_metrics(
    dose_pred: np.ndarray,
    dose_true: np.ndarray,
    possible_dose_mask: np.ndarray,
) -> dict:
    """Compute voxel-wise dose error metrics."""
    mask = possible_dose_mask.astype(bool).flatten()
    pred_flat = dose_pred.flatten()[mask]
    true_flat = dose_true.flatten()[mask]

    errors = np.abs(pred_flat - true_flat)

    return {
        "mae": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "max_error": float(np.max(errors)),
        "p95_error": float(np.percentile(errors, 95)),
        "errors": errors.tolist(),  # Full distribution for histograms
    }


def evaluate_attack(
    model: tf.keras.Model,
    data_loader: DataLoader,
    epsilon: float,
    attack_type: str = "fgsm",
    pgd_steps: int = 10,
) -> dict:
    """
    Evaluate model under adversarial attack at given epsilon.

    Args:
        model: Trained model
        data_loader: Validation data loader
        epsilon: Perturbation magnitude
        attack_type: 'fgsm' or 'pgd'
        pgd_steps: Number of steps for PGD

    Returns:
        Dictionary with per-patient results
    """
    results = {
        "attack": attack_type,
        "epsilon": epsilon,
        "pgd_steps": pgd_steps if attack_type == "pgd" else None,
        "patients": [],
    }

    attack_fn = fgsm_attack if attack_type == "fgsm" else pgd_attack

    for batch in data_loader.get_batches():
        # Convert to tensors
        ct = tf.constant(batch.ct, dtype=tf.float32)
        structure_masks = tf.constant(batch.structure_masks, dtype=tf.float32)
        dose_true = tf.constant(batch.dose, dtype=tf.float32)

        # Apply attack
        if attack_type == "pgd":
            ct_adv = attack_fn(model, ct, structure_masks, dose_true, epsilon, pgd_steps)
        else:
            ct_adv = attack_fn(model, ct, structure_masks, dose_true, epsilon)

        # Get prediction on adversarial input
        dose_pred = model([ct_adv, structure_masks], training=False)
        dose_pred = dose_pred.numpy() * batch.possible_dose_mask

        # Denormalize (model outputs normalized dose)
        dose_pred_gy = dose_pred * data_loader.DOSE_PRESCRIPTION
        dose_true_gy = batch.dose * data_loader.DOSE_PRESCRIPTION

        # Compute metrics
        metrics = compute_dose_metrics(
            dose_pred_gy, dose_true_gy, batch.possible_dose_mask
        )

        patient_id = batch.patient_list[0]
        results["patients"].append({
            "patient_id": patient_id,
            **{k: v for k, v in metrics.items() if k != "errors"},
        })

        # Store errors separately (large arrays)
        if "all_errors" not in results:
            results["all_errors"] = []
        results["all_errors"].extend(metrics["errors"][:10000])  # Sample for memory

    # Compute summary statistics
    maes = [p["mae"] for p in results["patients"]]
    results["summary"] = {
        "mean_mae": float(np.mean(maes)),
        "std_mae": float(np.std(maes)),
        "min_mae": float(np.min(maes)),
        "max_mae": float(np.max(maes)),
        "n_patients": len(results["patients"]),
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate adversarial robustness of dose prediction model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.keras file)",
    )
    parser.add_argument(
        "--attack",
        nargs="+",
        choices=["fgsm", "pgd"],
        default=["fgsm", "pgd"],
        help="Attack types to evaluate",
    )
    parser.add_argument(
        "--epsilons",
        type=str,
        default="0,0.001,0.005,0.01,0.02,0.05,0.1",
        help="Comma-separated epsilon values",
    )
    parser.add_argument(
        "--pgd-steps",
        type=int,
        default=10,
        help="Number of PGD steps (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="adversarial_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to validation data (default: provided-data/validation-pats)",
    )
    args = parser.parse_args()

    # Parse epsilons
    epsilons = [float(e) for e in args.epsilons.split(",")]

    # Setup paths
    model_path = Path(args.model)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        # Try relative to model or script location
        script_dir = Path(__file__).parent
        data_dir = script_dir / "provided-data" / "validation-pats"
        if not data_dir.exists():
            data_dir = script_dir.parent / "provided-data" / "validation-pats"

    if not data_dir.exists():
        print(f"ERROR: Validation data not found at {data_dir}")
        return 1

    print(f"Loading model from {model_path}")
    model = load_model(
        model_path,
        custom_objects={"InstanceNormalization": InstanceNormalization},
        compile=False,  # Skip compilation to avoid version mismatch issues
        safe_mode=False,  # Allow loading models from different Keras versions
    )

    print(f"Loading validation data from {data_dir}")
    validation_paths = get_paths(data_dir)
    print(f"Found {len(validation_paths)} validation patients")

    # Use training_model mode to get both CT and dose data
    data_loader = DataLoader(
        validation_paths,
        batch_size=1,
        normalize=True,  # CT/dose normalized to match model expectations
        cache_data=True,
    )
    data_loader.set_mode("training_model")

    # Run evaluations
    all_results = {
        "model": str(model_path),
        "timestamp": datetime.now().isoformat(),
        "epsilons": epsilons,
        "attacks": args.attack,
        "results": {},
    }

    for attack in args.attack:
        all_results["results"][attack] = {}
        for eps in epsilons:
            print(f"\nEvaluating {attack.upper()} attack with epsilon={eps:.4f}")
            result = evaluate_attack(
                model, data_loader, eps, attack, args.pgd_steps
            )

            # Save individual result
            result_file = output_dir / f"{attack}_eps{eps:.4f}.json"
            # Remove large error arrays for individual files
            result_save = {k: v for k, v in result.items() if k != "all_errors"}
            with open(result_file, "w") as f:
                json.dump(result_save, f, indent=2)

            # Store summary in combined results
            all_results["results"][attack][str(eps)] = result["summary"]

            print(f"  Mean MAE: {result['summary']['mean_mae']:.3f} Gy")

    # Save combined summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print(f"Summary: {summary_file}")

    return 0


if __name__ == "__main__":
    exit(main())
