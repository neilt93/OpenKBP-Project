#!/usr/bin/env python3
"""Step 2: Run model inference on baseline + all perturbed conditions.

Requires GPU (RunPod) with TensorFlow 2.18.0.

Usage:
    cd open-kbp-modified/
    python openkbp_hn_robustness/run_inference.py [--config CONFIG] [--model MODEL]

The script:
1. Loads the trained model (.keras)
2. Runs inference on baseline (unperturbed) validation patients
3. Runs inference on each perturbation/level condition
4. Saves predictions as sparse CSVs matching the OpenKBP format
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add project root to path for imports
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def save_dose_prediction(dose_pred: np.ndarray, output_path: Path) -> None:
    """Save a dose prediction as sparse CSV."""
    flat = dose_pred.flatten()
    mask = flat > 0
    indices = np.where(mask)[0]
    data = flat[mask]

    df = pd.DataFrame(data=data, index=indices, columns=["data"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path)


def run_condition(model, patient_paths: list, output_dir: Path, config: dict) -> None:
    """Run inference for a single condition (baseline or perturbed)."""
    from provided_code.data_loader import DataLoader

    normalize = config["inference"]["normalize"]
    dose_prescription = config["inference"]["dose_prescription"]

    loader = DataLoader(patient_paths, batch_size=1, normalize=normalize, cache_data=False)
    loader.set_mode("dose_prediction")

    output_dir.mkdir(parents=True, exist_ok=True)

    for batch in loader.get_batches():
        patient_id = batch.patient_list[0]
        out_path = output_dir / f"{patient_id}.csv"

        if out_path.exists():
            print(f"  Skipping {patient_id} (already exists)")
            continue

        dose_pred = model.predict([batch.ct, batch.structure_masks], verbose=0)
        dose_pred = dose_pred * batch.possible_dose_mask

        # Denormalize: model outputs normalized dose, we need Gy
        if normalize:
            dose_pred = dose_pred * dose_prescription

        dose_pred = np.squeeze(dose_pred)
        save_dose_prediction(dose_pred, out_path)
        print(f"  Saved {patient_id}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on perturbed data")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--model", type=str, default=None, help="Path to .keras model (overrides config)")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Only run specific conditions (e.g., baseline P1_noise/L1)")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline inference")
    args = parser.parse_args()

    # Load config
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = script_dir / "configs" / "default.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Import TensorFlow and load model
    import tensorflow as tf
    from provided_code.network_architectures import InstanceNormalization

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU detected: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("WARNING: No GPU detected, inference will be slow!")

    model_path = args.model or str(project_root / config["paths"]["model"])
    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"InstanceNormalization": InstanceNormalization},
    )
    print("Model loaded successfully")

    # Set up paths
    validation_dir = project_root / config["paths"]["validation_data"]
    perturbed_root = project_root / config["paths"]["data_perturbed"]
    predictions_root = project_root / config["paths"]["predictions"]

    # Build patient list
    start = config["patients"]["start"]
    end = config["patients"]["end"]
    patient_ids = [f"pt_{i}" for i in range(start, end + 1)]

    # 1. Baseline inference
    if not args.skip_baseline and (args.conditions is None or "baseline" in args.conditions):
        print(f"\n{'=' * 60}")
        print("Running baseline inference")
        print(f"{'=' * 60}")
        baseline_paths = [validation_dir / pid for pid in patient_ids if (validation_dir / pid).exists()]
        run_condition(model, baseline_paths, predictions_root / "baseline", config)

    # 2. Perturbed conditions
    for p_name, p_config in config["perturbations"].items():
        if not p_config.get("enabled", True):
            continue

        for level in p_config["levels"]:
            condition = f"{p_name}/{level}"

            if args.conditions is not None and condition not in args.conditions:
                continue

            condition_dir = perturbed_root / p_name / level
            if not condition_dir.exists():
                print(f"\nSkipping {condition} (data not found at {condition_dir})")
                continue

            print(f"\n{'=' * 60}")
            print(f"Running inference: {condition}")
            print(f"{'=' * 60}")

            patient_paths = sorted([
                condition_dir / pid for pid in patient_ids
                if (condition_dir / pid).exists()
            ])

            if not patient_paths:
                print(f"  No patient directories found, skipping")
                continue

            run_condition(model, patient_paths, predictions_root / p_name / level, config)

    print(f"\n{'=' * 60}")
    print("Inference complete!")
    print(f"Predictions saved to: {predictions_root}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
