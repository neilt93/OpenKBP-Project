#!/usr/bin/env python3
"""
Ensemble Prediction Script for OpenKBP

Averages predictions from multiple trained models to improve robustness and accuracy.

Usage:
    python ensemble_predict.py --models model1 model2 model3 --epoch 100 --output ensemble_v1

Example:
    # Train 3 models with different seeds
    python runpod_train.py --filters 64 --epochs 100 --use-se --use-aug --seed 42
    python runpod_train.py --filters 64 --epochs 100 --use-se --use-aug --seed 123
    python runpod_train.py --filters 64 --epochs 100 --use-se --use-aug --seed 456

    # Ensemble predictions
    python ensemble_predict.py \
        --models 64filter_100epoch_SE_AUG_NORM_seed42 64filter_100epoch_SE_AUG_NORM_seed123 64filter_100epoch_SE_AUG_NORM_seed456 \
        --epoch 100 \
        --output ensemble_3models
"""
import argparse
import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from tensorflow.keras.models import load_model

from provided_code import DataLoader, DoseEvaluator, get_paths
from provided_code.utils import sparse_vector_function


def ensemble_predict(
    model_names: List[str],
    results_dir: Path,
    validation_data_dir: Path,
    output_name: str,
    epoch: int,
    normalize: bool = True,
) -> Path:
    """
    Run ensemble prediction by averaging outputs from multiple models.

    Args:
        model_names: List of model names (folder names in results_dir)
        results_dir: Directory containing model results
        validation_data_dir: Directory containing validation patient data
        output_name: Name for the ensemble output folder
        epoch: Epoch number to load models from
        normalize: Whether the models were trained with normalization

    Returns:
        Path to the ensemble predictions directory
    """
    # Validate model paths
    model_paths = []
    for name in model_names:
        model_path = results_dir / name / "models" / f"epoch_{epoch}.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        model_paths.append(model_path)

    print(f"Loading {len(model_paths)} models for ensemble...")

    # Load all models
    models = []
    for path in model_paths:
        print(f"  Loading: {path.parent.parent.name}")
        models.append(load_model(path))

    # Setup data loader
    hold_out_plan_paths = get_paths(validation_data_dir)
    data_loader = DataLoader(hold_out_plan_paths, normalize=normalize)
    data_loader.set_mode("dose_prediction")

    # Output directory
    ensemble_dir = results_dir / output_name
    prediction_dir = ensemble_dir / "validation-predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running ensemble predictions on {len(hold_out_plan_paths)} patients...")

    # Run predictions
    for batch in tqdm(data_loader.get_batches()):
        # Get predictions from all models
        predictions = []
        for model in models:
            pred = model.predict([batch.ct, batch.structure_masks], verbose=0)
            predictions.append(pred)

        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)

        # Apply mask
        ensemble_pred = ensemble_pred * batch.possible_dose_mask

        # Denormalize if needed
        if normalize:
            ensemble_pred = ensemble_pred * data_loader.DOSE_PRESCRIPTION

        # Save
        dose_pred = np.squeeze(ensemble_pred)
        dose_to_save = sparse_vector_function(dose_pred)
        dose_df = pd.DataFrame(
            data=dose_to_save["data"].squeeze(),
            index=dose_to_save["indices"].squeeze(),
            columns=["data"]
        )
        (patient_id,) = batch.patient_list
        dose_df.to_csv(prediction_dir / f"{patient_id}.csv")

    print(f"Ensemble predictions saved to: {prediction_dir}")
    return prediction_dir


def main():
    parser = argparse.ArgumentParser(description='Ensemble prediction for OpenKBP')
    parser.add_argument('--models', nargs='+', required=True, help='Model names to ensemble')
    parser.add_argument('--epoch', type=int, required=True, help='Epoch number to load')
    parser.add_argument('--output', type=str, required=True, help='Output name for ensemble results')
    parser.add_argument('--no-normalize', action='store_true', help='If models were trained without normalization')
    parser.add_argument('--skip-eval', action='store_true', help='Skip evaluation after prediction')
    args = parser.parse_args()

    # Define directories
    primary_directory = Path(__file__).parent.resolve()

    if Path("/workspace").exists():
        results_dir = Path("/workspace/results")
    else:
        results_dir = primary_directory.parent / "results"

    validation_data_dir = primary_directory / "provided-data" / "validation-pats"

    if not validation_data_dir.exists():
        print(f"ERROR: Validation data not found at {validation_data_dir}")
        return 1

    print(f"=" * 60)
    print(f"OpenKBP Ensemble Prediction")
    print(f"Models: {', '.join(args.models)}")
    print(f"Output: {args.output}")
    print(f"=" * 60)

    # Run ensemble prediction
    prediction_dir = ensemble_predict(
        model_names=args.models,
        results_dir=results_dir,
        validation_data_dir=validation_data_dir,
        output_name=args.output,
        epoch=args.epoch,
        normalize=not args.no_normalize,
    )

    # Evaluate
    if not args.skip_eval:
        print(f"\nEvaluating ensemble predictions...")
        hold_out_plan_paths = get_paths(validation_data_dir)
        data_loader_eval = DataLoader(hold_out_plan_paths, normalize=False)  # Ground truth is not normalized
        prediction_paths = get_paths(prediction_dir, extension="csv")
        prediction_loader = DataLoader(prediction_paths, normalize=False)  # Saved predictions are denormalized
        dose_evaluator = DoseEvaluator(data_loader_eval, prediction_loader)

        dose_evaluator.evaluate()
        dose_score, dvh_score = dose_evaluator.get_scores()

        print(f"\n{'=' * 60}")
        print(f"ENSEMBLE RESULTS: {args.output}")
        print(f"{'=' * 60}")
        print(f"DVH Score:  {dvh_score:.3f}")
        print(f"Dose Score: {dose_score:.3f}")
        print(f"{'=' * 60}")

        # Save results
        ensemble_dir = results_dir / args.output
        results_file = ensemble_dir / "results.json"
        results_data = {
            "model": args.output,
            "type": "ensemble",
            "source_models": args.models,
            "epoch": args.epoch,
            "normalize": not args.no_normalize,
            "dvh_score": float(dvh_score),
            "dose_score": float(dose_score),
            "timestamp": datetime.now().isoformat(),
        }
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"Results saved: {results_file}")

        # Create submission zip
        submission_dir = results_dir / "submissions"
        submission_dir.mkdir(parents=True, exist_ok=True)
        zip_path = shutil.make_archive(
            str(submission_dir / args.output), "zip",
            prediction_dir
        )
        print(f"Submission saved: {zip_path}")

    return 0


if __name__ == "__main__":
    exit(main())
