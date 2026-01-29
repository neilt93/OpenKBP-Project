#!/usr/bin/env python3
"""
RunPod Training Script for OpenKBP
Usage: python runpod_train.py [--filters N] [--epochs N]

Setup on RunPod:
1. Start a GPU pod (RTX 3090/4090 recommended)
2. Clone repo: git clone https://github.com/neilt93/OpenKBP-Project.git /workspace/OpenKBP-Project
3. Get data: Use runpodctl or upload provided-data.zip
4. Run: cd /workspace/OpenKBP-Project/open-kbp && pip install -r requirements.txt && python runpod_train.py
"""
import argparse
import shutil
import os
from pathlib import Path

# Ensure TensorFlow uses GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from provided_code import DataLoader, DoseEvaluator, PredictionModel, get_paths


def main():
    parser = argparse.ArgumentParser(description='Train OpenKBP dose prediction model')
    parser.add_argument('--filters', type=int, default=64, help='Number of filters (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--save-freq', type=int, default=10, help='Save model every N epochs (default: 10)')
    parser.add_argument('--keep-history', type=int, default=5, help='Keep last N models (default: 5)')
    parser.add_argument('--predict-only', action='store_true', help='Skip training, only run predictions')
    args = parser.parse_args()

    num_filters = args.filters
    num_epochs = args.epochs
    prediction_name = f"{num_filters}filter_{num_epochs}epoch"

    print(f"=" * 60)
    print(f"OpenKBP Training - {prediction_name}")
    print(f"=" * 60)

    # Configure GPU with optimizations
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPU detected: {gpus}")
            # Allow memory growth to avoid OOM
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # Enable mixed precision for ~2x speedup on RTX 3090/4090
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("Mixed precision (float16) enabled for faster training")
        else:
            print("WARNING: No GPU detected, training will be slow!")
    except Exception as e:
        print(f"GPU config failed: {e}")

    # Define project directories - works for both local and RunPod
    primary_directory = Path(__file__).parent.resolve()

    # Check if we're on RunPod (workspace exists)
    if Path("/workspace").exists():
        results_dir = Path("/workspace/results")
    else:
        results_dir = primary_directory.parent / "results"

    provided_data_dir = primary_directory / "provided-data"
    training_data_dir = provided_data_dir / "train-pats"
    validation_data_dir = provided_data_dir / "validation-pats"

    # Verify data exists
    if not training_data_dir.exists():
        print(f"ERROR: Training data not found at {training_data_dir}")
        print("Please upload provided-data folder or extract provided-data.zip")
        return 1

    training_plan_paths = get_paths(training_data_dir)
    print(f"Found {len(training_plan_paths)} training patients")

    # Train model (unless predict-only)
    if not args.predict_only:
        print(f"\nStarting training: {num_filters} filters, {num_epochs} epochs")
        data_loader_train = DataLoader(training_plan_paths)
        dose_prediction_model_train = PredictionModel(
            data_loader_train, results_dir, prediction_name, "train", num_filters
        )
        dose_prediction_model_train.train_model(
            epochs=num_epochs,
            save_frequency=args.save_freq,
            keep_model_history=args.keep_history
        )
    else:
        print("Skipping training (--predict-only)")

    # Run predictions on validation set
    print(f"\nRunning predictions on validation set...")
    hold_out_plan_paths = get_paths(validation_data_dir)
    data_loader_hold_out = DataLoader(hold_out_plan_paths)
    dose_prediction_model_hold_out = PredictionModel(
        data_loader_hold_out, results_dir, prediction_name, "validation", num_filters
    )
    dose_prediction_model_hold_out.predict_dose(epoch=num_epochs)

    # Evaluate
    print(f"\nEvaluating predictions...")
    data_loader_hold_out_eval = DataLoader(hold_out_plan_paths)
    prediction_paths = get_paths(dose_prediction_model_hold_out.prediction_dir, extension="csv")
    hold_out_prediction_loader = DataLoader(prediction_paths)
    dose_evaluator = DoseEvaluator(data_loader_hold_out_eval, hold_out_prediction_loader)

    dose_evaluator.evaluate()
    dose_score, dvh_score = dose_evaluator.get_scores()

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {prediction_name}")
    print(f"{'=' * 60}")
    print(f"DVH Score:  {dvh_score:.3f}")
    print(f"Dose Score: {dose_score:.3f}")
    print(f"{'=' * 60}")

    # Create submission zip
    submission_dir = results_dir / "submissions"
    submission_dir.mkdir(parents=True, exist_ok=True)
    zip_path = shutil.make_archive(
        str(submission_dir / prediction_name), "zip",
        dose_prediction_model_hold_out.prediction_dir
    )
    print(f"Submission saved: {zip_path}")

    return 0


if __name__ == "__main__":
    exit(main())
