#!/usr/bin/env python3
"""
RunPod Training Script for OpenKBP
Usage: python runpod_train.py [--filters N] [--epochs N]

Setup on RunPod:
1. Start a GPU pod (RTX 3090/4090 recommended)
2. Clone repo: git clone https://github.com/neilt93/OpenKBP-Project.git /workspace/OpenKBP-Project
3. Get data: Use runpodctl or upload provided-data.zip
4. Run: cd /workspace/OpenKBP-Project/open-kbp-modified && pip install -r requirements.txt && python runpod_train.py
"""
import argparse
import json
import shutil
import os
import random
from pathlib import Path
from datetime import datetime

import numpy as np

# Ensure TensorFlow uses GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from provided_code import DataLoader, DoseEvaluator, PredictionModel, get_paths


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(description='Train OpenKBP dose prediction model')
    parser.add_argument('--filters', type=int, default=64, help='Number of filters (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
    parser.add_argument('--save-freq', type=int, default=10, help='Save model every N epochs (default: 10)')
    parser.add_argument('--keep-history', type=int, default=5, help='Keep last N models (default: 5)')
    parser.add_argument('--predict-only', action='store_true', help='Skip training, only run predictions')
    parser.add_argument('--eval-only', action='store_true', help='Skip training and predictions, only run evaluation')
    parser.add_argument('--use-se', action='store_true', help='Enable Squeeze-and-Excitation attention blocks')
    parser.add_argument('--use-dvh', action='store_true', help='Enable DVH-aware loss function')
    parser.add_argument('--dvh-weight', type=float, default=0.1, help='DVH loss weight (default: 0.1)')
    parser.add_argument('--use-aug', action='store_true', help='Enable data augmentation (flips, intensity scaling)')
    # Geometric / perturbation augmentation (robustness retraining). Any of these turns on
    # the numpy augmentation path (includes flips+intensity); needs --no-jit (XLA off).
    parser.add_argument('--aug-translate', type=float, default=0.0, help='Max translation fraction per axis (e.g. 0.1)')
    parser.add_argument('--aug-rotate', type=float, default=0.0, help='Max in-plane (axial) rotation in degrees (e.g. 10)')
    parser.add_argument('--aug-scale', type=float, default=0.0, help='Max in-plane scale delta (e.g. 0.1 = +/-10%%)')
    parser.add_argument('--aug-elastic', type=float, default=0.0, help='Elastic deformation strength in voxels (e.g. 3)')
    parser.add_argument('--aug-noise', type=float, default=0.0, help='CT Gaussian noise std, normalized units (e.g. 0.02)')
    # Inject pre-generated perturbed-CT sets into the training group (robustness retraining)
    parser.add_argument('--inject-perturbed', type=str, default=None, help='Root of pre-generated perturbed-CT sets to inject')
    parser.add_argument('--inject-glob', type=str, default='*/*/{pid}/ct.csv', help='Glob for perturbed CTs under --inject-perturbed ({pid}=patient id)')
    parser.add_argument('--inject-families', nargs='*', default=None, help='Only inject variants whose tag contains these (e.g. P2 P4)')
    parser.add_argument('--inject-levels', nargs='*', default=None, help='Only inject variants whose tag contains these (e.g. L3 L4)')
    parser.add_argument('--inject-max-per-patient', type=int, default=None, help='Cap injected variants per clean patient (clean:perturbed ratio control)')
    parser.add_argument('--inject-out', type=str, default=None, help='Dir for composed injected patients (default: <inject-perturbed>/_injected)')
    parser.add_argument('--inject-reuse-existing', action='store_true', help='Use perturbed dirs directly if they already have dose+masks')
    parser.add_argument('--no-masked-loss', action='store_true', help='Disable masked MAE loss (use unweighted MAE)')
    parser.add_argument('--ptv-weight', type=float, default=2.0, help='Extra weight on PTV voxels (default: 2.0, 0=no weighting)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility (for ensemble training)')
    parser.add_argument('--no-normalize', action='store_true', help='Disable CT/dose normalization (for testing)')
    parser.add_argument('--no-mixed-precision', action='store_true', help='Disable mixed precision (float16) training')
    parser.add_argument('--no-jit', action='store_true', help='Disable XLA JIT compilation')
    parser.add_argument('--no-cache', action='store_true', help='Disable data caching (match original behavior)')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size (default: 2, reduce if OOM)')
    parser.add_argument('--precomputed', type=str, default=None, help='Path to precomputed train_data.npz for instant loading')
    args = parser.parse_args()

    # Set random seeds if specified (for ensemble training)
    if args.seed is not None:
        set_seeds(args.seed)
        print(f"Random seed set to {args.seed}")

    num_filters = args.filters
    num_epochs = args.epochs

    # Build model name based on features
    name_parts = [f"{num_filters}filter", f"{num_epochs}epoch"]
    if args.use_se:
        name_parts.append("SE")
    if args.use_dvh:
        name_parts.append(f"DVH{args.dvh_weight}")
    if args.use_aug:
        name_parts.append("AUG")
    # Geometric augmentation strengths -> numpy augmentation path
    aug_params = {}
    if any([args.aug_translate, args.aug_rotate, args.aug_scale, args.aug_elastic, args.aug_noise]):
        aug_params = dict(
            translate_frac=args.aug_translate,
            rotate_deg=args.aug_rotate,
            scale_range=args.aug_scale,
            elastic_alpha=args.aug_elastic,
            noise_std=args.aug_noise,
        )
        name_parts.append("AUGGEO")
    if args.inject_perturbed:
        name_parts.append("INJ")
    if not args.no_masked_loss:
        name_parts.append(f"MASK_PTV{args.ptv_weight}")
    if not args.no_normalize:
        name_parts.append("NORM")
    if args.seed is not None:
        name_parts.append(f"seed{args.seed}")
    prediction_name = "_".join(name_parts)

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
            if not args.no_mixed_precision:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                print("Mixed precision (float16) enabled for faster training")
            else:
                print("Mixed precision disabled (using float32)")
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

    # Inject pre-generated perturbed-CT sets (perturbed CT + clean dose/masks) so the
    # model learns to predict the correct dose despite a corrupted CT. (Only when
    # training — no point building injected dirs for --predict-only / --eval-only.)
    if args.inject_perturbed and not (args.predict_only or args.eval_only):
        from provided_code.inject_perturbed import build_injected_set
        inject_out = Path(args.inject_out) if args.inject_out else Path(args.inject_perturbed) / "_injected"
        # Leakage guard: the validation patients must never enter training. Derive their
        # ids from the hold-out dir and forbid them in the injection.
        holdout_ids = {p.name for p in get_paths(validation_data_dir)} if validation_data_dir.exists() else set()
        injected = build_injected_set(
            training_data_dir, Path(args.inject_perturbed), inject_out,
            glob=args.inject_glob, families=args.inject_families, levels=args.inject_levels,
            max_per_patient=args.inject_max_per_patient, holdout_ids=holdout_ids,
            reuse_existing=args.inject_reuse_existing,
        )
        print(f"Injected {len(injected)} perturbed patients into the training group "
              f"(families={args.inject_families or 'all'})")
        training_plan_paths = list(training_plan_paths) + injected
        print(f"Training group is now {len(training_plan_paths)} patients")

    # Train model (unless predict-only or eval-only)
    if not args.predict_only and not args.eval_only:
        features = []
        if args.use_se:
            features.append("SE blocks")
        if args.use_dvh:
            features.append(f"DVH loss (weight={args.dvh_weight})")
        if args.use_aug:
            features.append("augmentation")
        if not args.no_masked_loss:
            features.append(f"masked loss (PTV weight={args.ptv_weight})")
        feature_str = f" with {', '.join(features)}" if features else ""
        print(f"\nStarting training: {num_filters} filters, {num_epochs} epochs{feature_str}")

        precomputed_path = Path(args.precomputed) if args.precomputed else None
        data_loader_train = DataLoader(training_plan_paths, batch_size=args.batch_size, normalize=not args.no_normalize, cache_data=not args.no_cache, precomputed_path=precomputed_path)
        dose_prediction_model_train = PredictionModel(
            data_loader_train, results_dir, prediction_name, "train", num_filters,
            use_se_blocks=args.use_se,
            use_dvh_loss=args.use_dvh,
            dvh_weight=args.dvh_weight,
            use_augmentation=args.use_aug or bool(aug_params),
            use_jit=not args.no_jit,
            use_masked_loss=not args.no_masked_loss,
            ptv_weight=args.ptv_weight,
            aug_params=aug_params,
        )
        dose_prediction_model_train.train_model(
            epochs=num_epochs,
            save_frequency=args.save_freq,
            keep_model_history=args.keep_history
        )
    elif args.eval_only:
        print("Skipping training (--eval-only)")
    else:
        print("Skipping training (--predict-only)")

    # Run predictions on validation set (unless eval-only)
    hold_out_plan_paths = get_paths(validation_data_dir)
    prediction_dir = results_dir / prediction_name / "validation-predictions"

    if not args.eval_only:
        print(f"\nRunning predictions on validation set...")
        data_loader_hold_out = DataLoader(hold_out_plan_paths, normalize=not args.no_normalize, cache_data=not args.no_cache)
        dose_prediction_model_hold_out = PredictionModel(
            data_loader_hold_out, results_dir, prediction_name, "validation", num_filters,
            use_se_blocks=args.use_se,
            use_jit=not args.no_jit,
        )
        dose_prediction_model_hold_out.predict_dose(epoch=num_epochs)
        prediction_dir = dose_prediction_model_hold_out.prediction_dir
    else:
        print("Skipping predictions (--eval-only)")

    # Evaluate
    # IMPORTANT: Evaluation must use normalize=False because:
    # - Predictions are saved denormalized (in Gy)
    # - Ground truth should also be in Gy for fair comparison
    print(f"\nEvaluating predictions...")
    data_loader_hold_out_eval = DataLoader(hold_out_plan_paths, normalize=False)
    prediction_paths = get_paths(prediction_dir, extension="csv")
    hold_out_prediction_loader = DataLoader(prediction_paths, normalize=False)
    dose_evaluator = DoseEvaluator(data_loader_hold_out_eval, hold_out_prediction_loader)

    dose_evaluator.evaluate()
    dose_score, dvh_score = dose_evaluator.get_scores()

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {prediction_name}")
    print(f"{'=' * 60}")
    print(f"DVH Score:  {dvh_score:.3f}")
    print(f"Dose Score: {dose_score:.3f}")
    print(f"{'=' * 60}")

    # Save results to JSON file
    model_results_dir = results_dir / prediction_name
    results_file = model_results_dir / "results.json"
    results_data = {
        "model": prediction_name,
        "filters": num_filters,
        "epochs": num_epochs,
        "use_se_blocks": args.use_se,
        "use_dvh_loss": args.use_dvh,
        "dvh_weight": args.dvh_weight if args.use_dvh else None,
        "use_augmentation": args.use_aug,
        "aug_geometric": aug_params or None,
        "inject_perturbed": args.inject_perturbed,
        "inject_families": args.inject_families,
        "use_masked_loss": not args.no_masked_loss,
        "ptv_weight": args.ptv_weight if not args.no_masked_loss else None,
        "normalize": not args.no_normalize,
        "seed": args.seed,
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
        str(submission_dir / prediction_name), "zip",
        prediction_dir
    )
    print(f"Submission saved: {zip_path}")

    return 0


if __name__ == "__main__":
    exit(main())
