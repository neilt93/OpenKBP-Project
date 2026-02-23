#!/usr/bin/env python3
"""Step 3: Evaluate DVH and dose scores for all conditions.

Computes metrics by comparing predicted dose distributions against ground truth
for baseline and each perturbed condition. Runs locally (no GPU needed).

Usage:
    cd open-kbp-modified/
    python openkbp_hn_robustness/evaluate_metrics.py [--config CONFIG]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add project root to path
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from provided_code.data_loader import DataLoader
from provided_code.dose_evaluation_class import DoseEvaluator
from provided_code.utils import get_paths


def evaluate_condition(condition_name: str, prediction_dir: Path,
                       validation_paths: list, config: dict) -> dict:
    """Evaluate a single condition against ground truth.

    Follows the pattern from runpod_train.py lines 190-196:
    - Reference loader: normalize=False (ground truth in Gy)
    - Prediction loader: normalize=False (predictions saved in Gy)
    """
    prediction_csvs = sorted(get_paths(prediction_dir, extension="csv"))
    if not prediction_csvs:
        print(f"  No predictions found in {prediction_dir}")
        return {}

    print(f"  Found {len(prediction_csvs)} prediction files")

    # Create data loaders (no normalization — both in Gy)
    ref_loader = DataLoader(validation_paths, normalize=False, cache_data=True)
    pred_loader = DataLoader(prediction_csvs, normalize=False, cache_data=False)

    # Run evaluation
    evaluator = DoseEvaluator(ref_loader, pred_loader)
    evaluator.evaluate()
    dose_score, dvh_score = evaluator.get_scores()

    print(f"  Dose Score: {dose_score:.3f} Gy")
    print(f"  DVH Score:  {dvh_score:.3f}")

    # Extract per-structure DVH details
    dvh_errors = np.abs(evaluator.reference_dvh_metrics_df - evaluator.prediction_dvh_metrics_df)

    # Per-structure mean errors
    structure_errors = {}
    for col in dvh_errors.columns:
        metric, structure = col
        key = f"{structure}_{metric}"
        structure_errors[key] = float(dvh_errors[col].mean())

    return {
        "condition": condition_name,
        "dose_score": float(dose_score),
        "dvh_score": float(dvh_score),
        "n_patients": len(prediction_csvs),
        "per_structure": structure_errors,
        "per_patient_dose": evaluator.dose_errors.to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate robustness metrics")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Only evaluate specific conditions (e.g., baseline P1_noise/L1)")
    args = parser.parse_args()

    # Load config
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = script_dir / "configs" / "default.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Set up paths
    validation_dir = project_root / config["paths"]["validation_data"]
    predictions_root = project_root / config["paths"]["predictions"]
    metrics_dir = project_root / config["paths"]["metrics"]
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Build patient list and validation paths
    start = config["patients"]["start"]
    end = config["patients"]["end"]
    patient_ids = [f"pt_{i}" for i in range(start, end + 1)]
    validation_paths = sorted([
        validation_dir / pid for pid in patient_ids
        if (validation_dir / pid).exists()
    ])

    print(f"Evaluating with {len(validation_paths)} reference patients")

    all_results = []

    # 1. Baseline
    if args.conditions is None or "baseline" in args.conditions:
        baseline_dir = predictions_root / "baseline"
        if baseline_dir.exists():
            print(f"\n{'=' * 60}")
            print("Evaluating: baseline")
            print(f"{'=' * 60}")
            result = evaluate_condition("baseline", baseline_dir, validation_paths, config)
            if result:
                all_results.append(result)
        else:
            print(f"Baseline predictions not found at {baseline_dir}")

    # 2. Perturbed conditions
    for p_name, p_config in config["perturbations"].items():
        if not p_config.get("enabled", True):
            continue

        for level in p_config["levels"]:
            condition = f"{p_name}/{level}"

            if args.conditions is not None and condition not in args.conditions:
                continue

            pred_dir = predictions_root / p_name / level
            if not pred_dir.exists():
                print(f"\nSkipping {condition} (predictions not found)")
                continue

            print(f"\n{'=' * 60}")
            print(f"Evaluating: {condition}")
            print(f"{'=' * 60}")
            result = evaluate_condition(condition, pred_dir, validation_paths, config)
            if result:
                all_results.append(result)

    if not all_results:
        print("No results to summarize!")
        return

    # Compute deltas vs baseline
    baseline_result = next((r for r in all_results if r["condition"] == "baseline"), None)

    summary_rows = []
    for result in all_results:
        row = {
            "perturbation": result["condition"],
            "dose_score": result["dose_score"],
            "dvh_score": result["dvh_score"],
            "n_patients": result["n_patients"],
        }
        if baseline_result and result["condition"] != "baseline":
            row["delta_dose"] = result["dose_score"] - baseline_result["dose_score"]
            row["delta_dvh"] = result["dvh_score"] - baseline_result["dvh_score"]
            if baseline_result["dose_score"] > 0:
                row["delta_dose_pct"] = 100.0 * row["delta_dose"] / baseline_result["dose_score"]
            if baseline_result["dvh_score"] > 0:
                row["delta_dvh_pct"] = 100.0 * row["delta_dvh"] / baseline_result["dvh_score"]
        summary_rows.append(row)

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = metrics_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")

    # Save detailed results as JSON
    details_path = metrics_dir / "detailed_results.json"
    with open(details_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Detailed results saved to {details_path}")

    # Save per-patient results
    per_patient_dir = metrics_dir / "per_patient"
    per_patient_dir.mkdir(parents=True, exist_ok=True)
    for result in all_results:
        condition_name = result["condition"].replace("/", "_")
        with open(per_patient_dir / f"{condition_name}.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

    # Print summary table
    print(f"\n{'=' * 90}")
    print(f"{'Condition':<25} {'Dose (Gy)':>10} {'DVH':>10} {'dDose':>10} {'dDVH':>10} {'dDose%':>10} {'dDVH%':>10}")
    print(f"{'=' * 90}")
    for row in summary_rows:
        line = f"{row['perturbation']:<25} {row['dose_score']:>10.3f} {row['dvh_score']:>10.3f}"
        if "delta_dose" in row:
            line += f" {row['delta_dose']:>+10.3f} {row['delta_dvh']:>+10.3f}"
            line += f" {row.get('delta_dose_pct', 0):>+10.1f} {row.get('delta_dvh_pct', 0):>+10.1f}"
        else:
            line += f" {'—':>10} {'—':>10} {'—':>10} {'—':>10}"
        print(line)
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
