"""
Calculate comprehensive DVH metrics for radiotherapy dose evaluation.

Computes for PTVs: D99, D95, D1
Computes for OARs: D_0.1cc, V30, V50

Usage:
    python calculate_dvh_metrics.py --model path/to/model.keras --patient-ids pt_201 pt_202
    python calculate_dvh_metrics.py --model path/to/model.keras --all  # All validation patients
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

from provided_code import DataLoader, get_paths
from provided_code.network_architectures import InstanceNormalization


class ComprehensiveDVHCalculator:
    """Calculate comprehensive DVH metrics for dose evaluation."""

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.rois = data_loader.rois

        # Define metrics for each structure type
        self.ptv_metrics = ["D_99", "D_95", "D_1"]
        self.oar_metrics = ["D_0.1_cc", "V_30", "V_50"]

        # Results storage
        self.results = []

    def calculate_for_patient(
        self,
        patient_id: str,
        dose_true: np.ndarray,
        dose_pred: np.ndarray,
        structure_masks: np.ndarray,
        voxel_dimensions: np.ndarray,
    ) -> Dict:
        """Calculate all DVH metrics for one patient."""
        voxel_size = np.prod(voxel_dimensions)
        voxels_in_01cc = max(1, int(100 / voxel_size))

        patient_results = {"patient_id": patient_id}

        # Calculate metrics for each structure
        for roi_idx, roi_name in enumerate(self.data_loader.full_roi_list):
            roi_mask = structure_masks[0, :, :, :, roi_idx].astype(bool).flatten()

            if not np.any(roi_mask):
                continue  # Skip if structure not contoured

            # Extract doses for this ROI
            dose_true_roi = dose_true[roi_mask]
            dose_pred_roi = dose_pred[roi_mask]

            # Determine structure type
            is_target = roi_name in self.rois["targets"]
            metrics_to_calc = self.ptv_metrics if is_target else self.oar_metrics

            # Calculate each metric
            for metric in metrics_to_calc:
                true_val = self._calculate_metric(
                    metric, dose_true_roi, voxels_in_01cc
                )
                pred_val = self._calculate_metric(
                    metric, dose_pred_roi, voxels_in_01cc
                )

                patient_results[f"{roi_name}_{metric}_true"] = true_val
                patient_results[f"{roi_name}_{metric}_pred"] = pred_val
                patient_results[f"{roi_name}_{metric}_diff"] = abs(true_val - pred_val)

        return patient_results

    def _calculate_metric(
        self, metric: str, roi_dose: np.ndarray, voxels_in_01cc: int
    ) -> float:
        """Calculate a single DVH metric."""
        if metric == "D_99":
            return np.percentile(roi_dose, 1)
        elif metric == "D_95":
            return np.percentile(roi_dose, 5)
        elif metric == "D_1":
            return np.percentile(roi_dose, 99)
        elif metric == "D_0.1_cc":
            roi_size = len(roi_dose)
            fractional_volume = 100 - (voxels_in_01cc / roi_size * 100)
            return np.percentile(roi_dose, fractional_volume)
        elif metric == "V_30":
            # Percentage of ROI receiving >= 30 Gy
            return (np.sum(roi_dose >= 30.0) / len(roi_dose)) * 100
        elif metric == "V_50":
            # Percentage of ROI receiving >= 50 Gy
            return (np.sum(roi_dose >= 50.0) / len(roi_dose)) * 100
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def add_result(self, result: Dict):
        """Add a patient's results to the collection."""
        self.results.append(result)

    def get_summary_table(self) -> pd.DataFrame:
        """Create summary table aggregating all patients."""
        if not self.results:
            return pd.DataFrame()

        df = pd.DataFrame(self.results)

        # Calculate mean values across all patients
        summary = {"Structure": [], "Metric": [], "Ground Truth": [], "Predicted": [], "Mean Abs Diff": []}

        for roi_name in self.data_loader.full_roi_list:
            is_target = roi_name in self.rois["targets"]
            metrics = self.ptv_metrics if is_target else self.oar_metrics

            for metric in metrics:
                true_col = f"{roi_name}_{metric}_true"
                pred_col = f"{roi_name}_{metric}_pred"
                diff_col = f"{roi_name}_{metric}_diff"

                if true_col in df.columns:
                    summary["Structure"].append(roi_name)
                    summary["Metric"].append(metric)
                    summary["Ground Truth"].append(df[true_col].mean())
                    summary["Predicted"].append(df[pred_col].mean())
                    summary["Mean Abs Diff"].append(df[diff_col].mean())

        return pd.DataFrame(summary)


def generate_predictions(model, data_loader: DataLoader) -> Dict[str, np.ndarray]:
    """Generate dose predictions for all patients."""
    predictions = {}

    print("Generating predictions...")
    for batch in data_loader.get_batches():
        patient_id = batch.patient_list[0]

        # Run model
        ct = tf.constant(batch.ct, dtype=tf.float32)
        masks = tf.constant(batch.structure_masks, dtype=tf.float32)
        dose_pred = model([ct, masks], training=False)

        # Denormalize dose (model outputs normalized by 70 Gy)
        dose_pred = dose_pred.numpy().flatten() * 70.0
        dose_true = batch.dose.flatten() * 70.0

        predictions[patient_id] = {
            "dose_pred": dose_pred,
            "dose_true": dose_true,
            "structure_masks": batch.structure_masks,
            "voxel_dimensions": batch.voxel_dimensions[0],
        }

        print(f"  {patient_id}: predicted")

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Calculate comprehensive DVH metrics")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.keras)")
    parser.add_argument("--data-dir", type=str, default="provided-data/validation-pats",
                        help="Path to patient data directory")
    parser.add_argument("--patient-ids", type=str, nargs="+", help="Specific patient IDs to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate all patients in data-dir")
    parser.add_argument("--output", type=str, default="dvh_metrics.csv", help="Output CSV file")

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}")
    model = load_model(
        args.model,
        custom_objects={"InstanceNormalization": InstanceNormalization},
        compile=False,
    )

    # Load data
    script_dir = Path(__file__).parent
    data_dir = script_dir / args.data_dir

    if args.all:
        patient_paths = get_paths(data_dir)
    elif args.patient_ids:
        patient_paths = [data_dir / pid for pid in args.patient_ids]
    else:
        raise ValueError("Must specify either --all or --patient-ids")

    print(f"Loading {len(patient_paths)} patients from {data_dir}")

    data_loader = DataLoader(
        patient_paths,
        batch_size=1,
        normalize=True,
        cache_data=True,
    )
    data_loader.set_mode("training_model")

    # Calculate metrics
    calculator = ComprehensiveDVHCalculator(data_loader)
    predictions = generate_predictions(model, data_loader)

    print("\nCalculating DVH metrics...")
    for patient_id, pred_data in predictions.items():
        result = calculator.calculate_for_patient(
            patient_id,
            pred_data["dose_true"],
            pred_data["dose_pred"],
            pred_data["structure_masks"],
            pred_data["voxel_dimensions"],
        )
        calculator.add_result(result)
        print(f"  {patient_id}: metrics calculated")

    # Generate summary table
    summary_df = calculator.get_summary_table()

    # Format and display
    print("\n" + "="*80)
    print("DVH METRICS SUMMARY (averaged across all patients)")
    print("="*80)

    # Display PTVs
    print("\nPLANNING TARGET VOLUMES (PTVs)")
    print("-" * 80)
    ptv_df = summary_df[summary_df["Structure"].str.contains("PTV")]
    print(ptv_df.to_string(index=False))

    # Display OARs
    print("\nORGANS AT RISK (OARs)")
    print("-" * 80)
    oar_df = summary_df[~summary_df["Structure"].str.contains("PTV")]
    print(oar_df.to_string(index=False))

    # Save to CSV
    output_file = Path(args.output)
    summary_df.to_csv(output_file, index=False, float_format="%.2f")
    print(f"\n✓ Results saved to {output_file}")

    # Also save detailed per-patient results
    detailed_file = output_file.with_stem(f"{output_file.stem}_detailed")
    pd.DataFrame(calculator.results).to_csv(detailed_file, index=False, float_format="%.2f")
    print(f"✓ Detailed per-patient results saved to {detailed_file}")


if __name__ == "__main__":
    main()
