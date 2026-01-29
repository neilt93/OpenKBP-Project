"""Run predictions only (no training) for a saved model"""
import shutil
from pathlib import Path

from provided_code import DataLoader, DoseEvaluator, PredictionModel, get_paths

if __name__ == "__main__":
    # Configuration - set to match existing model
    num_filters = 16
    num_epochs = 5

    prediction_name = f"{num_filters}filter_{num_epochs}epoch"

    # Define project directories
    primary_directory = Path().resolve()
    provided_data_dir = primary_directory / "provided-data"
    validation_data_dir = provided_data_dir / "validation-pats"
    results_dir = primary_directory.parent / "results"

    # Get validation paths
    hold_out_plan_paths = get_paths(validation_data_dir)
    print(f"Running predictions for {prediction_name} on {len(hold_out_plan_paths)} validation patients")

    # Predict dose
    data_loader_hold_out = DataLoader(hold_out_plan_paths)
    dose_prediction_model = PredictionModel(data_loader_hold_out, results_dir, prediction_name, "validation", num_filters)
    dose_prediction_model.predict_dose(epoch=num_epochs)

    # Evaluate
    data_loader_eval = DataLoader(hold_out_plan_paths)
    prediction_paths = get_paths(dose_prediction_model.prediction_dir, extension="csv")
    prediction_loader = DataLoader(prediction_paths)
    dose_evaluator = DoseEvaluator(data_loader_eval, prediction_loader)

    dose_evaluator.evaluate()
    dose_score, dvh_score = dose_evaluator.get_scores()
    print(f"\n=== Results for {prediction_name} ===")
    print(f"DVH Score:  {dvh_score:.3f}")
    print(f"Dose Score: {dose_score:.3f}")
