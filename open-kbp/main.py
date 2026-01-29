import shutil
from pathlib import Path

from provided_code import DataLoader, DoseEvaluator, PredictionModel, get_paths

if __name__ == "__main__":

    # Configuration
    num_filters = 1   # Recommend 64+ for real training
    num_epochs = 2    # Recommend 100-200 for real training
    test_time = False # Set True to evaluate on test set

    prediction_name = f"{num_filters}filter_{num_epochs}epoch"

    # Define project directories
    primary_directory = Path().resolve()  # directory where everything is stored
    provided_data_dir = primary_directory / "provided-data"
    training_data_dir = provided_data_dir / "train-pats"
    validation_data_dir = provided_data_dir / "validation-pats"
    testing_data_dir = provided_data_dir / "test-pats"
    results_dir = primary_directory.parent / "results"  # OUTSIDE repo to preserve on re-clone

    # Prepare the data directory
    training_plan_paths = get_paths(training_data_dir)  # gets the path of each plan's directory

    # Train a model
    data_loader_train = DataLoader(training_plan_paths)
    dose_prediction_model_train = PredictionModel(data_loader_train, results_dir, prediction_name, "train", num_filters)
    dose_prediction_model_train.train_model(num_epochs, save_frequency=1, keep_model_history=20)

    # Define hold out set
    hold_out_data_dir = validation_data_dir if test_time is False else testing_data_dir
    stage_name, _ = hold_out_data_dir.stem.split("-")
    hold_out_plan_paths = get_paths(hold_out_data_dir)

    # Predict dose for the held out set
    data_loader_hold_out = DataLoader(hold_out_plan_paths)
    dose_prediction_model_hold_out = PredictionModel(data_loader_hold_out, results_dir, prediction_name, stage_name, num_filters)
    dose_prediction_model_hold_out.predict_dose(epoch=num_epochs)

    # Evaluate dose metrics
    data_loader_hold_out_eval = DataLoader(hold_out_plan_paths)
    prediction_paths = get_paths(dose_prediction_model_hold_out.prediction_dir, extension="csv")
    hold_out_prediction_loader = DataLoader(prediction_paths)
    dose_evaluator = DoseEvaluator(data_loader_hold_out_eval, hold_out_prediction_loader)

    # print out scores if data was left for a hold out set
    if not data_loader_hold_out_eval.patient_paths:
        print("No patient information was given to calculate metrics")
    else:
        dose_evaluator.evaluate()
        dose_score, dvh_score = dose_evaluator.get_scores()
        print(f"For this out-of-sample test on {stage_name}:\n\tthe DVH score is {dvh_score:.3f}\n\tthe dose score is {dose_score:.3f}")

    # Zip dose to submit
    submission_dir = results_dir / "submissions"
    submission_dir.mkdir(exist_ok=True)
    shutil.make_archive(str(submission_dir / prediction_name), "zip", dose_prediction_model_hold_out.prediction_dir)
