import os
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from provided_code.data_loader import DataLoader
from provided_code.network_architectures import DefineDoseFromCT
from provided_code.utils import get_paths, sparse_vector_function


def augment_batch(ct: np.ndarray, structure_masks: np.ndarray, dose: np.ndarray,
                   flip_prob: float = 0.5, intensity_scale: float = 0.1) -> tuple:
    """
    Apply data augmentation to a training batch.

    Args:
        ct: (batch, D, H, W, 1) CT images
        structure_masks: (batch, D, H, W, 10) ROI masks
        dose: (batch, D, H, W, 1) dose distributions
        flip_prob: probability of flipping along each axis
        intensity_scale: max CT intensity scaling factor (±scale)

    Returns:
        Augmented (ct, structure_masks, dose) tuple
    """
    ct = ct.copy()
    structure_masks = structure_masks.copy()
    dose = dose.copy()

    batch_size = ct.shape[0]

    for b in range(batch_size):
        # Random left-right flip (axis 2 = left-right in patient coords)
        if np.random.random() < flip_prob:
            ct[b] = np.flip(ct[b], axis=2)
            structure_masks[b] = np.flip(structure_masks[b], axis=2)
            dose[b] = np.flip(dose[b], axis=2)

        # Random anterior-posterior flip (axis 1)
        if np.random.random() < flip_prob:
            ct[b] = np.flip(ct[b], axis=1)
            structure_masks[b] = np.flip(structure_masks[b], axis=1)
            dose[b] = np.flip(dose[b], axis=1)

        # CT intensity scaling (only CT, not dose)
        if intensity_scale > 0:
            scale = 1.0 + np.random.uniform(-intensity_scale, intensity_scale)
            ct[b] = ct[b] * scale

    return ct, structure_masks, dose


def soft_percentile(values: tf.Tensor, percentile: float, temperature: float = 0.5) -> tf.Tensor:
    """
    Differentiable approximation of percentile using soft sorting.

    Args:
        values: 1D tensor of values
        percentile: target percentile (0-100)
        temperature: controls sharpness (lower = closer to true percentile)

    Returns:
        Soft approximation of the percentile value
    """
    n = tf.cast(tf.shape(values)[0], tf.float32)

    # Target rank for this percentile
    target_rank = n * percentile / 100.0

    # Compute soft ranks using pairwise comparisons
    values_col = tf.expand_dims(values, 1)
    values_row = tf.expand_dims(values, 0)
    diff_matrix = values_col - values_row

    # Soft comparison: sigmoid gives smooth 0-1 for each comparison
    soft_comparisons = tf.sigmoid(diff_matrix / temperature)
    soft_ranks = tf.reduce_sum(soft_comparisons, axis=1)

    # Weight each value by how close its rank is to target
    rank_distances = tf.abs(soft_ranks - target_rank)
    weights = tf.nn.softmax(-rank_distances / temperature)

    # Weighted sum gives soft percentile
    return tf.reduce_sum(weights * values)


class PredictionModel(DefineDoseFromCT):
    def __init__(
        self,
        data_loader: DataLoader,
        results_patent_path: Path,
        model_name: str,
        stage: str,
        num_filters: int = 1,
        use_se_blocks: bool = False,
        use_dvh_loss: bool = False,
        dvh_weight: float = 0.1,
        use_augmentation: bool = False,
        use_jit: bool = True,
    ) -> None:
        """
        :param data_loader: An object that loads batches of image data
        :param results_patent_path: The path at which all results and generated models will be saved
        :param model_name: The name of your model, used when saving and loading data
        :param stage: Identify stage of model development (train, validation, test)
        :param num_filters: Initial number of filters in U-Net (recommend 64+ for real training)
        :param use_se_blocks: Enable Squeeze-and-Excitation attention blocks
        :param use_dvh_loss: Enable DVH-aware loss function
        :param dvh_weight: Weight for DVH loss term (combined: MAE + dvh_weight * DVH)
        :param use_augmentation: Enable data augmentation (flips, intensity scaling)
        :param use_jit: Enable XLA JIT compilation for faster training
        """
        super().__init__(
            data_shapes=data_loader.data_shapes,
            initial_number_of_filters=num_filters,
            filter_size=(4, 4, 4),
            stride_size=(2, 2, 2),
            gen_optimizer=Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999),
            use_se_blocks=use_se_blocks,
            use_jit=use_jit,
        )

        # Training settings
        self.use_dvh_loss = use_dvh_loss
        self.dvh_weight = dvh_weight
        self.use_augmentation = use_augmentation

        # set attributes for data shape from data loader
        self.generator = None
        self.model_name = model_name
        self.data_loader = data_loader
        self.full_roi_list = data_loader.full_roi_list

        # Define training parameters
        self.current_epoch = 0
        self.last_epoch = 200

        # Make directories for data and models
        model_results_path = results_patent_path / model_name
        self.model_dir = model_results_path / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.prediction_dir = model_results_path / f"{stage}-predictions"
        self.prediction_dir.mkdir(parents=True, exist_ok=True)

        # Training log file
        self.loss_log_path = model_results_path / "training_loss.csv"

        # Make template for model path
        self.model_path_template = self.model_dir / "epoch_"

    def _get_target_roi_indices(self) -> List[int]:
        """Get indices of PTV target structures for DVH loss."""
        targets = ['PTV56', 'PTV63', 'PTV70']
        indices = []
        for target in targets:
            if target in self.full_roi_list:
                indices.append(self.full_roi_list.index(target))
        return indices

    def _compute_dvh_loss(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, structure_masks: tf.Tensor, temperature: float = 0.5
    ) -> tf.Tensor:
        """
        Compute DVH loss for target structures (PTVs).

        Args:
            y_true: (batch, D, H, W, 1) ground truth dose
            y_pred: (batch, D, H, W, 1) predicted dose
            structure_masks: (batch, D, H, W, 10) ROI masks
            temperature: soft percentile temperature

        Returns:
            Scalar DVH loss
        """
        target_indices = self._get_target_roi_indices()
        percentiles = {'D_99': 1.0, 'D_95': 5.0, 'D_1': 99.0}

        dvh_losses = []
        batch_size = tf.shape(y_true)[0]

        for b in range(batch_size):
            y_true_b = y_true[b, ..., 0]  # (D, H, W)
            y_pred_b = y_pred[b, ..., 0]  # (D, H, W)
            masks_b = structure_masks[b]   # (D, H, W, 10)

            for roi_idx in target_indices:
                roi_mask = masks_b[..., roi_idx]  # (D, H, W)
                mask_sum = tf.reduce_sum(roi_mask)

                # Skip empty ROIs
                if mask_sum < 1.0:
                    continue

                # Extract dose values within ROI
                true_dose_roi = tf.boolean_mask(y_true_b, roi_mask > 0.5)
                pred_dose_roi = tf.boolean_mask(y_pred_b, roi_mask > 0.5)

                for _, percentile in percentiles.items():
                    true_p = soft_percentile(true_dose_roi, percentile, temperature)
                    pred_p = soft_percentile(pred_dose_roi, percentile, temperature)
                    dvh_losses.append(tf.abs(true_p - pred_p))

        if len(dvh_losses) == 0:
            return tf.constant(0.0, dtype=tf.float32)

        return tf.reduce_mean(tf.stack(dvh_losses))

    def _train_step_with_dvh(
        self, ct: tf.Tensor, structure_masks: tf.Tensor, dose_true: tf.Tensor
    ) -> dict:
        """Custom training step with DVH loss."""
        with tf.GradientTape() as tape:
            dose_pred = self.generator([ct, structure_masks], training=True)

            # MAE loss
            mae_loss = tf.reduce_mean(tf.abs(dose_true - dose_pred))

            # DVH loss
            dvh_loss = self._compute_dvh_loss(dose_true, dose_pred, structure_masks)
            total_loss = mae_loss + self.dvh_weight * dvh_loss

        # Compute and apply gradients
        gradients = tape.gradient(total_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        return {
            'loss': float(total_loss),
            'mae': float(mae_loss),
            'dvh': float(dvh_loss)
        }

    def train_model(self, epochs: int = 200, save_frequency: int = 5, keep_model_history: int = 2) -> None:
        """
        :param epochs: the number of epochs the model will be trained over
        :param save_frequency: how often the model will be saved (older models will be deleted to conserve storage)
        :param keep_model_history: how many models are kept on a rolling basis (deletes older than save_frequency * keep_model_history epochs)
        """
        self._set_epoch_start()
        self.last_epoch = epochs
        self.initialize_networks()
        if self.current_epoch == epochs:
            print(f"The model has already been trained for {epochs}, so no more training will be done.")
            return
        self.data_loader.set_mode("training_model")

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            self.data_loader.shuffle_data()

            epoch_metrics = {'loss': [], 'mae': [], 'dvh': []}

            for idx, batch in enumerate(self.data_loader.get_batches()):
                # Get batch data
                ct, structure_masks, dose = batch.ct, batch.structure_masks, batch.dose

                # Apply augmentation if enabled
                if self.use_augmentation:
                    ct, structure_masks, dose = augment_batch(ct, structure_masks, dose)

                if self.use_dvh_loss:
                    # Custom training step with DVH loss
                    metrics = self._train_step_with_dvh(ct, structure_masks, dose)
                    for k, v in metrics.items():
                        epoch_metrics[k].append(v)
                else:
                    # Standard training
                    model_loss = self.generator.train_on_batch([ct, structure_masks], [dose])
                    epoch_metrics['loss'].append(model_loss)
                    epoch_metrics['mae'].append(model_loss)

            # Log metrics
            avg_metrics = {k: sum(v) / len(v) for k, v in epoch_metrics.items() if v}
            if self.use_dvh_loss:
                print(f"Epoch {self.current_epoch}: loss={avg_metrics['loss']:.4f}, mae={avg_metrics['mae']:.4f}, dvh={avg_metrics['dvh']:.4f}")
            else:
                print(f"Epoch {self.current_epoch}: avg_loss={avg_metrics['loss']:.4f}")
            self._log_loss(avg_metrics['loss'])
            self.manage_model_storage(save_frequency, keep_model_history)

    def _log_loss(self, loss: float) -> None:
        """Append loss to CSV file for monitoring."""
        write_header = not self.loss_log_path.exists()
        with open(self.loss_log_path, "a") as f:
            if write_header:
                f.write("epoch,loss\n")
            f.write(f"{self.current_epoch},{loss:.6f}\n")

    def _set_epoch_start(self) -> None:
        all_model_paths = get_paths(self.model_dir, extension="keras")
        for model_path in all_model_paths:
            *_, epoch_number = model_path.stem.split("epoch_")
            if epoch_number.isdigit():
                self.current_epoch = max(self.current_epoch, int(epoch_number))

    def initialize_networks(self) -> None:
        if self.current_epoch >= 1:
            self.generator = load_model(self._get_generator_path(self.current_epoch))
            # Recompile (optionally with XLA JIT for faster GPU execution)
            self.generator.compile(loss="mean_absolute_error", optimizer=Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999), jit_compile=self.use_jit)
        else:
            self.generator = self.define_generator()

    def manage_model_storage(self, save_frequency: int = 1, keep_model_history: Optional[int] = None) -> None:
        """
        Manage the model storage while models are trained. Note that old models are deleted based on how many models the users has asked to keep.
        We overwrite old files (rather than deleting them) to ensure the Collab users don't fill up their Google Drive trash.
        :param save_frequency: how often the model will be saved (older models will be deleted to conserve storage)
        :param keep_model_history: how many models back are kept (older models will be deleted to conserve storage)
        """
        effective_epoch_number = self.current_epoch + 1  # Epoch number + 1 because we're at the start of the next epoch
        if 0 < np.mod(effective_epoch_number, save_frequency) and effective_epoch_number != self.last_epoch:
            Warning(f"Model at the end of epoch {self.current_epoch} was not saved because it is skipped when save frequency {save_frequency}.")
            return

        # The code below is clunky and was only included to bypass the Google Drive trash, which fills quickly with normal save/delete functions
        epoch_to_overwrite = effective_epoch_number - keep_model_history * (save_frequency or float("inf"))
        if epoch_to_overwrite >= 0:
            initial_model_path = self._get_generator_path(epoch_to_overwrite)
            self.generator.save(initial_model_path)
            os.rename(initial_model_path, self._get_generator_path(effective_epoch_number))  # Helps bypass Google Drive trash
        else:  # Save via more conventional method because there is no model to overwrite
            self.generator.save(self._get_generator_path(effective_epoch_number))

    def _get_generator_path(self, epoch: Optional[int] = None) -> Path:
        epoch = epoch or self.current_epoch
        return self.model_dir / f"epoch_{epoch}.keras"

    def predict_dose(self, epoch: int = 1) -> None:
        """Predicts the dose for the given epoch number"""
        self.generator = load_model(self._get_generator_path(epoch))
        os.makedirs(self.prediction_dir, exist_ok=True)
        self.data_loader.set_mode("dose_prediction")

        print("Predicting dose with generator.")
        for batch in self.data_loader.get_batches():
            dose_pred = self.generator.predict([batch.ct, batch.structure_masks])
            dose_pred = dose_pred * batch.possible_dose_mask

            # Denormalize dose if normalization was used during training
            if self.data_loader.normalize:
                dose_pred = dose_pred * self.data_loader.DOSE_PRESCRIPTION

            dose_pred = np.squeeze(dose_pred)
            dose_to_save = sparse_vector_function(dose_pred)
            dose_df = pd.DataFrame(data=dose_to_save["data"].squeeze(), index=dose_to_save["indices"].squeeze(), columns=["data"])
            (patient_id,) = batch.patient_list
            dose_df.to_csv("{}/{}.csv".format(self.prediction_dir, patient_id))
