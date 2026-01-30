import os
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from provided_code.data_loader import DataLoader
from provided_code.network_architectures import DefineDoseFromCT, InstanceNormalization
from provided_code.utils import get_paths, sparse_vector_function


@tf.function
def augment_batch_tf(ct: tf.Tensor, structure_masks: tf.Tensor, dose: tf.Tensor,
                     possible_dose_mask: tf.Tensor,
                     flip_prob: float = 0.5, intensity_scale: float = 0.1) -> tuple:
    """
    TensorFlow-based data augmentation (XLA-compatible).

    Uses tf.cond for proper graph-mode behavior.

    Args:
        ct: (batch, D, H, W, 1) CT images
        structure_masks: (batch, D, H, W, 10) ROI masks
        dose: (batch, D, H, W, 1) dose distributions
        possible_dose_mask: (batch, D, H, W, 1) mask for dose region
        flip_prob: probability of flipping along each axis
        intensity_scale: max CT intensity scaling factor (±scale)

    Returns:
        Augmented (ct, structure_masks, dose, possible_dose_mask) tuple
    """
    # Random left-right flip (axis 3 in BDHWC format)
    do_lr_flip = tf.random.uniform([]) < flip_prob
    ct = tf.cond(do_lr_flip, lambda: tf.reverse(ct, axis=[3]), lambda: ct)
    structure_masks = tf.cond(do_lr_flip, lambda: tf.reverse(structure_masks, axis=[3]), lambda: structure_masks)
    dose = tf.cond(do_lr_flip, lambda: tf.reverse(dose, axis=[3]), lambda: dose)
    possible_dose_mask = tf.cond(do_lr_flip, lambda: tf.reverse(possible_dose_mask, axis=[3]), lambda: possible_dose_mask)

    # Random anterior-posterior flip (axis 2)
    do_ap_flip = tf.random.uniform([]) < flip_prob
    ct = tf.cond(do_ap_flip, lambda: tf.reverse(ct, axis=[2]), lambda: ct)
    structure_masks = tf.cond(do_ap_flip, lambda: tf.reverse(structure_masks, axis=[2]), lambda: structure_masks)
    dose = tf.cond(do_ap_flip, lambda: tf.reverse(dose, axis=[2]), lambda: dose)
    possible_dose_mask = tf.cond(do_ap_flip, lambda: tf.reverse(possible_dose_mask, axis=[2]), lambda: possible_dose_mask)

    # CT intensity scaling (always apply if intensity_scale > 0, just vary the scale)
    scale = 1.0 + tf.random.uniform([], -intensity_scale, intensity_scale)
    ct = ct * scale
    ct = tf.clip_by_value(ct, 0.0, 1.0)  # Keep CT in normalized range

    return ct, structure_masks, dose, possible_dose_mask


def augment_batch(ct: np.ndarray, structure_masks: np.ndarray, dose: np.ndarray,
                   flip_prob: float = 0.5, intensity_scale: float = 0.1) -> tuple:
    """NumPy fallback for augmentation (slower, breaks XLA)."""
    ct = ct.copy()
    structure_masks = structure_masks.copy()
    dose = dose.copy()

    batch_size = ct.shape[0]

    for b in range(batch_size):
        if np.random.random() < flip_prob:
            ct[b] = np.flip(ct[b], axis=2)
            structure_masks[b] = np.flip(structure_masks[b], axis=2)
            dose[b] = np.flip(dose[b], axis=2)

        if np.random.random() < flip_prob:
            ct[b] = np.flip(ct[b], axis=1)
            structure_masks[b] = np.flip(structure_masks[b], axis=1)
            dose[b] = np.flip(dose[b], axis=1)

        if intensity_scale > 0:
            scale = 1.0 + np.random.uniform(-intensity_scale, intensity_scale)
            ct[b] = ct[b] * scale

    return ct, structure_masks, dose


def histogram_percentile(values: tf.Tensor, percentile: float, num_bins: int = 100) -> tf.Tensor:
    """
    Fast differentiable percentile using histogram-based approximation.

    O(n + num_bins) instead of O(n²). Much faster and more stable than soft sorting.

    Args:
        values: 1D tensor of values (float32)
        percentile: target percentile (0-100)
        num_bins: number of histogram bins

    Returns:
        Approximate percentile value
    """
    values = tf.cast(values, tf.float32)

    # Get value range
    v_min = tf.reduce_min(values)
    v_max = tf.reduce_max(values)
    v_range = v_max - v_min + 1e-8

    # Normalize values to [0, 1]
    values_norm = (values - v_min) / v_range

    # Create soft histogram using sigmoid
    bin_edges = tf.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Soft assignment to bins (differentiable)
    # Each value contributes to nearby bins with Gaussian weight
    sigma = 1.0 / num_bins
    values_expanded = tf.expand_dims(values_norm, 1)  # (n, 1)
    bin_centers_expanded = tf.expand_dims(bin_centers, 0)  # (1, num_bins)
    weights = tf.exp(-0.5 * tf.square((values_expanded - bin_centers_expanded) / sigma))
    hist = tf.reduce_sum(weights, axis=0)  # (num_bins,)

    # Normalize to PDF then CDF
    hist = hist / (tf.reduce_sum(hist) + 1e-8)
    cdf = tf.cumsum(hist)

    # Find bin where CDF crosses target percentile
    target = percentile / 100.0
    # Soft argmax for the crossing point
    crossing_weights = tf.nn.softmax(-tf.abs(cdf - target) * 20.0)
    bin_idx = tf.reduce_sum(crossing_weights * tf.cast(tf.range(num_bins), tf.float32))

    # Interpolate to get value
    percentile_norm = bin_idx / tf.cast(num_bins, tf.float32)
    return v_min + percentile_norm * v_range


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
        use_masked_loss: bool = True,
        ptv_weight: float = 2.0,
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
        :param use_masked_loss: Use masked MAE (only compute loss in possible_dose_mask region)
        :param ptv_weight: Extra weight on PTV voxels (0 = no extra weight, 2.0 = 3x weight on PTVs)
        """
        super().__init__(
            data_shapes=data_loader.data_shapes,
            initial_number_of_filters=num_filters,
            filter_size=(3, 3, 3),  # 3x3x3 is lighter than 4x4x4, reduces bandwidth pressure
            stride_size=(2, 2, 2),
            gen_optimizer=Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999),
            use_se_blocks=use_se_blocks,
            use_residual=True,  # Always use residual connections
            use_jit=use_jit,
        )

        # Training settings
        self.use_dvh_loss = use_dvh_loss
        self.dvh_weight = dvh_weight
        self.use_augmentation = use_augmentation
        self.use_masked_loss = use_masked_loss
        self.ptv_weight = ptv_weight

        # set attributes for data shape from data loader
        self.generator = None
        self.model_name = model_name
        self.data_loader = data_loader
        self.full_roi_list = data_loader.full_roi_list

        # Cache PTV indices for masked loss (avoids hardcoding)
        self.ptv56_idx = self.full_roi_list.index("PTV56")
        self.ptv63_idx = self.full_roi_list.index("PTV63")
        self.ptv70_idx = self.full_roi_list.index("PTV70")

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
        self, y_true: tf.Tensor, y_pred: tf.Tensor, structure_masks: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute DVH loss for target structures (PTVs).

        Uses histogram-based percentile approximation for speed and stability.
        All computations in float32 for mixed precision compatibility.

        Args:
            y_true: (batch, D, H, W, 1) ground truth dose
            y_pred: (batch, D, H, W, 1) predicted dose
            structure_masks: (batch, D, H, W, 10) ROI masks

        Returns:
            Scalar DVH loss (float32)
        """
        target_indices = self._get_target_roi_indices()
        if not target_indices:
            return tf.constant(0.0, dtype=tf.float32)

        # Cast to float32 for stability
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        structure_masks = tf.cast(structure_masks, tf.float32)

        percentile_list = [1.0, 5.0, 99.0]  # D_99, D_95, D_1

        # Process first sample in batch only (for speed)
        y_true_0 = y_true[0, ..., 0]  # (D, H, W)
        y_pred_0 = y_pred[0, ..., 0]
        masks_0 = structure_masks[0]

        def compute_roi_percentile_loss(roi_idx, percentile):
            """Compute single percentile loss for one ROI."""
            roi_mask = masks_0[..., roi_idx]
            mask_bool = roi_mask > 0.5
            mask_count = tf.reduce_sum(tf.cast(mask_bool, tf.float32))

            def compute_loss():
                true_dose_roi = tf.boolean_mask(y_true_0, mask_bool)
                pred_dose_roi = tf.boolean_mask(y_pred_0, mask_bool)
                true_p = histogram_percentile(true_dose_roi, percentile)
                pred_p = histogram_percentile(pred_dose_roi, percentile)
                return tf.abs(true_p - pred_p)

            # Use tf.cond instead of Python if
            return tf.cond(
                mask_count >= 10.0,
                compute_loss,
                lambda: tf.constant(0.0, dtype=tf.float32)
            )

        # Compute all losses (static unrolling for small loops is OK)
        all_losses = []
        for roi_idx in target_indices:
            for percentile in percentile_list:
                loss = compute_roi_percentile_loss(roi_idx, percentile)
                all_losses.append(loss)

        # Stack and compute mean of non-zero losses
        stacked = tf.stack(all_losses)
        nonzero_mask = stacked > 0.0
        nonzero_count = tf.reduce_sum(tf.cast(nonzero_mask, tf.float32))

        return tf.cond(
            nonzero_count > 0,
            lambda: tf.reduce_sum(tf.boolean_mask(stacked, nonzero_mask)) / nonzero_count,
            lambda: tf.constant(0.0, dtype=tf.float32)
        )

    def _train_step_logic(
        self, ct: tf.Tensor, structure_masks: tf.Tensor, dose_true: tf.Tensor,
        possible_dose_mask: tf.Tensor
    ) -> tuple:
        """Core training step logic (wrapped with tf.function during init).

        Uses masked MAE loss that focuses on clinically relevant voxels:
        1. Base mask: possible_dose_mask (where dose can exist)
        2. PTV weighting: extra weight on target structures (controlled by self.ptv_weight)
        """
        with tf.GradientTape() as tape:
            dose_pred = self.generator([ct, structure_masks], training=True)

            # Cast to float32 for loss computation (important for mixed precision)
            dose_pred_f32 = tf.cast(dose_pred, tf.float32)
            dose_true_f32 = tf.cast(dose_true, tf.float32)
            mask_f32 = tf.cast(possible_dose_mask, tf.float32)
            structure_masks_f32 = tf.cast(structure_masks, tf.float32)

            # Compute absolute error
            abs_err = tf.abs(dose_true_f32 - dose_pred_f32)

            if self.use_masked_loss:
                # Masked MAE: only compute loss where dose can exist
                # Optionally add PTV weighting for better DVH scores
                if self.ptv_weight > 0:
                    # Use cached PTV indices (set in __init__)
                    i56, i63, i70 = self.ptv56_idx, self.ptv63_idx, self.ptv70_idx
                    ptv_mask = (structure_masks_f32[..., i56:i56+1] +
                               structure_masks_f32[..., i63:i63+1] +
                               structure_masks_f32[..., i70:i70+1])
                    ptv_mask = tf.minimum(ptv_mask, 1.0)  # Clip overlapping regions
                    # Weight: 1 + ptv_weight on PTV voxels
                    weights = mask_f32 * (1.0 + self.ptv_weight * ptv_mask)
                else:
                    weights = mask_f32

                # Weighted masked MAE
                weighted_err = weights * abs_err
                mae_loss = tf.reduce_sum(weighted_err) / (tf.reduce_sum(weights) + 1e-8)
            else:
                # Original unmasked MAE (for comparison)
                mae_loss = tf.reduce_mean(abs_err)

            # DVH loss (already in float32)
            if self.use_dvh_loss:
                dvh_loss = self._compute_dvh_loss(dose_true_f32, dose_pred_f32, structure_masks_f32)
                total_loss = mae_loss + self.dvh_weight * dvh_loss
            else:
                dvh_loss = tf.constant(0.0, dtype=tf.float32)
                total_loss = mae_loss

        # Compute and apply gradients
        gradients = tape.gradient(total_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        return total_loss, mae_loss, dvh_loss

    def _train_step_with_dvh(
        self, ct: tf.Tensor, structure_masks: tf.Tensor, dose_true: tf.Tensor,
        possible_dose_mask: tf.Tensor
    ) -> dict:
        """Wrapper for training step (uses compiled version if available)."""
        total_loss, mae_loss, dvh_loss = self._train_step_fn(ct, structure_masks, dose_true, possible_dose_mask)
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
                # Get batch data and convert to tensors (convert_to_tensor allows overlap)
                ct = tf.convert_to_tensor(batch.ct, dtype=tf.float32)
                structure_masks = tf.convert_to_tensor(batch.structure_masks, dtype=tf.float32)
                dose = tf.convert_to_tensor(batch.dose, dtype=tf.float32)
                possible_dose_mask = tf.convert_to_tensor(batch.possible_dose_mask, dtype=tf.float32)

                # Apply TF augmentation if enabled (XLA-compatible)
                if self.use_augmentation:
                    ct, structure_masks, dose, possible_dose_mask = augment_batch_tf(
                        ct, structure_masks, dose, possible_dose_mask
                    )

                # Use compiled training step with masked loss
                metrics = self._train_step_with_dvh(ct, structure_masks, dose, possible_dose_mask)
                for k, v in metrics.items():
                    epoch_metrics[k].append(v)

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
            self.generator = load_model(self._get_generator_path(self.current_epoch), custom_objects={'InstanceNormalization': InstanceNormalization})
            # Recompile (optionally with XLA JIT for faster GPU execution)
            self.generator.compile(loss="mean_absolute_error", optimizer=Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999), jit_compile=self.use_jit)
        else:
            self.generator = self.define_generator()

        # Create training step function with or without JIT compilation
        # Note: DVH loss uses tf.boolean_mask which breaks XLA, so disable JIT if DVH is enabled
        jit_ok = self.use_jit and (not self.use_dvh_loss)
        if jit_ok:
            self._train_step_fn = tf.function(self._train_step_logic, jit_compile=True)
            print("XLA JIT compilation enabled for training step")
        else:
            self._train_step_fn = tf.function(self._train_step_logic, jit_compile=False)
            if self.use_dvh_loss and self.use_jit:
                print("XLA JIT disabled (DVH loss uses tf.boolean_mask which breaks XLA)")
            else:
                print("XLA JIT compilation disabled")

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
        self.generator = load_model(self._get_generator_path(epoch), custom_objects={'InstanceNormalization': InstanceNormalization})
        os.makedirs(self.prediction_dir, exist_ok=True)
        self.data_loader.set_mode("dose_prediction")

        print("Predicting dose with generator.")
        for batch in self.data_loader.get_batches():
            dose_pred = self.generator.predict([batch.ct, batch.structure_masks])
            dose_pred = dose_pred * batch.possible_dose_mask

            # Denormalize dose if normalization was used during training
            # Note: DataLoader uses fixed 70.0 normalization for ALL patients
            if self.data_loader.normalize:
                dose_pred = dose_pred * self.data_loader.DOSE_PRESCRIPTION

            dose_pred = np.squeeze(dose_pred)
            dose_to_save = sparse_vector_function(dose_pred)
            dose_df = pd.DataFrame(data=dose_to_save["data"].squeeze(), index=dose_to_save["indices"].squeeze(), columns=["data"])
            (patient_id,) = batch.patient_list
            dose_df.to_csv("{}/{}.csv".format(self.prediction_dir, patient_id))
