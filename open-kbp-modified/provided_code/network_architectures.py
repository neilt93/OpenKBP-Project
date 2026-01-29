""" Neural net architectures """
from typing import Optional, Any

import tensorflow as tf
from tensorflow.keras.layers import (
    Activation, AveragePooling3D, Conv3D, Conv3DTranspose,
    Input, LeakyReLU, SpatialDropout3D, Concatenate, BatchNormalization,
    GlobalAveragePooling3D, Dense, Reshape, Multiply, Add, Layer
)
from tensorflow.keras.models import Model

# Type aliases for Keras 3 compatibility
KerasTensor = Any
OptimizerV2 = Any

from provided_code.data_shapes import DataShapes


class InstanceNormalization(Layer):
    """Instance Normalization layer - normalizes per sample per channel.

    Better than BatchNorm for small batch sizes (1-2) common in 3D medical imaging.
    """
    def __init__(self, epsilon: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        # Learnable scale (gamma) and shift (beta) per channel
        self.gamma = self.add_weight(
            name='gamma',
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        # Normalize over spatial dimensions (1, 2, 3) for 3D data
        mean = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
        var = tf.math.reduce_variance(x, axis=[1, 2, 3], keepdims=True)
        x_norm = (x - mean) / tf.sqrt(var + self.epsilon)
        return self.gamma * x_norm + self.beta

    def get_config(self):
        config = super().get_config()
        config.update({'epsilon': self.epsilon})
        return config


class DefineDoseFromCT:
    """This class defines the architecture for a U-NET and must be inherited by a child class that
    executes various functions like training or predicting"""

    def __init__(
        self,
        data_shapes: DataShapes,
        initial_number_of_filters: int,
        filter_size: tuple[int, int, int],
        stride_size: tuple[int, int, int],
        gen_optimizer: OptimizerV2,
        use_se_blocks: bool = False,
        se_reduction_ratio: int = 8,
        use_residual: bool = True,
        use_jit: bool = True,
    ):
        self.data_shapes = data_shapes
        self.initial_number_of_filters = initial_number_of_filters
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.gen_optimizer = gen_optimizer
        self.use_se_blocks = use_se_blocks
        self.se_reduction_ratio = se_reduction_ratio
        self.use_residual = use_residual
        self.use_jit = use_jit

    def squeeze_excitation_block(self, x: KerasTensor, reduction_ratio: int = None) -> KerasTensor:
        """
        Squeeze-and-Excitation block for channel attention.
        Recalibrates channel-wise feature responses by modeling inter-channel dependencies.
        """
        reduction_ratio = reduction_ratio or self.se_reduction_ratio
        filters = int(x.shape[-1])
        hidden_units = max(8, filters // reduction_ratio)

        # Squeeze: global average pooling
        se = GlobalAveragePooling3D()(x)

        # Excitation: FC -> ReLU -> FC -> Sigmoid
        se = Dense(hidden_units, use_bias=False)(se)
        se = Activation('relu')(se)
        se = Dense(filters, use_bias=False)(se)

        # Sigmoid with float32 for numerical stability in mixed precision
        se = tf.cast(se, tf.float32)
        se = Activation('sigmoid')(se)
        se = tf.cast(se, x.dtype)

        # Scale: broadcast and multiply
        se = Reshape((1, 1, 1, filters))(se)
        x = Multiply()([x, se])

        return x

    def make_convolution_block(self, x: KerasTensor, num_filters: int, use_norm: bool = True, use_se: bool = None) -> KerasTensor:
        """Encoder block with optional residual connection, InstanceNorm, and SE."""
        use_se = use_se if use_se is not None else self.use_se_blocks

        # Main path
        out = Conv3D(num_filters, self.filter_size, strides=self.stride_size, padding="same", use_bias=False)(x)
        if use_norm:
            out = InstanceNormalization()(out)
        out = LeakyReLU(negative_slope=0.2)(out)

        # Optional residual: add 1x1 conv to match dimensions if needed
        if self.use_residual and x.shape[-1] == num_filters and self.stride_size == (1, 1, 1):
            out = Add()([out, x])

        if use_se:
            out = self.squeeze_excitation_block(out)
        return out

    def make_convolution_transpose_block(
        self, x: KerasTensor, num_filters: int, use_dropout: bool = True, skip_x: Optional[KerasTensor] = None, use_se: bool = None
    ) -> KerasTensor:
        """Decoder block with InstanceNorm and optional SE."""
        use_se = use_se if use_se is not None else self.use_se_blocks
        if skip_x is not None:
            x = Concatenate()([x, skip_x])

        out = Conv3DTranspose(num_filters, self.filter_size, strides=self.stride_size, padding="same", use_bias=False)(x)
        out = InstanceNormalization()(out)
        if use_dropout:
            out = SpatialDropout3D(0.2)(out)
        out = Activation("relu")(out)

        if use_se:
            out = self.squeeze_excitation_block(out)
        return out

    def define_generator(self) -> Model:
        """Makes a generator that takes a CT image as input to generate a dose distribution of the same dimensions"""

        # Define inputs
        ct_image = Input(self.data_shapes.ct)
        roi_masks = Input(self.data_shapes.structure_masks)

        # Build Model starting with Conv3D layers
        x = Concatenate()([ct_image, roi_masks])
        x1 = self.make_convolution_block(x, self.initial_number_of_filters)
        x2 = self.make_convolution_block(x1, 2 * self.initial_number_of_filters)
        x3 = self.make_convolution_block(x2, 4 * self.initial_number_of_filters)
        x4 = self.make_convolution_block(x3, 8 * self.initial_number_of_filters)
        x5 = self.make_convolution_block(x4, 8 * self.initial_number_of_filters)
        x6 = self.make_convolution_block(x5, 8 * self.initial_number_of_filters, use_batch_norm=False)

        # Build model back up from bottleneck
        x5b = self.make_convolution_transpose_block(x6, 8 * self.initial_number_of_filters, use_dropout=False)
        x4b = self.make_convolution_transpose_block(x5b, 8 * self.initial_number_of_filters, skip_x=x5)
        x3b = self.make_convolution_transpose_block(x4b, 4 * self.initial_number_of_filters, use_dropout=False, skip_x=x4)
        x2b = self.make_convolution_transpose_block(x3b, 2 * self.initial_number_of_filters, skip_x=x3)
        x1b = self.make_convolution_transpose_block(x2b, self.initial_number_of_filters, use_dropout=False, skip_x=x2)

        # Final layer (use float32 output for numerical stability with mixed precision)
        x0b = Concatenate()([x1b, x1])
        x0b = Conv3DTranspose(1, self.filter_size, strides=self.stride_size, padding="same", dtype="float32")(x0b)
        x_final = AveragePooling3D((3, 3, 3), strides=(1, 1, 1), padding="same", dtype="float32")(x0b)
        final_dose = Activation("relu", dtype="float32")(x_final)

        # Compile model (optionally with XLA JIT for faster GPU execution)
        generator = Model(inputs=[ct_image, roi_masks], outputs=final_dose, name="generator")
        generator.compile(loss="mean_absolute_error", optimizer=self.gen_optimizer, jit_compile=self.use_jit)
        generator.summary()
        return generator
