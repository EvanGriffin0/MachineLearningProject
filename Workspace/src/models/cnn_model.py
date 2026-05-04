"""
cnn_model.py
------------
Defines a lightweight CNN architecture built from scratch using Keras Sequential API.

The network follows a standard conv → pool → conv → pool → ... → dense pattern
and is intended as a baseline before applying transfer learning.
"""

import tensorflow as tf
from tensorflow import keras


def build_cnn_model(
    input_shape: tuple = (224, 224, 3),
    num_classes: int = 10,
    conv_filters: tuple = (32, 64, 128),
    dense_units: int = 256,
    dropout_rate: float = 0.5,
    learning_rate: float = 1e-3,
    use_augmentation: bool = False,
) -> keras.Model:
    """Build and return a simple CNN model compiled and ready for training.

    Architecture
    ------------
    Three convolutional blocks (Conv2D → MaxPooling2D), followed by a
    fully-connected head with dropout regularisation.

    Parameters
    ----------
    input_shape : tuple
        Shape of a single input image ``(H, W, C)``.
    num_classes : int
        Number of output classes.

    Returns
    -------
    keras.Model
        Compiled Keras model.
    """
    layers = [
        keras.Input(shape=input_shape),
        keras.layers.Rescaling(1.0 / 255),
    ]

    if use_augmentation:
        layers.extend(
            [
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomRotation(0.08),
                keras.layers.RandomZoom(0.1),
            ]
        )

    for filters in conv_filters:
        layers.extend(
            [
                keras.layers.Conv2D(filters, (3, 3), activation="relu", padding="same"),
                keras.layers.MaxPooling2D((2, 2)),
            ]
        )

    layers.extend(
        [
            keras.layers.Flatten(),
            keras.layers.Dense(dense_units, activation="relu"),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model = keras.Sequential(layers, name="cnn_from_scratch")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
