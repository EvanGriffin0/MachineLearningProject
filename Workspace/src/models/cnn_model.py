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
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),

            # --- Block 1 ---
            keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D((2, 2)),

            # --- Block 2 ---
            keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D((2, 2)),

            # --- Block 3 ---
            keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D((2, 2)),

            # --- Classifier head ---
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="cnn_from_scratch",
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
