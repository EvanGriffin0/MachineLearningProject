"""
train.py
--------
Training loop utilities for compiled Keras models.

This module provides a thin wrapper around ``model.fit()`` so that notebooks
and scripts can invoke training with a consistent interface.
"""

import tensorflow as tf
from tensorflow import keras


def default_callbacks(model_path=None, patience=4):
    """Create standard callbacks for the project experiments."""
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )
    ]
    if model_path:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                model_path,
                monitor="val_loss",
                save_best_only=True,
            )
        )
    return callbacks


def train_model(
    model: keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = 20,
    callbacks: list = None,
    class_weight: dict = None,
) -> keras.callbacks.History:
    """Train a compiled Keras model and return its history.

    Parameters
    ----------
    model : keras.Model
        A compiled Keras model (loss, optimiser, and metrics already set).
    train_ds : tf.data.Dataset
        Batched training dataset.
    val_ds : tf.data.Dataset
        Batched validation dataset used for monitoring.
    epochs : int
        Number of full passes through the training data.
    callbacks : list, optional
        Keras callbacks to apply during training
        (e.g. ``EarlyStopping``, ``ModelCheckpoint``).
    class_weight : dict, optional
        Class weights passed through to ``model.fit()`` for imbalanced data.

    Returns
    -------
    keras.callbacks.History
        History object containing per-epoch loss and metric values for both
        the training and validation sets.
    """
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks or [],
        class_weight=class_weight,
    )
    return history
