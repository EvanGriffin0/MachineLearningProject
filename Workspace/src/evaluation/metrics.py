"""
metrics.py
----------
Evaluation metrics for image classification models.

Wraps ``scikit-learn`` metric functions behind a consistent interface and
provides a ``get_predictions`` helper that runs inference over a full
``tf.data.Dataset`` and returns flat NumPy arrays suitable for all metric
functions below.
"""

import numpy as np
import tensorflow as tf
from timeit import default_timer as timer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def get_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
):
    """Run inference on a full dataset and return true labels and predictions.

    Parameters
    ----------
    model : tf.keras.Model
        A trained Keras model.
    dataset : tf.data.Dataset
        Batched dataset yielding ``(images, labels)`` tuples.

    Returns
    -------
    y_true : np.ndarray
        Ground-truth integer class labels.
    y_pred : np.ndarray
        Predicted integer class labels (argmax of softmax output).
    """
    y_true, y_pred = [], []
    for images, labels in dataset:
        probabilities = model.predict(images, verbose=0)
        label_values = labels.numpy()
        if label_values.ndim > 1:
            label_values = np.argmax(label_values, axis=1)
        y_true.extend(label_values)
        y_pred.extend(np.argmax(probabilities, axis=1))
    return np.array(y_true), np.array(y_pred)


def measure_inference_time(model: tf.keras.Model, dataset: tf.data.Dataset) -> dict:
    """Measure inference time over a dataset without using it for training."""
    total_images = 0
    start = timer()
    for images, _ in dataset:
        model.predict(images, verbose=0)
        total_images += int(images.shape[0])
    elapsed_seconds = timer() - start
    return {
        "total_images": total_images,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_image": elapsed_seconds / total_images if total_images else 0.0,
    }


def count_trainable_parameters(model: tf.keras.Model) -> dict:
    """Return total, trainable, and non-trainable parameter counts."""
    trainable = int(
        np.sum([tf.keras.backend.count_params(weight) for weight in model.trainable_weights])
    )
    non_trainable = int(
        np.sum([tf.keras.backend.count_params(weight) for weight in model.non_trainable_weights])
    )
    return {
        "total": trainable + non_trainable,
        "trainable": trainable,
        "non_trainable": non_trainable,
    }


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return overall classification accuracy."""
    return accuracy_score(y_true, y_pred)


def compute_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "weighted",
) -> float:
    """Return precision score.

    Parameters
    ----------
    average : str
        Averaging strategy passed to ``sklearn`` — ``'weighted'``, ``'macro'``,
        or ``'micro'``.
    """
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def compute_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "weighted",
) -> float:
    """Return recall score."""
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def compute_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "weighted",
) -> float:
    """Return F1 score."""
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Return the confusion matrix as a 2-D NumPy array."""
    return confusion_matrix(y_true, y_pred)


def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = None,
) -> str:
    """Return per-class precision, recall, F1, and support."""
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def print_summary(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print a formatted table of all metrics to stdout."""
    print(f"Accuracy : {compute_accuracy(y_true, y_pred):.4f}")
    print(f"Precision: {compute_precision(y_true, y_pred):.4f}")
    print(f"Recall   : {compute_recall(y_true, y_pred):.4f}")
    print(f"F1 Score : {compute_f1(y_true, y_pred):.4f}")
    print(f"Macro F1 : {compute_f1(y_true, y_pred, average='macro'):.4f}")
