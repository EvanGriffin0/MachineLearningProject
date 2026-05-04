"""
dataset_loader.py
-----------------
Dataset loading and preprocessing utilities for image classification.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_datasets(
    dataset_path,
    image_size=(224, 224),
    batch_size=32,
    seed=419296,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
    color_mode="rgb",
    label_mode="int",
):
    """Load an image dataset from a directory and split it into train/val/test.

    The default integer labels match sparse_categorical_crossentropy and the
    scikit-learn metrics used elsewhere in the project.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0 (got {train_ratio + val_ratio + test_ratio:.4f})"
        )
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")

    full_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        color_mode=color_mode,
        label_mode=label_mode,
    )

    class_names = full_ds.class_names
    total_batches = full_ds.cardinality().numpy()
    train_size = int(total_batches * train_ratio)
    val_size = int(total_batches * val_ratio)

    train_ds = full_ds.take(train_size)
    remaining = full_ds.skip(train_size)
    val_ds = remaining.take(val_size)
    test_ds = remaining.skip(val_size)

    return train_ds, val_ds, test_ds, class_names


def prepare_dataset(dataset, training=False, seed=419296, cache=True):
    """Apply cache, optional shuffle, and prefetch to a tf.data dataset."""
    if cache:
        dataset = dataset.cache()
    if training:
        dataset = dataset.shuffle(buffer_size=1000, seed=seed)
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


def count_images_per_class(dataset_path):
    """Count valid image files in each immediate class subdirectory."""
    counts = {}
    for class_name in sorted(os.listdir(dataset_path)):
        if class_name.startswith("."):
            continue
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        image_count = sum(
            1
            for fname in os.listdir(class_dir)
            if not fname.startswith(".")
            and os.path.splitext(fname)[1].lower() in VALID_IMAGE_EXTENSIONS
        )
        counts[class_name] = image_count
    return counts


def compute_class_weights(class_counts):
    """Compute balanced Keras class weights from class file counts."""
    class_names = sorted(class_counts.keys())
    class_indices = list(range(len(class_names)))

    labels = []
    for idx, name in enumerate(class_names):
        labels.extend([idx] * class_counts[name])

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(class_indices),
        y=np.array(labels),
    )
    return dict(zip(class_indices, weights))


def plot_class_distribution(class_counts):
    """Create a bar chart showing the number of images per class."""
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    fig, ax = plt.subplots(figsize=(max(8, len(classes)), 5))
    ax.bar(classes, counts, color="steelblue", edgecolor="black")
    ax.set_title("Number of Images per Class", fontsize=14)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Image Count", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


# Backwards-compatible names used by the starter notebook.
load_dataset = load_datasets
prefetch_dataset = prepare_dataset
