"""
dataset_loader.py
-----------------
Dataset loading and preprocessing utilities for image classification.
Uses keras.utils.image_dataset_from_directory to build tf.data pipelines
that load images lazily from disk, avoiding memory issues with large datasets.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

# Image file extensions considered valid
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# A. Load and split dataset
# ---------------------------------------------------------------------------

def load_datasets(
    dataset_path,
    image_size=(224, 224),
    batch_size=32,
    seed=419296,
    train_ratio=0.7,
    val_ratio=0.1,
    test_ratio=0.2,
):
    """Load an image dataset from a directory and split into train/val/test.

    Uses keras.utils.image_dataset_from_directory which reads images lazily
    from disk, avoiding loading the entire dataset into RAM.

    The dataset is shuffled once using a fixed seed before splitting, so the
    split is deterministic and reproducible across runs.

    Parameters
    ----------
    dataset_path : str
        Root directory whose sub-folders are class labels.
    image_size : tuple
        Target (height, width) to resize all images to.
    batch_size : int
        Number of images per batch.
    seed : int
        Random seed for shuffling. Use your student ID for reproducibility.
    train_ratio : float
        Proportion of batches used for training.
    val_ratio : float
        Proportion of batches used for validation.
    test_ratio : float
        Proportion of batches reserved for final testing.

    Returns
    -------
    train_ds, val_ds, test_ds : tf.data.Dataset
    class_names : list[str]
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0 (got {train_ratio + val_ratio + test_ratio:.4f})"
        )

    # Load the full dataset in one pass — splitting is done via take/skip below
    full_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        label_mode="categorical",
    )

    class_names = full_ds.class_names
    total_batches = full_ds.cardinality().numpy()

    train_size = int(total_batches * train_ratio)
    val_size   = int(total_batches * val_ratio)
    # test gets whatever remains, avoiding rounding gaps

    train_ds  = full_ds.take(train_size)
    remaining = full_ds.skip(train_size)
    val_ds    = remaining.take(val_size)
    test_ds   = remaining.skip(val_size)

    return train_ds, val_ds, test_ds, class_names


# ---------------------------------------------------------------------------
# B. Prepare dataset (cache + optional shuffle + prefetch)
# ---------------------------------------------------------------------------

def prepare_dataset(dataset, training=False):
    """Apply performance optimisations to a tf.data.Dataset.

    Caching stores the dataset in memory after the first epoch, removing
    repeated disk reads. Prefetching overlaps data loading with model
    computation so the GPU is never idle waiting for the next batch.

    Parameters
    ----------
    dataset : tf.data.Dataset
        A batched dataset returned by load_datasets().
    training : bool
        If True, shuffle the dataset each epoch (only for training set).

    Returns
    -------
    tf.data.Dataset
    """
    dataset = dataset.cache()
    if training:
        dataset = dataset.shuffle(buffer_size=1000, seed=419296)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


# ---------------------------------------------------------------------------
# C. Count images per class
# ---------------------------------------------------------------------------

def count_images_per_class(dataset_path):
    """Count valid image files in each class subdirectory.

    Parameters
    ----------
    dataset_path : str
        Root directory whose immediate sub-folders are class labels.

    Returns
    -------
    dict
        {class_name: image_count} sorted alphabetically by class name.
    """
    counts = {}
    for class_name in sorted(os.listdir(dataset_path)):
        # Skip hidden files/folders (e.g. .DS_Store)
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


# ---------------------------------------------------------------------------
# D. Compute class weights
# ---------------------------------------------------------------------------

def compute_class_weights(class_counts):
    """Compute balanced class weights to handle class imbalance.

    Classes with fewer images receive a higher weight so the model does not
    become biased towards majority classes during training.

    Classes are sorted alphabetically before index assignment so the mapping
    is consistent with the order used by image_dataset_from_directory.

    Parameters
    ----------
    class_counts : dict
        {class_name: image_count} as returned by count_images_per_class().

    Returns
    -------
    dict
        {class_index: weight} for use with model.fit(class_weight=...).
    """
    class_names   = sorted(class_counts.keys())
    class_indices = list(range(len(class_names)))

    # Build a flat label array expected by sklearn
    y = []
    for idx, name in enumerate(class_names):
        y.extend([idx] * class_counts[name])

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(class_indices),
        y=np.array(y),
    )

    return dict(zip(class_indices, weights))


# ---------------------------------------------------------------------------
# E. Plot class distribution
# ---------------------------------------------------------------------------

def plot_class_distribution(class_counts):
    """Create a bar chart showing the number of images per class.

    Parameters
    ----------
    class_counts : dict
        {class_name: image_count} as returned by count_images_per_class().

    Returns
    -------
    matplotlib.figure.Figure
    """
    classes = list(class_counts.keys())
    counts  = list(class_counts.values())

    fig, ax = plt.subplots(figsize=(max(8, len(classes)), 5))
    ax.bar(classes, counts, color="steelblue", edgecolor="black")
    ax.set_title("Number of Images per Class", fontsize=14)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Image Count", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig
