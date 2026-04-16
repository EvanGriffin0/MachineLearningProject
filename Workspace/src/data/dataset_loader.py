"""
dataset_loader.py
-----------------
Utilities for loading image datasets from a directory structure using
``keras.utils.image_dataset_from_directory``.

Expected directory layout
(one sub-folder per class label)::

    dataset_root/
        class_a/
            img1.jpg
            img2.jpg
        class_b/
            img3.jpg
            ...
"""

import tensorflow as tf


def load_dataset(
    directory: str,
    image_size: tuple = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 42,
):
    """Load an image dataset split into training and validation subsets.

    Parameters
    ----------
    directory : str
        Root directory of the dataset. Sub-folders are treated as class labels.
    image_size : tuple
        Target ``(height, width)`` to resize all images to.
    batch_size : int
        Number of samples per batch.
    validation_split : float
        Fraction of data reserved for validation (0 < split < 1).
    seed : int
        Random seed used for the train/validation split.

    Returns
    -------
    train_ds : tf.data.Dataset
    val_ds : tf.data.Dataset
    class_names : list[str]
        Ordered list of class label strings inferred from sub-folder names.
    """
    shared_kwargs = dict(
        validation_split=validation_split,
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        subset="training",
        **shared_kwargs,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        subset="validation",
        **shared_kwargs,
    )

    class_names = train_ds.class_names
    return train_ds, val_ds, class_names


def prefetch_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Apply caching and prefetching for improved training throughput.

    Parameters
    ----------
    dataset : tf.data.Dataset
        A batched ``tf.data.Dataset``.

    Returns
    -------
    tf.data.Dataset
        The same dataset with ``.cache().prefetch()`` applied.
    """
    autotune = tf.data.AUTOTUNE
    return dataset.cache().prefetch(buffer_size=autotune)
