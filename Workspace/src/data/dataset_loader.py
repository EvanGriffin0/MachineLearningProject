"""
dataset_loader.py
-----------------
Placeholder module for dataset loading utilities.
Full implementation will be added in the next commit.
"""


def load_dataset(dataset_path, image_size, batch_size, seed):
    """
    Loads dataset using keras.utils.image_dataset_from_directory.
    Will support training, validation and test splits.
    Implementation will be added in the next commit.
    """
    pass


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.1):
    """
    Splits a tf.data.Dataset into train, validation, and test subsets.
    Default ratios produce a 70 / 10 / 20 split.
    Implementation will be added in the next commit.
    """
    pass


def prepare_dataset(dataset):
    """
    Applies caching and prefetching to a tf.data.Dataset for improved
    training throughput using tf.data.AUTOTUNE.
    Implementation will be added in the next commit.
    """
    pass
