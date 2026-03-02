"""
preprocessing.py — Data preprocessing utilities for ML pipelines.
"""

import math
import random


def normalize(data: list) -> list:
    """
    Min-max normalization to scale values into [0, 1].

    Args:
        data: List of numeric values.

    Returns:
        Normalized list.
    """
    if not data:
        return []
    min_val, max_val = min(data), max(data)
    if min_val == max_val:
        return [0.0] * len(data)
    return [(x - min_val) / (max_val - min_val) for x in data]


def standardize(data: list) -> list:
    """
    Z-score standardization: (x - mean) / std.

    Returns:
        Standardized list with mean ≈ 0 and std ≈ 1.
    """
    if not data:
        return []
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std = math.sqrt(variance)
    if std == 0:
        return [0.0] * len(data)
    return [(x - mean) / std for x in data]


def train_test_split(data: list, test_size: float = 0.2, shuffle: bool = True,
                     seed: int = None) -> tuple:
    """
    Split a dataset into training and test sets.

    Args:
        data:      List of samples.
        test_size: Fraction for the test set (0 < test_size < 1).
        shuffle:   Randomly shuffle before splitting.
        seed:      Random seed for reproducibility.

    Returns:
        (train_data, test_data) tuple.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    dataset = data[:]
    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(dataset)
    split_idx = int(len(dataset) * (1 - test_size))
    return dataset[:split_idx], dataset[split_idx:]


def one_hot_encode(labels: list) -> tuple:
    """
    One-hot encode a list of categorical labels.

    Returns:
        (encoded, classes) where encoded is a list of binary lists
        and classes is the sorted list of unique labels.
    """
    classes = sorted(set(labels))
    class_index = {c: i for i, c in enumerate(classes)}
    encoded = []
    for label in labels:
        vec = [0] * len(classes)
        vec[class_index[label]] = 1
        encoded.append(vec)
    return encoded, classes


def flatten(nested: list) -> list:
    """Recursively flatten a nested list."""
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
