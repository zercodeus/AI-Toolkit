"""
metrics.py — Evaluation metrics for classification and regression models.
"""


def _validate(y_true: list, y_pred: list):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if not y_true:
        raise ValueError("Input lists cannot be empty.")


def accuracy(y_true: list, y_pred: list) -> float:
    """Fraction of correct predictions."""
    _validate(y_true, y_pred)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def confusion_matrix(y_true: list, y_pred: list) -> dict:
    """
    Binary confusion matrix.

    Returns:
        dict with keys: tp, fp, tn, fn
    """
    _validate(y_true, y_pred)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision(y_true: list, y_pred: list) -> float:
    """TP / (TP + FP). Returns 0 if no positive predictions."""
    cm = confusion_matrix(y_true, y_pred)
    denom = cm["tp"] + cm["fp"]
    return cm["tp"] / denom if denom > 0 else 0.0


def recall(y_true: list, y_pred: list) -> float:
    """TP / (TP + FN). Returns 0 if no actual positives."""
    cm = confusion_matrix(y_true, y_pred)
    denom = cm["tp"] + cm["fn"]
    return cm["tp"] / denom if denom > 0 else 0.0


def f1_score(y_true: list, y_pred: list) -> float:
    """Harmonic mean of precision and recall."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def mean_squared_error(y_true: list, y_pred: list) -> float:
    """Mean squared error for regression tasks."""
    _validate(y_true, y_pred)
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)


def mean_absolute_error(y_true: list, y_pred: list) -> float:
    """Mean absolute error for regression tasks."""
    _validate(y_true, y_pred)
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)
