import pytest
from src.metrics import (
    accuracy, confusion_matrix, precision, recall,
    f1_score, mean_squared_error, mean_absolute_error
)

Y_TRUE = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
Y_PRED = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]


class TestAccuracy:
    def test_basic(self):
        assert accuracy([1, 0, 1], [1, 0, 0]) == pytest.approx(2 / 3)

    def test_perfect(self):
        assert accuracy([1, 1, 0], [1, 1, 0]) == 1.0

    def test_all_wrong(self):
        assert accuracy([1, 1], [0, 0]) == 0.0

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            accuracy([1, 0], [1])


class TestConfusionMatrix:
    def test_all_correct(self):
        cm = confusion_matrix([1, 0, 1], [1, 0, 1])
        assert cm == {"tp": 2, "fp": 0, "tn": 1, "fn": 0}

    def test_all_false_positive(self):
        cm = confusion_matrix([0, 0], [1, 1])
        assert cm["fp"] == 2


class TestPrecision:
    def test_basic(self):
        p = precision(Y_TRUE, Y_PRED)
        assert 0 <= p <= 1

    def test_no_positive_predictions(self):
        assert precision([1, 1], [0, 0]) == 0.0


class TestRecall:
    def test_basic(self):
        r = recall(Y_TRUE, Y_PRED)
        assert 0 <= r <= 1

    def test_no_actual_positives(self):
        assert recall([0, 0], [1, 0]) == 0.0


class TestF1Score:
    def test_perfect(self):
        assert f1_score([1, 0, 1], [1, 0, 1]) == 1.0

    def test_zero(self):
        assert f1_score([1, 1], [0, 0]) == 0.0

    def test_between_precision_and_recall(self):
        p = precision(Y_TRUE, Y_PRED)
        r = recall(Y_TRUE, Y_PRED)
        f = f1_score(Y_TRUE, Y_PRED)
        assert min(p, r) <= f <= max(p, r)


class TestRegressionMetrics:
    def test_mse_perfect(self):
        assert mean_squared_error([1, 2, 3], [1, 2, 3]) == 0.0

    def test_mse_basic(self):
        assert mean_squared_error([0, 0], [1, 1]) == 1.0

    def test_mae_basic(self):
        assert mean_absolute_error([0, 2], [1, 3]) == 1.0
