import pytest
from src.neural import (
    sigmoid, sigmoid_derivative, relu, relu_derivative,
    tanh, softmax, initialize_weights, dot_product, forward_pass
)


class TestSigmoid:
    def test_zero_input(self):
        assert sigmoid(0) == pytest.approx(0.5)

    def test_large_positive(self):
        assert sigmoid(100) == pytest.approx(1.0, abs=1e-6)

    def test_large_negative(self):
        assert sigmoid(-100) == pytest.approx(0.0, abs=1e-6)

    def test_derivative_at_zero(self):
        assert sigmoid_derivative(0) == pytest.approx(0.25)


class TestReLU:
    def test_positive(self):
        assert relu(5.0) == 5.0

    def test_negative(self):
        assert relu(-3.0) == 0.0

    def test_zero(self):
        assert relu(0.0) == 0.0

    def test_derivative_positive(self):
        assert relu_derivative(2.0) == 1.0

    def test_derivative_negative(self):
        assert relu_derivative(-1.0) == 0.0


class TestSoftmax:
    def test_output_sums_to_one(self):
        result = softmax([1.0, 2.0, 3.0])
        assert sum(result) == pytest.approx(1.0)

    def test_largest_gets_highest_prob(self):
        result = softmax([1.0, 2.0, 10.0])
        assert result[2] == max(result)

    def test_empty_input(self):
        assert softmax([]) == []

    def test_uniform_input(self):
        result = softmax([1.0, 1.0, 1.0])
        assert all(p == pytest.approx(1 / 3) for p in result)


class TestDotProduct:
    def test_basic(self):
        assert dot_product([1, 2, 3], [4, 5, 6]) == 32

    def test_zeros(self):
        assert dot_product([0, 0], [1, 1]) == 0

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            dot_product([1, 2], [1])


class TestInitializeWeights:
    def test_shape_xavier(self):
        w = initialize_weights(4, 3, method="xavier")
        assert len(w) == 3
        assert len(w[0]) == 4

    def test_zeros_method(self):
        w = initialize_weights(3, 2, method="zeros")
        assert all(v == 0.0 for row in w for v in row)

    def test_he_method(self):
        w = initialize_weights(4, 3, method="he")
        assert len(w) == 3


class TestForwardPass:
    def test_sigmoid(self):
        result = forward_pass([1, 0], [0, 0], bias=0, activation="sigmoid")
        assert result == pytest.approx(0.5)

    def test_relu_negative(self):
        result = forward_pass([-1, -1], [1, 1], activation="relu")
        assert result == 0.0

    def test_invalid_activation(self):
        with pytest.raises(ValueError):
            forward_pass([1], [1], activation="unknown")
