"""
neural.py — Activation functions and neural network helpers.
"""

import math


def sigmoid(x: float) -> float:
    """Sigmoid activation: maps any value to (0, 1)."""
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x: float) -> float:
    """Derivative of sigmoid — used in backpropagation."""
    s = sigmoid(x)
    return s * (1 - s)


def relu(x: float) -> float:
    """Rectified Linear Unit: max(0, x)."""
    return max(0.0, x)


def relu_derivative(x: float) -> float:
    """Derivative of ReLU."""
    return 1.0 if x > 0 else 0.0


def tanh(x: float) -> float:
    """Hyperbolic tangent activation: maps to (-1, 1)."""
    return math.tanh(x)


def tanh_derivative(x: float) -> float:
    """Derivative of tanh."""
    return 1 - math.tanh(x) ** 2


def softmax(values: list) -> list:
    """
    Softmax function — converts logits to probabilities.
    Uses numerically stable version (subtracts max).
    """
    if not values:
        return []
    max_val = max(values)
    exps = [math.exp(v - max_val) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def initialize_weights(n_inputs: int, n_outputs: int, method: str = "xavier") -> list:
    """
    Initialize weights for a fully connected layer.

    Args:
        n_inputs:  Number of input neurons.
        n_outputs: Number of output neurons.
        method:    'xavier' | 'he' | 'zeros'

    Returns:
        2D list of shape (n_outputs, n_inputs).
    """
    import random

    if method == "zeros":
        return [[0.0] * n_inputs for _ in range(n_outputs)]
    elif method == "he":
        std = math.sqrt(2.0 / n_inputs)
    else:  # xavier
        std = math.sqrt(2.0 / (n_inputs + n_outputs))

    return [
        [random.gauss(0, std) for _ in range(n_inputs)]
        for _ in range(n_outputs)
    ]


def dot_product(vec_a: list, vec_b: list) -> float:
    """Compute the dot product of two vectors."""
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have the same length.")
    return sum(a * b for a, b in zip(vec_a, vec_b))


def forward_pass(inputs: list, weights: list, bias: float = 0.0,
                 activation: str = "sigmoid") -> float:
    """
    Single neuron forward pass.

    Args:
        inputs:     Input values.
        weights:    Weight values (same length as inputs).
        bias:       Bias term.
        activation: Activation function name ('sigmoid' | 'relu' | 'tanh').

    Returns:
        Activated output value.
    """
    activations = {"sigmoid": sigmoid, "relu": relu, "tanh": tanh}
    if activation not in activations:
        raise ValueError(f"Unknown activation '{activation}'. Choose from: {list(activations)}")

    z = dot_product(inputs, weights) + bias
    return activations[activation](z)
