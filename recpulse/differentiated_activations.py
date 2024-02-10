from typing import Any

import numpy as np

import recpulse.activations as activations

PRECISIONS = float | np.float16 | np.float32 | np.float64
TENSOR_TYPE = np.ndarray[PRECISIONS, Any]


def linear(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of linear activation function."""
    return 1


def sigmoid(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of sigmoid activation function."""
    return activations.sigmoid(x) * (1 - activations.sigmoid(x))


def softmax(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of softmax activation function."""
    exponentiated = np.exp(x)
    return exponentiated / np.sum(exponentiated)


def relu(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of ReLU activation function."""
    return np.where(x >= 0, x, 0)


def leaky_relu(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of leaky ReLU activation function."""
    return np.where(x >= 0, x, 0.1 * x)


def parametric_relu(x: TENSOR_TYPE, alpha: float = 0) -> TENSOR_TYPE:
    """Derivative of parametric ReLU activation function."""
    return np.where(x >= 0, x, alpha * x)


def binary_step(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of binary step activation function."""
    return np.where(x >= 0, 1, 0)


def tanh(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of tanh activation function."""
    return np.tanh(x)


def arctan(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of arctan activation function."""
    return np.arctan(x)


def elu(x: TENSOR_TYPE, alpha: float = 1) -> TENSOR_TYPE:
    """Derivative of ELU activation function."""
    return np.where(x >= 0, x, alpha * (np.exp(-x) - 1))


def swish(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of swish activation function."""
    return x / (1 + np.exp(-x))
