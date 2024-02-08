from typing import Any

import numpy as np

PRECISIONS = float | np.float16 | np.float32 | np.float64
TENSOR_TYPE = np.ndarray[PRECISIONS, Any]


def linear(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Linear activation function."""
    return x


def sigmoid(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def softmax(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Softmax activation function."""
    exponentiated = np.exp(x)
    return exponentiated / np.sum(exponentiated)


def relu(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """ReLU activation function."""
    return np.where(x >= 0, x, 0)


def leaky_relu(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Leaky ReLU activation function."""
    return np.where(x >= 0, x, 0.1 * x)


def parametric_relu(x: TENSOR_TYPE, alpha: float = 0) -> TENSOR_TYPE:
    """Parametric ReLU activation function."""
    return np.where(x >= 0, x, alpha * x)


def binary_step(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Binary step activation function."""
    return np.where(x >= 0, 1, 0)


def tanh(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Tanh activation function."""
    return np.tanh(x)


def arctan(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Arctan activation function."""
    return np.arctan(x)


def elu(x: TENSOR_TYPE, alpha: float = 1) -> TENSOR_TYPE:
    """ELU activation function."""
    return np.where(x >= 0, x, alpha * (np.exp(-x) - 1))


def swish(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Swish activation function."""
    return x / (1 + np.exp(-x))
