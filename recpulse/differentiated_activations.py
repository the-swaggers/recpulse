from typing import Any

import numpy as np

import recpulse.activations as activations

PRECISIONS = float | np.float16 | np.float32 | np.float64
TENSOR_TYPE = np.ndarray[PRECISIONS, Any]


def reshape(matrix: TENSOR_TYPE, shape: tuple[int]) -> TENSOR_TYPE:
    """Reshape Jacobi matrix to appropriate tensor."""
    return matrix.reshape(shape + shape)


def linear(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of linear activation function."""
    shape = x.shape
    matrix = np.indentity(x.size)
    return reshape(matrix, shape)


def sigmoid(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of sigmoid activation function."""
    shape = x.shape
    x.flatten()
    matrix = np.diag(activations.sigmoid(x) * (1 - activations.sigmoid(x)))
    return reshape(matrix, shape)


# TODO - make softmax activation
def softmax(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of softmax activation function."""
    exponentiated = np.exp(x)
    return exponentiated / np.sum(exponentiated)


def relu(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of ReLU activation function."""
    shape = x.shape
    x.flatten()
    matrix = np.diag(np.where(x >= 0, 1, 0))
    return reshape(matrix, shape)


def leaky_relu(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of leaky ReLU activation function."""
    shape = x.shape
    x.flatten()
    matrix = np.diag(np.where(x >= 0, 1, 0.1))
    return reshape(matrix, shape)


def parametric_relu(x: TENSOR_TYPE, alpha: float = 0) -> TENSOR_TYPE:
    """Derivative of parametric ReLU activation function."""
    shape = x.shape
    x.flatten()
    matrix = np.diag(np.where(x >= 0, 1, alpha))
    return reshape(matrix, shape)


def tanh(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of tanh activation function."""
    shape = x.shape
    x.flatten()
    matrix = np.diag(1 - np.tanh(x) ** 2)
    return reshape(matrix, shape)


def arctan(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of arctan activation function."""
    shape = x.shape
    x.flatten()
    matrix = np.diag(1 / (1 + x**2))
    return reshape(matrix, shape)


def elu(x: TENSOR_TYPE, alpha: float = 1) -> TENSOR_TYPE:
    """Derivative of ELU activation function."""
    shape = x.shape
    x.flatten()
    matrix = np.diag(np.where(x >= 0, 1, -alpha * np.exp(-x)))
    return reshape(matrix, shape)


def swish(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of swish activation function."""
    shape = x.shape
    x.flatten()
    matrix = np.diag(activations.sigmoid(x) * (1 + x * (1 - activations.sigmoid(x))))
    return reshape(matrix, shape)