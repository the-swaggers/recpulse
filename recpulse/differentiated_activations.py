import numpy as np

import recpulse.activations as activations
from recpulse.np_dtypes import TENSOR_TYPE


def reshape(matrix: TENSOR_TYPE, shape: tuple) -> TENSOR_TYPE:
    """Reshape Jacobi matrix to appropriate tensor."""
    return matrix.reshape(shape + shape)


def linear(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of linear activation function."""
    shape = x.shape
    matrix = np.identity(x.size)
    return reshape(matrix, shape)


def sigmoid(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of sigmoid activation function."""
    shape = x.shape
    x = x.flatten()
    matrix = np.diag(activations.sigmoid(x) * (1 - activations.sigmoid(x)))
    return reshape(matrix, shape)


def softmax(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of softmax activation function."""
    shape = x.shape
    x = x.flatten()
    activated = activations.softmax(x)
    matrix = np.diag(activated) - np.tensordot(activated, activated, axes=0)
    return reshape(matrix, shape)


def relu(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of ReLU activation function."""
    shape = x.shape
    x = x.flatten()
    matrix = np.diag(np.where(x >= 0, 1, 0))
    return reshape(matrix, shape)


def leaky_relu(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of leaky ReLU activation function."""
    shape = x.shape
    x = x.flatten()
    matrix = np.diag(np.where(x >= 0, 1, 0.1))
    return reshape(matrix, shape)


def parametric_relu(x: TENSOR_TYPE, alpha: float = 0) -> TENSOR_TYPE:
    """Derivative of parametric ReLU activation function."""
    shape = x.shape
    x = x.flatten()
    matrix = np.diag(np.where(x >= 0, 1, alpha))
    return reshape(matrix, shape)


def tanh(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of tanh activation function."""
    shape = x.shape
    x = x.flatten()
    matrix = np.diag(1 - np.tanh(x) ** 2)
    return reshape(matrix, shape)


def arctan(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of arctan activation function."""
    shape = x.shape
    x = x.flatten()
    matrix = np.diag(1 / (1 + x**2))
    return reshape(matrix, shape)


def elu(x: TENSOR_TYPE, alpha: float = 1) -> TENSOR_TYPE:
    """Derivative of ELU activation function."""
    shape = x.shape
    x = x.flatten()
    matrix = np.diag(np.where(x >= 0, 1, -alpha * np.exp(-x)))
    return reshape(matrix, shape)


def swish(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of swish activation function."""
    shape = x.shape
    x = x.flatten()
    matrix = np.diag(activations.sigmoid(x) * (1 + x * (1 - activations.sigmoid(x))))
    return reshape(matrix, shape)
