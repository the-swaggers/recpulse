"""Activations.
================================================================
Contains:
  Functions:
    linear         : Linear activation function (pass-through).
    sigmoid        : Sigmoid activation function.
    softmax        : Softmax activation function.
    relu           : Rectified Linear Unit (ReLU) activation function.
    leaky_relu     : Leaky ReLU activation function.
    parametric_relu: Parametric ReLU activation function.
    tanh           : Hyperbolic tangent activation function.
    arctan         : Arctangent activation function.
    elu            : Exponential Linear Unit (ELU) activation function.
    swish          : Swish activation function.
================================================================
Authors: maksym-petrenko
Contact: maksym.petrenko.a@gmail.com
License: MIT
================================================================
"""
import numpy as np

from recpulse.np_dtypes import TENSOR_TYPE


def linear(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Linear activation function."""
    return x


def sigmoid(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def softmax(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Softmax activation function."""
    x = x - np.max(x)
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
