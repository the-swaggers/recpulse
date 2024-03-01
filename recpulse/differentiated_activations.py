"""Differentiated activations.
================================================================
Contains:
  Derivatives (Jacobi matrices) of these functions:
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
from typing import Callable

import numpy as np
from numba import njit  # type: ignore

import recpulse.activations as activations
from recpulse.np_dtypes import TENSOR_TYPE


def reshape(func: Callable) -> Callable:
    """Reshape Jacobi matrix to appropriate tensor."""

    def wrapper(x: TENSOR_TYPE, *alpha):
        shape = x.shape
        if len(alpha) == 0:
            matrix = func(x.flatten())
        else:
            matrix = func(x.flatten(), alpha[0])
        return matrix.reshape(shape + shape)

    return wrapper


@reshape
@njit
def linear(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of linear activation function."""
    return np.identity(x.size)


@reshape
@njit
def sigmoid(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of sigmoid activation function."""
    return np.diag(activations.sigmoid(x) * (1 - activations.sigmoid(x)))


@reshape
@njit
def softmax(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of softmax activation function."""
    activated = activations.softmax(x)
    return np.subtract(np.diag(activated), np.dot(np.expand_dims(activated, axis=0), activated.T))


@reshape
@njit
def relu(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of ReLU activation function."""
    return np.diag(np.where(x >= 0, 1, 0))


@reshape
@njit
def leaky_relu(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of leaky ReLU activation function."""
    return np.diag(np.where(x >= 0, 1, 0.1))


@reshape
@njit
def parametric_relu(x: TENSOR_TYPE, alpha: float = 0) -> TENSOR_TYPE:
    """Derivative of parametric ReLU activation function."""
    return np.diag(np.where(x >= 0, 1, alpha))


@reshape
@njit
def tanh(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of tanh activation function."""
    return np.diag(1 - np.tanh(x) ** 2)


@reshape
@njit
def arctan(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of arctan activation function."""
    return np.diag(1 / (1 + x**2))


@reshape
@njit
def elu(x: TENSOR_TYPE, alpha: float = 1) -> TENSOR_TYPE:
    """Derivative of ELU activation function."""
    return np.diag(np.where(x >= 0, 1, -alpha * np.exp(-x)))


@reshape
@njit
def swish(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of swish activation function."""
    return np.diag(activations.sigmoid(x) * (1 + x * (1 - activations.sigmoid(x))))
