"""Derivative loss functions.
================================================================
Provides derivative implementations of commonly used loss functions
for use in gradient-based optimization of machine learning models.
This module includes:

* mse: Calculates the derivative of the Mean Squared Error (MSE) loss function.
* mae: Calculates the derivative of the Mean Absolute Error (MAE) loss function.
* cross_entropy:  Calculates the derivative of the cross-entropy loss function
    for multi-class classification.
* binary_cross_entropy: Calculates the derivative of the binary cross-entropy
    loss function for binary classification.

These derivative functions are essential for updating model parameters during training.

================================================================
Authors: maksym-petrenko
Contact: maksym.petrenko.a@gmail.com
License: MIT
================================================================
"""
import numpy as np
from numba import njit  # type: ignore

from recpulse.losses import validate_tensors
from recpulse.np_dtypes import TENSOR_TYPE


@validate_tensors
@njit
def mse(x: TENSOR_TYPE, y: TENSOR_TYPE) -> TENSOR_TYPE:
    """Calculates the derivative of the Mean Squared Error (MSE) loss function.

    The derivative of MSE is used in gradient-based optimization algorithms to
    update model parameters. It represents the rate of change of the loss with
    respect to the model's predictions.

    Args:
        x (TENSOR_TYPE): An array of model predictions.
        y (TENSOR_TYPE): An array of true target values.

    Returns:
        TENSOR_TYPE: The gradient of the MSE loss, with the same shape as the inputs.

    Raises:
        ValueError: If the shapes of 'x' and 'y' are not compatible.
    """
    size = x.size

    return (2 * (y - x)) / size


@validate_tensors
@njit
def mae(x: TENSOR_TYPE, y: TENSOR_TYPE) -> TENSOR_TYPE:
    """Calculates the derivative of the Mean Absolute Error (MAE) loss function.

    The derivative of MAE is less sensitive to outliers compared to MSE. It
    provides a constant gradient magnitude, indicating the direction of change needed
    to minimize the loss.

    Args:
        x (TENSOR_TYPE): An array of model predictions.
        y (TENSOR_TYPE): An array of true target values.

    Returns:
        TENSOR_TYPE: The gradient of the MAE loss, with the same shape as the inputs.

    Raises:
        ValueError: If the shapes of 'x' and 'y' are not compatible.
    """
    size = x.size

    return np.sign(y - x) / size


@validate_tensors
@njit
def cross_entropy(x: TENSOR_TYPE, y: TENSOR_TYPE) -> TENSOR_TYPE:
    """Calculates the derivative of the cross-entropy loss function for multiclass classification.

    The derivative of cross-entropy is used to update model parameters during
    training. It measures how sensitive the loss is to changes in predictions.

    Args:
        x (TENSOR_TYPE): An array of model predictions (probabilities for each class).
        y (TENSOR_TYPE): An array of true target labels (one-hot encoded).

    Returns:
        TENSOR_TYPE: The gradient of the cross-entropy loss, with the same shape as the inputs.

    Raises:
        ValueError: If the shapes of 'x' and 'y' are not compatible.
        ValueError: If the values in 'x' or 'y' are outside the range [0, 1].
    """

    if not ((0 <= x).all() and (x <= 1).all() and (0 <= y).all() and (x <= 1).all()):
        raise ValueError(
            "Both predictions and outputs must be within range of [0; 1]. "
            "You can use softmax to deal with it."
        )

    return y / x


@validate_tensors
@njit
def binary_cross_entropy(x: TENSOR_TYPE, y: TENSOR_TYPE) -> TENSOR_TYPE:
    """Calculates the derivative of the binary cross-entropy loss function.

    The derivative of binary cross-entropy is used for updating model parameters in
    binary classification problems.

    Args:
        x (TENSOR_TYPE): An array of model predictions (probabilities for the positive class).
        y (TENSOR_TYPE): An array of true target labels (0 or 1).

    Returns:
        TENSOR_TYPE: The gradient of the binary cross-entropy loss.

    Raises:
        ValueError: If the shapes of 'x' and 'y' are not compatible.
        ValueError: If the values in 'x' or 'y' are outside the range [0, 1].
    """
    if x.shape != (1,):
        raise ValueError(
            "Incorrect shape. If dealing with multiple classes use Cross Entropy instead."
        )
    if not (0 <= x[0] <= 1 and 0 <= y[0] <= 1):
        raise ValueError(
            "Both predictions and outputs must be within range of [0; 1]. "
            "You can use sigmoid to deal with it."
        )

    return y / x - (1 - y) / (1 - x)
