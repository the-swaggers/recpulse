"""Losses.
================================================================
Contains:
  Functions:
    validate_tensors: Verifies compatibility of tensor shapes.
    mse: Calculates Mean Squared Error (MSE) loss.
    mae: Calculates Mean Absolute Error (MAE) loss.
    cross_entropy: Calculates cross-entropy loss (multiclass).
    binary_cross_entropy: Calculates binary cross-entropy loss.

================================================================
Authors: maksym-petrenko
Contact: maksym.petrenko.a@gmail.com
License: MIT
================================================================
"""
import numpy as np

from recpulse.np_dtypes import PRECISIONS, TENSOR_TYPE


def validate_tensors(x: TENSOR_TYPE, y: TENSOR_TYPE) -> None:
    """Verifies that two tensors have compatible shapes.

    This function is intended to ensure that input data shapes are suitable for
    operations that require matching dimensions.

    Args:
        x (TENSOR_TYPE): The first tensor.
        y (TENSOR_TYPE): The second tensor.

    Raises:
        ValueError: If the tensors have incompatible shapes.
    """
    if x.shape != y.shape:
        raise ValueError("Incompatible tensors.")


def mse(x: TENSOR_TYPE, y: TENSOR_TYPE) -> PRECISIONS:
    """Calculates the Mean Squared Error (MSE) between predictions and targets.

    MSE is a common loss function for regression problems, measuring the average
    squared difference between predicted and true values.

    Args:
        x (TENSOR_TYPE): An array of model predictions.
        y (TENSOR_TYPE): An array of true target values.

    Returns:
        TENSOR_TYPE: The calculated MSE value.

    Raises:
        ValueError: If the shapes of 'x' and 'y' are not compatible.
    """
    validate_tensors(x, y)
    size = x.size

    return np.sum((x - y) ** 2) / size


def mae(x: TENSOR_TYPE, y: TENSOR_TYPE) -> PRECISIONS:
    """Calculates the Mean Absolute Error (MAE) between predictions and targets.

    MAE is another regression loss function, measuring the average absolute difference
    between predicted and true values. It's less sensitive to outliers than MSE.

    Args:
        x (TENSOR_TYPE): An array of model predictions.
        y (TENSOR_TYPE): An array of true target values.

    Returns:
        TENSOR_TYPE: The calculated MAE value.

    Raises:
        ValueError: If the shapes of 'x' and 'y' are not compatible.
    """
    validate_tensors(x, y)
    size = x.size

    return np.sum(abs(x - y)) / size


def cross_entropy(x: TENSOR_TYPE, y: TENSOR_TYPE) -> PRECISIONS:
    """Calculates the cross-entropy loss for multiclass classification.

    Cross-entropy is a commonly used loss function for classification problems
    with multiple categories. It measures the difference between the predicted
    probability distribution and the true probability distribution (represented as a
    one-hot encoded vector).

    Args:
        x (TENSOR_TYPE): An array of model predictions (probabilities for each class).
        y (TENSOR_TYPE): An array of true target labels (one-hot encoded).

    Returns:
        TENSOR_TYPE: The calculated cross-entropy value.

    Raises:
        ValueError: If the shapes of 'x' and 'y' are not compatible.
        ValueError: If the values in 'x' or 'y' are outside the range [0, 1].
    """
    validate_tensors(x, y)

    if not ((0 <= x <= 1).all() and (0 <= y <= 1).all()):
        raise ValueError(
            "Both predictions and outputs must be within range of [0; 1]. "
            "You can use softmax to deal with it."
        )

    return -np.sum(y * np.log(x))


def binary_cross_entropy(x: TENSOR_TYPE, y: TENSOR_TYPE) -> PRECISIONS:
    """Calculates the binary cross-entropy loss for binary classification.

    Binary cross-entropy is a specialized version of cross-entropy for problems
    with only two classes. It measures the difference between the predicted
    probability of the positive class and the true binary target label.

    Args:
        x (TENSOR_TYPE): An array of model predictions (probabilities for the positive class).
        y (TENSOR_TYPE): An array of true target labels (0 or 1).

    Returns:
        TENSOR_TYPE: The calculated binary cross-entropy value.

    Raises:
        ValueError: If the shapes of 'x' and 'y' are not compatible.
        ValueError: If the values in 'x' or 'y' are outside the range [0, 1].
    """
    validate_tensors(x, y)

    if x.shape != (1,):
        raise ValueError(
            "Incorrect shape. If dealing with multiple classes use Cross Entropy instead."
        )
    if not (0 <= x[0] <= 1 and 0 <= y[0] <= 1):
        raise ValueError(
            "Both predictions and outputs must be within range of [0; 1]. "
            "You can use sigmoid to deal with it."
        )

    return y[0] * np.log(x[0]) + (1 - y[0]) * np.log((1 - x[0]))
