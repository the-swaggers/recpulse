from typing import Any

import numpy as np

PRECISIONS = float | np.float16 | np.float32 | np.float64
TENSOR_TYPE = np.ndarray[PRECISIONS, Any]


def mse(x: TENSOR_TYPE, y: TENSOR_TYPE) -> PRECISIONS:
    """Mean Square Error loss function."""
    if x.shape != y.shape:
        raise ValueError("Incompatible tensors.")
    size = x.size

    return np.sum((x - y) ** 2) / size


def mae(x: TENSOR_TYPE, y: TENSOR_TYPE) -> PRECISIONS:
    """Mean Absolute Error loss function."""
    if x.shape != y.shape:
        raise ValueError("Incompatible tensors.")
    size = x.size

    return np.sum(abs(x - y)) / size


def cross_entropy(x: TENSOR_TYPE, y: TENSOR_TYPE) -> PRECISIONS:
    """Cross Entropy loss function."""

    if x.shape != y.shape:
        raise ValueError("Incompatible tensors.")

    indices = list(zip(*[axis.flatten() for axis in np.indices(x.shape)]))

    result = 0

    for index in indices:
        if not (0 <= x[0] <= 1 or 0 <= x[0] <= 1):
            raise ValueError(
                "Both predictions and outputs must be within range of [0; 1]. "
                "You can use softmax to deal with it."
            )
        result -= x[index] * np.log(y[index])

    return result


def binary_cross_entropy(x: TENSOR_TYPE, y: TENSOR_TYPE) -> PRECISIONS:
    """Binary Cross Entropy loss function."""

    if x.shape != y.shape:
        raise ValueError("Incompatible tensors.")
    if x.shape != (1,):
        raise ValueError(
            "Incorrect shape. If dealing with multiple classes use Cross Entropy instead."
        )
    if not (0 <= x[0] <= 1 or 0 <= x[0] <= 1):
        raise ValueError(
            "Both predictions and outputs must be within range of [0; 1]. "
            "You can use sigmoid to deal with it."
        )

    return x[0] * np.log(y[0]) + (1 - x[0]) * np.log((1 - y[0]))
