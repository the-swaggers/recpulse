import numpy as np

from recpulse.np_dtypes import PRECISIONS, TENSOR_TYPE


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
        if not (0 <= x[index] <= 1 and 0 <= y[index] <= 1):
            raise ValueError(
                "Both predictions and outputs must be within range of [0; 1]. "
                "You can use softmax to deal with it."
            )
        result -= y[index] * np.log(x[index])

    return result


def binary_cross_entropy(x: TENSOR_TYPE, y: TENSOR_TYPE) -> PRECISIONS:
    """Binary Cross Entropy loss function."""

    if x.shape != y.shape:
        raise ValueError("Incompatible tensors.")
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
