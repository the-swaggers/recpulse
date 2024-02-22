import numpy as np

from recpulse.np_dtypes import TENSOR_TYPE


def mse(x: TENSOR_TYPE, y: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of Mean Square Error loss function."""
    if x.shape != y.shape:
        raise ValueError("Incompatible tensors.")
    size = x.size

    return (2 * (y - x)) / size


def mae(x: TENSOR_TYPE, y: TENSOR_TYPE) -> TENSOR_TYPE:
    """Mean Absolute Error loss function."""
    if x.shape != y.shape:
        raise ValueError("Incompatible tensors.")
    size = x.size

    return np.sign(y - x) / size


def cross_entropy(x: TENSOR_TYPE, y: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of Cross Entropy loss function."""

    if x.shape != y.shape:
        raise ValueError("Incompatible tensors.")

    indices = list(zip(*[axis.flatten() for axis in np.indices(x.shape)]))

    for index in indices:
        if not (0 <= x[index] <= 1 and 0 <= y[index] <= 1):
            raise ValueError(
                "Both predictions and outputs must be within range of [0; 1]. "
                "You can use softmax to deal with it."
            )

    return y / x


def binary_cross_entropy(x: TENSOR_TYPE, y: TENSOR_TYPE) -> TENSOR_TYPE:
    """Derivative of Binary Cross Entropy loss function."""

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

    return y / x - (1 - y) / (1 - x)
