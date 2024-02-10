from typing import Any

import numpy as np

PRECISIONS = float | np.float16 | np.float32 | np.float64
TENSOR_TYPE = np.ndarray[PRECISIONS, Any]


def linear(x: TENSOR_TYPE) -> TENSOR_TYPE:
    """Linear activation function."""
    return x
