from typing import Any

import numpy as np
from pydantic import StrictInt

PRECISIONS = float | np.float16 | np.float32 | np.float64
TENSOR_TYPE = np.ndarray[PRECISIONS, Any]
SHAPE_TYPE = tuple[StrictInt]
