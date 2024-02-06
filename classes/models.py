from typing import Any

import numpy as np

from classes.layers import Dense

PRECISIONS = float | np.float16 | np.float32 | np.float64
TENSOR_TYPE = np.ndarray[PRECISIONS, Any]


class Sequential:
    """Sequential model class."""

    def __init__(self, input_shape: tuple[int], layers: list[Dense]) -> None:
        """Model initializer."""
        self.input_shape = input_shape
        self.layers = layers

    def compile(self) -> None:
        """Set input sizes and initializes weights."""
        input_shape = self.input_shape
        for layer in self.layers:
            if layer.input_shape is None:
                layer.input_shape = input_shape
            elif layer.input_shape != input_shape:
                raise ValueError("Incompatible layers' shapes")
            input_shape = layer.output_shape
            layer.initialize_weights()

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Pass inputs through all the layers and return outputs.

        Returns:
            TENSOR_TYPE: tensor with predictions.
        """

        x = inputs

        for layer in self.layers:
            x = layer.propagate(x)

        return x

    @property
    def size(self) -> int:
        """Property

        Returns:
            int: number of layers.
        """
        return len(self.layers)
