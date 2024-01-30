import numpy as np

from classes.layers import Dense


class Sequential:
    """Sequential model class."""

    def __init__(
        self,
        input_shape: tuple,
        layers: list[Dense],
        precision: int = 16,
    ):
        self.layers = layers
        self.precision = precision
        self.weights = np.zeros(shape=(self.size, self.size), dtype=object)

    @property
    def size(self) -> int:
        """Returns numer of layers"""
        return len(self.layers)
