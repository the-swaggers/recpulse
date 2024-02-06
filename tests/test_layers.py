import pytest
from pydantic import ValidationError

import classes.layers as layers

ACTIVATIONS = [
    "linear",
    "sigmoid",
    "softmax",
    "relu",
    "leaky_relu",
    "parametric_relu",
    "binary_step",
    "tanh",
    "arctan",
    "elu",
    "swish",
]


class TestLayerClass:
    def test_dense_init(self):
        """Test Dense initialization"""
        layers.Dense((4,))
        layers.Dense(4)
        layers.Dense(4, input_shape=(12,))
        for activation in ACTIVATIONS:
            layers.Dense(4, activation=activation)
        layers.Dense(4, name="abc")

        with pytest.raises(ValueError):
            layers.Dense(-16)
        with pytest.raises(ValueError):
            layers.Dense(tuple())
        with pytest.raises(ValueError):
            layers.Dense((4, 4))
        with pytest.raises(ValueError):
            layers.Dense((3.1,))
        with pytest.raises(ValidationError):
            layers.Dense(4, input_shape=12)
        with pytest.raises(ValueError):
            layers.Dense(4, input_shape=tuple())
        with pytest.raises(ValueError):
            layers.Dense(4, input_shape=(-2,))
        with pytest.raises(ValueError):
            layers.Dense(4, input_shape=(2, 2))
        with pytest.raises(ValueError):
            layers.Dense(4, input_shape=(2.1,))
        with pytest.raises(Exception):
            layers.Dense(4, activation="abc")
        with pytest.raises(ValidationError):
            layers.Dense(4, name=4)
