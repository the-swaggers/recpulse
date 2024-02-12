import numpy as np
import pytest
from pydantic import ValidationError

import recpulse.activations as activations
import recpulse.layers as layers

ACTIVATIONS = [
    "linear",
    "sigmoid",
    "softmax",
    "relu",
    "leaky_relu",
    "parametric_relu",
    "tanh",
    "arctan",
    "elu",
    "swish",
]


class TestLayerClass:
    def test_dense_init(self):
        """Test Dense initialization."""
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

    def test_dense_assignment(self):
        """Test Dense assignment."""
        layer1 = layers.Dense(4, input_shape=(2,), name="abc", activation="relu")
        assert layer1.output_shape == (4,)
        assert layer1.input_shape == (2,)
        assert layer1.name == "abc"

        layer2 = layers.Dense(4, input_shape=(4,), name="abc", activation="sigmoid")
        with pytest.raises(Exception):
            layer2.input_shape = (8,)

        layer3 = layers.Dense(4, input_shape=(2,), name="abc", activation="softmax")
        layer3.name = "aaa"
        assert layer3.name == "aaa"

        layer4 = layers.Dense(4)
        layer4.input_shape = (2,)
        layer4.input_shape = (3,)
        layer4.initialize_weights()
        with pytest.raises(Exception):
            layer2.input_shape = (8,)

        layer5 = layers.Dense(4, input_shape=(2,), name="abc", activation="elu")
        with pytest.raises(Exception):
            layer5.activation = activations.relu

        layer6 = layers.Dense(4, input_shape=(2,), name="abc", activation="elu")
        assert layer6._weights.shape == (4, 3)
        with pytest.raises(AttributeError):
            layer6.initialize_weights()

    def test_dense_propagation(self):
        """Test Dense propagate function."""
        x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        layer1 = layers.Dense(4, input_shape=(8,), activation="softmax")
        y1 = layer1.propagate(x1)
        assert y1.shape == (4,)
        assert (y1 > 0).all
        assert np.isclose(np.sum(y1), 1)

        x2 = np.array([-12.113, 125.842, 634.735, 235.876, -532.645, 3424.75, -234.234, 1.23])
        layer2 = layers.Dense(12, input_shape=(8,), activation="relu")
        y2 = layer2.propagate(x2)
        assert y2.shape == (12,)
        assert (y2 > 0).all
