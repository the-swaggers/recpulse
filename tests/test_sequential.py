import pytest

import classes.layers as layers
from classes.models import Sequential


class TestClassSequential:
    def test_model_init(self):
        """Test Sequential initialization."""

        model_1 = Sequential(
            input_shape=(2,),
            layers=[
                layers.Dense(4),
                layers.Dense(6),
                layers.Dense(12),
            ],
        )
        model_1.compile()

        model_2 = Sequential(
            input_shape=(4,),
            layers=[
                layers.Dense(1),
                layers.Dense(5),
                layers.Dense(198),
            ],
        )
        model_2.compile()

        with pytest.raises(Exception):
            Sequential(
                input_shape=(-4,),
                layers=[
                    layers.Dense(1),
                ],
            )
        with pytest.raises(Exception):
            Sequential(
                input_shape=16,
                layers=[
                    layers.Dense(1),
                ],
            )
        with pytest.raises(Exception):
            Sequential(
                input_shape=2.2,
                layers=[
                    layers.Dense(1),
                ],
            )
