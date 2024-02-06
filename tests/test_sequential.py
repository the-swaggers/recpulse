import classes.layers as layers
from classes.models import Sequential


class TestClassSequential:
    def test_model_inti(self):
        """Test Sequential initialization."""

        model_1 = Sequential(
            input_shape=(8,),
            layers=[
                layers.Dense(4),
                layers.Dense(8),
                layers.Dense(2),
            ],
        )

        model_1.compile()
