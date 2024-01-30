from pydantic import BaseModel, model_validator, StrictInt, StrictStr, ValidationError
import numpy as np


ACTIVATION_FUNCTIONS = Literal["linear", "relu", "sigmoid", "softmax"]


class Dense(BaseModel):
    """Dense layer class.

    Layer that connects all inputs to all outputs and adds biases.
    """

    output_size: StrictInt
    activation: ACTIVATION_FUNCTIONS = "linear"
    name: StrictStr | None = None
    weights: list[np.float16 | np.float32 | np.float64] | None = None

    @model_validator(mode="after")
    def validator(self) -> None:
        """Validate parameters of the layer."""

        if size <= 0:
            raise ValidationError("Output size must be greater than 0.")

    def initialize_weights(
        self, input_shape: int, mean: float | None = None, variance: float | None = None
    ) -> None:
        """Weights initializer."""

        return None
