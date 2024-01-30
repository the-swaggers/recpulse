from pydantic import BaseModel, model_validator, StrictInt, StrictStr, ValidationError
import numpy as np


ACTIVATION_FUNCTIONS = Literal["linear", "relu", "sigmoid", "softmax"]


class Dense(BaseModel):
    """Dense layer class.

    Layer that connects all inputs to all outputs and adds biases.
    """

    output_size: StrictInt
    input_size: StrictInt | None = None
    activation: ACTIVATION_FUNCTIONS = "linear"
    name: StrictStr | None = None
    _bias: float | None = None
    _weights: list[float] | None = None

    @model_validator(mode="after")
    def validator(self) -> None:
        """Validate parameters of the layer."""

        if size <= 0:
            raise ValidationError("Output size must be greater than 0.")

    def initialize_weights(
        self,
        input_size: int | None = None,
        mean: float | None = None,
        variance: float | None = None,
    ) -> None:
        """Weights initializer."""

        if self.input_size is None:
            assert (
                input_size is not None
            ), "input_size is not specified neither in the function execution, nor in the class's instance"
            self.input_size = input_size

        if variance is None:
            variance = 1
        elif variance <= 0:
            raise ValueError("variance must be greater than zero")

        if mean is None:
            mean = 0

        self._weights = np.random.normal(
            mean, variance, size=(input_size, self.output_size)
        )
        self._bias = np.random.normal(mean, variance)
