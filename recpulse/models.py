from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt, field_validator

from recpulse.dtypes import LOSSES, OPTIMIZERS, STR2LOSS
from recpulse.layers import Dense


class Sequential(BaseModel):
    """Sequential model class."""

    model_config = ConfigDict(validate_assignment=True)

    input_shape: tuple[StrictInt]
    layers: list[Any]
    learning_rate: StrictFloat = 0.001
    loss: Callable[[Any, Any], Any] | None = None
    optimizer: OPTIMIZERS | None = None
    _compiled: bool = False

    @field_validator("layers", mode="after")
    @classmethod
    def validate_layers(cls, layers: list[object]) -> list[object]:
        """Layers validator."""
        for layer in layers:
            if not isinstance(layer, Dense):
                raise ValueError("Wrong layers class is used")

        return layers

    @field_validator("input_shape", mode="after")
    @classmethod
    def validate_input_shape(cls, shape: tuple) -> tuple:
        """Input shape validator."""
        if len(shape) == 0:
            raise ValueError("The output shape must contain output neurons.")
        for dim in shape:
            if dim <= 0:
                raise ValueError("Input shape must contain only positive integers.")

        return shape

    def __setattr__(self, name: str, value: Any) -> None:
        """Override __setattr__ to set _compiled to False on attribute change,
        except for _compiled itself.
        """
        if name != "_compiled":
            self.__dict__["_compiled"] = False
        super().__setattr__(name, value)

    def compile(
        self, loss: LOSSES = "MSE", learning_rate: float = 0.001, optimizer: OPTIMIZERS = "SGD"
    ) -> None:
        """Set input sizes and initializes weights."""
        input_shape = self.input_shape
        for layer in self.layers:
            if layer.input_shape is None:
                layer.input_shape = input_shape
            elif layer.input_shape != input_shape:
                raise ValueError("Incompatible layers' shapes")
            input_shape = layer.output_shape
            if layer.initialized is not None:
                layer.initialize_weights()
        self.learning_rate = learning_rate
        self.loss = STR2LOSS[loss]
        self.optimizer = optimizer
        self._compiled = True

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Pass inputs through all the layers and return outputs.

        Returns:
            TENSOR_TYPE: tensor with predictions.
        """

        x = inputs

        for layer in self.layers:
            x = layer.propagate(x)

        return x