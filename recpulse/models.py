from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt, field_validator
from tqdm import tqdm

from recpulse.dtypes import LOSSES, METRICS, OPTIMIZERS, STR2DLOSS, STR2LOSS
from recpulse.layers import Dense
from recpulse.metrics import metric

PRECISIONS = float | np.float16 | np.float32 | np.float64


class Sequential(BaseModel):
    """Sequential model class."""

    model_config = ConfigDict(validate_assignment=True)

    input_shape: tuple[StrictInt]
    layers: list[Any]
    learning_rate: StrictFloat = 0.001
    loss: LOSSES | None = None
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
        if name in {"input_shape", "layers", "learning_rate", "optimizer"}:
            self._compiled = False
        super().__setattr__(name, value)

    @property
    def output_shape(self):
        return self.layers[-1].output_shape

    def compile(
        self, loss: LOSSES = "MSE", learning_rate: float = 0.001, optimizer: OPTIMIZERS = "SGD"
    ) -> None:
        """Set input shapes and initializes weights."""
        input_shape = self.input_shape
        for layer in self.layers:
            if layer.input_shape is None:
                layer.input_shape = input_shape
            elif layer.input_shape != input_shape:
                raise ValueError("Incompatible layers' shapes")
            input_shape = layer.output_shape
            if layer.initialized is False:
                layer.initialize_weights()
        self.learning_rate = learning_rate
        self.loss = loss
        self.optimizer = optimizer
        self._compiled = True

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Pass inputs through all the layers and return outputs.

        Returns:
            TENSOR_TYPE: tensor with predictions.
        """
        if not self._compiled:
            raise Exception("You must compile model before making predictions!")

        x = inputs

        for layer in self.layers:
            x = layer.propagate(x)

        return x

    def fit(self, train_x: np.ndarray, train_y: np.ndarray, epochs: int = 1) -> list[float]:
        """Optimize model's parameters depending on data."""
        if not self._compiled:
            raise Exception("You must compile model before fitting it!")

        history = []
        for epoch in range(epochs):
            metric_val = 0.0
            msg = f"Epoch {epoch+1}/{epochs}"
            data_len = len(train_x)
            for sample in tqdm(range(data_len), desc=msg):
                x, y = train_x[sample], train_y[sample]
                intermediate_results = [x]
                for layer in self.layers:
                    x = layer.propagate(x)
                    intermediate_results.append(x)

                loss = STR2LOSS[self.loss](intermediate_results[-1], y)  # type: ignore
                metric_val += loss  # type: ignore

                error = STR2DLOSS[self.loss](intermediate_results[-1], y)  # type: ignore

                for layer in range(len(self.layers) - 1, -1, -1):
                    error = self.layers[layer].back_propagate(
                        error=error,
                        inputs=intermediate_results[layer],
                        learning_rate=self.learning_rate,
                        tune=True,
                    )
                data_len += 1
            metric_val /= data_len
            print(metric_val)
            history.append(metric_val)

        return history

    def evaluate(self, x: np.ndarray, y: np.ndarray, metric_type: METRICS) -> PRECISIONS:
        """Evaluate the model."""

        if x.shape[0] != y.shape[0]:
            raise ValueError("Different sizes of input dataset sand output dataset sizes.")
        if x[0].shape != self.input_shape:
            raise ValueError("Wrong input shape.")
        if y[0].shape != self.output_shape:
            raise ValueError("Wrong output shape.")

        sum: PRECISIONS = 0.0

        for sample in range(len(x)):
            sum += metric(self.predict(x[sample]), y, metric_type=metric_type)

        sum /= len(x)

        return sum
