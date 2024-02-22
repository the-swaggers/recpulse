from typing import Any

from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt, field_validator
from tqdm import tqdm

from recpulse.dtypes import (
    LOSSES,
    METRICS,
    OPTIMIZERS,
    PRECISIONS,
    STR2DLOSS,
    STR2LOSS,
    TENSOR_TYPE,
)
from recpulse.layers import Dense
from recpulse.metrics import metric


class Sequential(BaseModel):
    """Defines a sequential neural network model.

    A sequential model consists of a linear stack of layers, allowing for the
    construction of common feedforward neural network architectures. This class
    provides essential methods for compiling, training, and evaluating such models.

    Args:
        input_shape: A tuple of integers defining the shape of the model's input data.
        layers: A list of `Dense` layer objects defining the model's architecture.
        learning_rate: A float specifying the learning rate for the optimizer (default: 0.001).
        loss: A member of the `LOSSES` enumeration:
            * 'MSE' (Mean Squared Error)
            * 'MAE' (Mean Absolute Error)
            * 'multiclass_cross_entropy' (Categorical Crossentropy)
            * 'binary_cross_entropy' (Binary Crossentropy)
        optimizer:  Currently supports only 'SGD' (Stochastic Gradient Descent).

    Attributes:
        model_config: A Pydantic configuration class ensuring data validation.
        _compiled:  A boolean flag indicating whether the model has been compiled.
        output_shape: A tuple representing the shape of the model's output.

    Raises:
        ValueError: If any of the following conditions occur:
            - An invalid layer type is provided in the `layers` list.
            - The `input_shape` contains non-positive dimensions.
            - Layer input and output shapes are incompatible.
    """

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
        """Ensures all layers provided in the `layers` list are valid.

        This method acts as a validator, guaranteeing that only instances of the
        `Dense` class are included within the model architecture.

        Args:
            layers: A list of layer objects intended for the sequential model.

        Raises:
            ValueError: If any layer in the `layers` list is not a `Dense` object.

        Returns:
            list[object]: The original list of layers if validation is successful.
        """
        for layer in layers:
            if not isinstance(layer, Dense):
                raise ValueError("Wrong layers class is used")

        return layers

    @field_validator("input_shape", mode="after")
    @classmethod
    def validate_input_shape(cls, shape: tuple) -> tuple:
        """Validates the provided input shape for the sequential model.

        Ensures the input shape meets these criteria:
            * Contains only positive integer dimensions.
            * Has at least one dimension (specifying output neurons).

        Args:
            shape: A tuple of integers representing the expected input shape.

        Raises:
            ValueError: If either of the following conditions are violated:
                * The input shape contains any non-positive dimensions.
                * The input shape is empty.

        Returns:
            tuple: The original input shape if validation is successful.
        """
        if len(shape) == 0:
            raise ValueError("The output shape must contain output neurons.")
        for dim in shape:
            if dim <= 0:
                raise ValueError("Input shape must contain only positive integers.")

        return shape

    def __setattr__(self, name: str, value: Any) -> None:
        """Overrides default attribute setting behavior.

        Primarily used to manage model compilation status. If any of the following
        attributes are modified, the model's compiled status is reset:
            * 'input_shape'
            * 'layers'
            * 'learning_rate'
            * 'optimizer'

        Args:
            name: The name of the attribute being set.
            value: The new value to be assigned to the attribute.
        """
        if name in {"input_shape", "layers", "learning_rate", "optimizer"}:
            self._compiled = False
        super().__setattr__(name, value)

    @property
    def output_shape(self) -> tuple[StrictInt]:
        """Gets the output shape of the sequential model.

        The output shape is in fact the output shape of the final layer in the
        model's `layers` list.

        Returns:
            tuple[StrictInt]: A tuple representing the output shape of the model.
        """
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

    def predict(self, inputs: TENSOR_TYPE) -> TENSOR_TYPE:
        """Generates predictions for the provided input data.

        Args:
            inputs: A NumPy array containing the input data. The shape of the array
                should be compatible with the model's `input_shape`.

        Returns:
            A NumPy array containing the model's predictions for the given inputs.

        Raises:
            Exception: If the model has not been compiled.
        """
        if not self._compiled:
            raise Exception("You must compile model before making predictions!")

        x = inputs

        for layer in self.layers:
            x = layer.propagate(x)

        return x

    def fit(self, train_x: TENSOR_TYPE, train_y: TENSOR_TYPE, epochs: int = 1) -> list[float]:
        """Trains the model on the provided data.

        Args:
            train_x: A NumPy array containing the training input data.
            train_y: A NumPy array containing the corresponding target outputs.
            epochs: The number of iterations to run over the training dataset.

        Returns:
            A list of floats representing the loss value for each epoch.

        Raises:
            Exception: If the model has not been compiled.
        """
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

    def evaluate(self, x: TENSOR_TYPE, y: TENSOR_TYPE, metric_type: METRICS) -> PRECISIONS:
        """Evaluates the model's performance on provided data using a specified metric.

        Args:
            x: A NumPy array containing the input data.
            y: A NumPy array containing the corresponding target outputs.
            metric_type: A member of the `METRICS` enum indicating the metric to use
                for evaluation.

        Returns:
            The calculated metric value (e.g., an accuracy score). The exact data type
            depends on the chosen metric.

        Raises:
            ValueError: If there's a mismatch between input and output shapes, or if
                the input shape is incompatible with the model.
        """
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
