from typing import Any, Callable

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from recpulse.dtypes import ACTIVATIONS, STR2ACTIVATION, STR2DACTIVATION
from recpulse.np_dtypes import PRECISIONS, SHAPE_TYPE, TENSOR_TYPE


class Dense(BaseModel):
    """Dense layer class.

    A fully connected layer where each input neuron connects to all output
    neurons, with biases added.

    Args:
        output_shape (SHAPE_TYPE | int): The desired output shape of the layer,
            specified as a tuple of integers (e.g., (10,) for 10 output neurons).
            If an integer is provided, it's converted to a tuple.
        input_shape (SHAPE_TYPE | None, optional): The input shape expected by the
            layer. If None, weights will be initialized upon the first call
            to the `propagate` method. Defaults to None.
        activation (ACTIVATIONS, optional): The activation function to be
            applied. Can be one of the supported strings defined in the
            `ACTIVATIONS` literal type. Defaults to "linear".
        name (StrictStr | None, optional): An optional name for the layer.
            Defaults to None.

    Attributes:
        model_config (ConfigDict): Internal configuration for the Pydantic model.
        output_shape (SHAPE_TYPE): The finalized output shape of the layer.
        input_shape (SHAPE_TYPE | None): The finalized input shape of the layer.
        name (StrictStr | None): The name of the layer.
        activation (Callable[[Any], TENSOR_TYPE]): The callable activation function.
        d_activation (Callable[[Any], TENSOR_TYPE]): The callable derivative
            of the activation function.
        _weights (TENSOR_TYPE | None): The internal weight matrix of the layer.

    Raises:
        ValueError: If invalid values for `output_shape`, `input_shape`,
            `standard_deviation` in `initialize_weights`, or
            `input_shape` in `propagate` are provided.
        AttributeError: If `weights` is accessed before initialization.
    """

    model_config = ConfigDict(validate_assignment=True)

    output_shape: SHAPE_TYPE = Field(frozen=True)
    input_shape: SHAPE_TYPE | None = None
    name: StrictStr | None = None
    activation: Callable[[Any], TENSOR_TYPE] = Field(frozen=True)
    d_activation: Callable[[Any], TENSOR_TYPE] = Field(frozen=True)
    _weights: TENSOR_TYPE | None = None

    @field_validator("output_shape", mode="after")
    @classmethod
    def validate_output_shape(cls, shape: SHAPE_TYPE) -> SHAPE_TYPE:
        """Validates the provided output shape for the Dense layer.

        Args:
            shape (SHAPE_TYPE): The proposed output shape for the layer.

        Returns:
             SHAPE_TYPE: The validated output shape.

        Raises:
            ValueError: If the output shape is empty, has more than one dimension,
                contains a non-integer value, or has a non-positive output size.
        """
        if len(shape) == 0:
            raise ValueError("The output shape must contain output neurons.")
        if len(shape) > 1:
            raise ValueError("Dense layer can only have output of shape (n, ).")
        if not isinstance(shape[0], int):
            raise ValueError("Output size must be integer.")
        if shape[0] <= 0:
            raise ValueError("Output size must be positive.")

        return shape

    @field_validator("input_shape", mode="after")
    @classmethod
    def validate_input_shape(cls, shape: SHAPE_TYPE) -> SHAPE_TYPE | None:
        """Validates the provided input shape for the Dense layer.

        Args:
            shape (SHAPE_TYPE): The proposed input shape for the layer.

        Returns:
            SHAPE_TYPE | None: The validated input shape as a tuple, or None if the
                input shape is not specified.

        Raises:
             ValueError: If the input shape is empty, has more than one dimension,
                 contains a non-integer value, or has a non-positive input size.
        """
        if shape is None:
            return None
        if len(shape) == 0:
            raise ValueError("The output shape must contain output neurons.")
        if len(shape) > 1:
            raise ValueError("Dense layer can only have output of shape (n, ).")
        if not isinstance(shape[0], int):
            raise ValueError("Input size must be integer.")
        if shape[0] <= 0:
            raise ValueError("Input size must be positive.")

        return shape

    @model_validator(mode="after")  # type: ignore
    def validate_model_params(self) -> None:
        """Ensures consistency between the input shape and weight matrix dimensions.

        Raises:
            ValueError: If the input shape (+1 for bias) does not match the
                number of columns in the initialized weight matrix.
        """
        if self.input_shape is not None and self._weights is not None:
            if self.input_shape[0] + 1 != self._weights.shape[1]:
                raise ValueError("Input shape does not match weights shape.")

    def __init__(
        self,
        output_shape: SHAPE_TYPE | StrictInt,
        input_shape: SHAPE_TYPE | None = None,
        activation: ACTIVATIONS = "linear",
        name: StrictStr | None = None,
    ) -> None:
        """Initializes a Dense layer instance.

        Args:
            output_shape (SHAPE_TYPE | StrictInt): The desired output shape of the layer.
                If an integer is provided, it's converted to a tuple (e.g., 10 becomes (10,)).
            input_shape (SHAPE_TYPE | None, optional): The expected input shape. If
                None, weights will be initialized later during the first call
                to the `propagate` method. Defaults to None.
            activation (ACTIVATIONS, optional): The activation function to use.
                Must be one of the supported strings defined in the `ACTIVATIONS`
                literal type. Defaults to "linear".
            name (StrictStr | None, optional): An optional name for the layer.
                Defaults to None.
        """
        if isinstance(output_shape, int):
            output_shape = (output_shape,)
        super().__init__(
            output_shape=output_shape,
            input_shape=input_shape,
            name=name,
            activation=STR2ACTIVATION[activation],  # type: ignore
            d_activation=STR2DACTIVATION[activation],  # type: ignore
        )

        if input_shape is not None:
            self.initialize_weights()

    def initialize_weights(
        self,
        input_shape: SHAPE_TYPE | None = None,
        mean: PRECISIONS | None = None,
        standard_deviation: PRECISIONS | None = None,
    ) -> None:
        """Initializes the weight matrix for the Dense layer.

        Weights are randomly generated using a normal distribution. If no input shape
        is provided, it checks if the `input_shape` attribute is set.

        Args:
            input_shape (SHAPE_TYPE | None, optional): The input shape for the layer.
                Used if not already defined in the layer's `input_shape` attribute.
                Defaults to None.
            mean (PRECISIONS | None, optional): The mean of the normal distribution
                for weight initialization. Defaults to 0.
            standard_deviation (PRECISIONS | None, optional): The standard deviation of
                the normal distribution. Defaults to 1.

        Raises:
            ValueError: If an input shape is not provided and cannot be inferred
                from the `input_shape` attribute.
            ValueError: If the standard deviation is less than or equal to zero.
        """
        if self._weights is not None:
            raise AttributeError("Weights are already initialized.")

        if self.input_shape is None:
            if input_shape is None:
                raise ValueError(
                    "input_shape is not specified neither in the function execution, "
                    "nor in the class's instance"
                )
            self.input_shape = input_shape

        if standard_deviation is None:
            standard_deviation = 1
        elif standard_deviation <= 0:
            raise ValueError("standard deviation must be greater than zero")

        if mean is None:
            mean = 0

        self._weights = np.random.normal(
            mean,
            standard_deviation,
            size=(self.output_shape[0], self.input_shape[0] + 1),  # type: ignore
        )

    def propagate(self, inputs: TENSOR_TYPE) -> TENSOR_TYPE:
        """Propagates an input tensor through the Dense layer.

        Args:
            inputs (TENSOR_TYPE): The input array representing data for the layer.

        Returns:
            TENSOR_TYPE: The output array after applying weights and the activation function.

        Raises:
            ValueError: If weights or the activation function haven't been initialized.
        """
        if self._weights is None:
            raise ValueError("Weights aren't initialized!")
        if self.activation is None:
            raise ValueError("Activation function isn't initialized!")

        # add 1 so that the bias doesn't need any additional code apart from matmul
        modified_input = np.concatenate((inputs, [1]), axis=0)  # type: ignore

        # get new layer
        propagated = np.dot(self._weights, modified_input)

        return self.activation(propagated)

    @property
    def initialized(self) -> bool:
        """Indicates whether the layer's weights have been initialized.

        Returns:
            bool: True if weights are initialized, False otherwise.
        """
        return self._weights is not None

    def back_propagate(
        self,
        error: TENSOR_TYPE,
        inputs: TENSOR_TYPE,
        learning_rate: PRECISIONS = 0.001,
        tune: bool = True,
    ) -> PRECISIONS:
        """Performs backpropagation for the Dense layer.

        Calculates the error gradient, updates weights (if `tune` is True), and
        returns the error to be propagated to the previous layer.

        Args:
            error (TENSOR_TYPE): The error gradient from the subsequent layer.
            inputs (TENSOR_TYPE): The input data used in forward propagation.
            learning_rate (PRECISIONS, optional): The learning rate to apply during
                weight updates. Defaults to 0.001.
            tune (bool, optional): Controls whether to update the layer's weights.
                Defaults to True.

        Returns:
            PRECISIONS: The error value to be propagated to the previous layer.

        Raises:
            ValueError: If the layer's weights haven't been initialized.
        """
        if self._weights is None:
            raise ValueError("Weights aren't initialized!")

        modified_input = np.concatenate((inputs, [1]), axis=0)  # type: ignore
        pred = np.dot(self._weights, modified_input)
        propagated_error = np.dot(np.dot(error, self._weights)[:-1], self.d_activation(inputs))

        if tune:
            dW = np.tensordot(np.dot(error, self.d_activation(pred)), modified_input, axes=0)
            self._weights += learning_rate * dW

        return propagated_error
