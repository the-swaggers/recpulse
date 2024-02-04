from classes.layers import Dense


class Sequential:
    """Sequential model class."""

    def __init__(
        self,
        input_shape: tuple,
        layers: list[Dense],
    ):
        self.layers = layers
        self.input_shape = input_shape

    def compile(self):
        """Set input sizes and initializes weights."""
        input_shape = self.input_shape
        for layer in self.layers:
            if layer.input_shape is None:
                layer.input_shape = input_shape
            elif layer.input_shape != input_shape:
                raise ValueError("Incompatible layers' sizes")
            input_shape = layer.output_shape
            layer.initialize_weights()

    def predict(self, inputs):
        """Pass inputs through all the layers and return outputs."""

        x = inputs

        for layer in self.layers:
            x = layer.propadate(x)

        return x

    @property
    def size(self) -> int:
        """Returns number of layers"""
        return len(self.layers)
