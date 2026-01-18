import json
from typing import Any

import numpy as np

import recpulse.activations as activations
import recpulse.differentiated_activations as d_activations

# load test data from file
# arrays are stored and not generated for better reproducibility of any errors
with open("tests/test_data/ndarrays4activations.json") as f:
    NDARRAYS = json.load(f)
    for key in NDARRAYS.keys():
        NDARRAYS[key] = np.array(NDARRAYS[key])


class TestClassDifferentiatedActivations:
    @staticmethod
    def match_d_functions(function, activation, parametric: bool = False, alpha: Any = None):
        """Go through all elements of each np.ndarray from all test
        sets and check activations at each point."""
        for array in NDARRAYS.values():
            if parametric:
                function_answer = function(array, alpha)
                expected = activation(array, alpha)
            else:
                function_answer = function(array)
                expected = activation(array)
            assert np.isclose(function_answer, expected).all

    @staticmethod
    def reshape(x, shape):
        """Reshape Jacobian to a more useful shape."""
        return x.reshape(shape + shape)

    def d_linear(self, x):
        """Differentiated linear function."""
        shape = x.shape
        matrix = np.identity(x.size)
        return self.reshape(matrix, shape)

    def test_d_linear(self):
        """Python tester."""
        self.match_d_functions(self.d_linear, d_activations.linear)

    def d_sigmoid(self, x):
        """Differentiated sigmoid function."""
        shape = x.shape
        x = x.flatten()
        matrix = np.diag(activations.sigmoid(x) * (1 - activations.sigmoid(x)))
        return self.reshape(matrix, shape)

    def test_d_sigmoid(self):
        """Python tester."""
        self.match_d_functions(self.d_sigmoid, d_activations.sigmoid)

    def d_softmax(self, x):
        """Differentiated softmax function."""
        shape = x.shape
        x = x.flatten()
        activated = activations.softmax(x)
        matrix = np.diag(activated) - np.tensordot(activated, activated, axes=0)
        return self.reshape(matrix, shape)

    def test_d_softamax(self):
        """Python tester."""
        self.match_d_functions(self.d_softmax, d_activations.softmax)

    def d_relu(self, x):
        """Differentiated relu function."""
        shape = x.shape
        x = x.flatten()
        matrix = np.diag(np.where(x >= 0, 1, 0))
        return self.reshape(matrix, shape)

    def test_d_relu(self):
        """Python tester."""
        self.match_d_functions(self.d_relu, d_activations.relu)

    def d_leaky_relu(self, x):
        """Differentiated leaky relu function."""
        shape = x.shape
        x = x.flatten()
        matrix = np.diag(np.where(x >= 0, 1, 0.1))
        return self.reshape(matrix, shape)

    def test_d_leaky_relu(self):
        """Python tester."""
        self.match_d_functions(self.d_leaky_relu, d_activations.leaky_relu)

    def d_parametric_relu(self, x, alpha):
        """Differentiated parametric relu function."""
        shape = x.shape
        x = x.flatten()
        matrix = np.diag(np.where(x >= 0, 1, alpha))
        return self.reshape(matrix, shape)

    def test_d_parametric_relu(self):
        """Python tester."""
        for param in [1, 2, 3, 16, 0.23523]:
            self.match_d_functions(
                self.d_parametric_relu, d_activations.parametric_relu, True, param
            )

    def d_tanh(self, x):
        """Differentiated tanh function."""
        shape = x.shape
        x = x.flatten()
        matrix = np.diag(1 - np.tanh(x) ** 2)
        return self.reshape(matrix, shape)

    def test_d_tanh(self):
        """Python tester."""
        self.match_d_functions(self.d_tanh, d_activations.tanh)

    def d_arctan(self, x):
        """Differentiated arctan function."""
        shape = x.shape
        x = x.flatten()
        matrix = np.diag(1 - np.arctan(x) ** 2)
        return self.reshape(matrix, shape)

    def test_d_arctan(self):
        """Python tester."""
        self.match_d_functions(self.d_arctan, d_activations.arctan)

    def d_elu(self, x, alpha):
        """Differentiated elu function."""
        shape = x.shape
        x = x.flatten()
        matrix = np.diag(np.where(x >= 0, 1, alpha))
        return self.reshape(matrix, shape)

    def test_d_elu(self):
        """Python tester."""
        for param in [1, 2, 3, 16, 0.23523]:
            self.match_d_functions(self.d_elu, d_activations.elu, True, param)

    def d_swish(self, x):
        """Differentiated swish function."""
        shape = x.shape
        x = x.flatten()
        matrix = np.diag(activations.sigmoid(x) * (1 + x * (1 - activations.sigmoid(x))))
        return self.reshape(matrix, shape)

    def test_d_swish(self):
        """Python tester."""
        self.match_d_functions(self.d_swish, d_activations.swish)
