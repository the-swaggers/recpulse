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
