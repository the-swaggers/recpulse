import json

import numpy as np

import recpulse.activations as activations

# load test data from file
# arrays are stored and not generated for better reproducibility of any errors
with open("tests/test_data/ndarrays4activations.json") as f:
    NDARRAYS = json.load(f)
    for key in NDARRAYS.keys():
        NDARRAYS[key] = np.array(NDARRAYS[key])


class TestClassActivations:
    @staticmethod
    def match_functions(function, activation, parametric: bool = False, alpha=None):
        """Go through all elements of each np.ndarray from all test
        sets and check activations at each point."""
        for array in NDARRAYS.values():
            if parametric:
                function_answer = activation(array, alpha)
            else:
                function_answer = activation(array)
            # get a list of all indices
            indices = list(zip(*[axis.flatten() for axis in np.indices(array.shape)]))

            if parametric:
                for index in indices:
                    assert np.isclose(function(array[index], alpha), function_answer[index]).all
            else:
                for index in indices:
                    assert np.isclose(function(array[index]), function_answer[index]).all

    @staticmethod
    def linear(x):
        """Linear function"""
        return x

    def test_linear(self):
        """Python tester."""
        self.match_functions(self.linear, activations.linear)

    @staticmethod
    def sigmoid(x):
        """Linear function"""
        return 1 / (1 + np.exp(-x))

    def test_sigmoid(self):
        """Python tester."""
        self.match_functions(self.sigmoid, activations.sigmoid)

    def test_softmax(self):
        """Test softmax activation function."""
        for array in NDARRAYS.values():
            function_answer = activations.softmax(array)
            # get a list of all indices
            indices = list(zip(*[axis.flatten() for axis in np.indices(array.shape)]))

            sum = np.sum(np.exp(array))

            for index in indices:
                assert np.isclose((np.exp(array[index]) / sum), function_answer[index])

            # check whether the output is normalized
            assert np.isclose(np.sum(function_answer), 1)

    @staticmethod
    def relu(x):
        """Linear function"""
        return x if x >= 0 else 0

    def test_relu(self):
        """Python tester."""
        self.match_functions(self.relu, activations.relu)

    @staticmethod
    def leaky_relu(x):
        """Linear function"""
        return x if x >= 0 else 0.1 * x

    def test_leaky_relu(self):
        """Python tester."""
        self.match_functions(self.leaky_relu, activations.leaky_relu)

    @staticmethod
    def parametric_relu(x, alpha):
        """Linear function"""
        return x if x >= 0 else alpha * x

    def test_parametric_relu(self):
        """Python tester."""
        # check with different parameters
        self.match_functions(
            self.parametric_relu, activations.parametric_relu, parametric=True, alpha=0
        )
        self.match_functions(
            self.parametric_relu, activations.parametric_relu, parametric=True, alpha=1
        )
        self.match_functions(
            self.parametric_relu, activations.parametric_relu, parametric=True, alpha=16
        )
        self.match_functions(
            self.parametric_relu, activations.parametric_relu, parametric=True, alpha=0.23521426573
        )

    @staticmethod
    def binary_step(x):
        """Linear function"""
        return 1 if x >= 0 else 0

    def test_binary_step(self):
        """Python tester."""
        self.match_functions(self.binary_step, activations.binary_step)

    @staticmethod
    def tanh(x):
        """Linear function"""
        return np.tanh(x)

    def test_tanh(self):
        """Python tester."""
        self.match_functions(self.tanh, activations.tanh)

    @staticmethod
    def arctan(x):
        """Linear function"""
        return np.arctan(x)

    def test_arctan(self):
        """Python tester."""
        self.match_functions(self.arctan, activations.arctan)

    @staticmethod
    def elu(x, alpha):
        """Linear function"""
        return x if x >= 0 else alpha * (np.exp(-x) - 1)

    def test_elu(self):
        """Python tester."""
        # check with different parameters
        self.match_functions(self.elu, activations.elu, parametric=True, alpha=1)
        self.match_functions(self.elu, activations.elu, parametric=True, alpha=2)
        self.match_functions(self.elu, activations.elu, parametric=True, alpha=4)
        self.match_functions(self.elu, activations.elu, parametric=True, alpha=-0.1234)

    @staticmethod
    def swish(x):
        """Linear function"""
        return x / (1 + np.exp(-x))

    def test_swish(self):
        """Python tester."""
        self.match_functions(self.swish, activations.swish)
