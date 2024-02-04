import json

import numpy as np

import classes.activations as activations

# load test data from file
# arrays are stored and not generated for better reproducibility of any errors
with open("tests/test_data/ndarrays.json") as f:
    NDARRAYS = json.load(f)
    for key in NDARRAYS.keys():
        NDARRAYS[key] = np.array(NDARRAYS[key])


class TestClassActivations:
    @staticmethod
    def match_functions(function, activation):
        """Go through all elements of each np.ndarray from all test
        sets and check activations at each point."""
        for array in NDARRAYS.values():
            function_answer = activation(array)
            # get a list of all indices
            indices = list(zip(*[axis.flatten() for axis in np.indices(array.shape)]))

            for index in indices:
                assert function(array[index]) == function_answer[index]

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

    # skip softmax for now as unlike other functions it's normed, therefore,
    # values can't be tested individually
    # TODO - add softmax tests

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

    # skip parametric relu for now as unlike other functions takes two input parameters
    # TODO - add parametric relu tests

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

    # skip elu for now as unlike other functions takes two input parameters
    # TODO - add elu tests

    @staticmethod
    def swish(x):
        """Linear function"""
        return x / (1 + np.exp(-x))

    def test_elu(self):
        """Python tester."""
        self.match_functions(self.swish, activations.swish)
