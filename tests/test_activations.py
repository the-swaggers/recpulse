import json

import numpy as np

import classes.activations as activations

# load test data from file
with open("tests/test_data/ndarrays.json") as f:
    NDARRAYS = json.load(f)
    for key in NDARRAYS.keys():
        NDARRAYS[key] = np.array(NDARRAYS[key])


class TestClassActivations:
    @staticmethod
    def match_functions(function, activation):
        for array in NDARRAYS.values():
            function_answer = activation(array)
            # get a list of all indices
            indices = list(zip(*[axis.flatten() for axis in np.indices(array.shape)]))

            for index in indices:
                assert function(array[index]) == function_answer[index]

    @staticmethod
    def linear(x):
        return x

    def test_linear(self):
        self.match_functions(self.linear, activations.linear)
