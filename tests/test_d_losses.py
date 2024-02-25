import json

import numpy as np

# load test data from file
# arrays are stored and not generated for better reproducibility of any errors
with open("tests/test_data/ndarrays4activations.json") as f:
    NDARRAYS = json.load(f)
    for key in NDARRAYS.keys():
        NDARRAYS[key] = np.array(NDARRAYS[key])


class TestClassDifferentiatedLosses:
    @staticmethod
    def match_d_functions(function, activation):
        """Go through all elements of each np.ndarray from all test
        sets and check activations at each point."""
        for array in NDARRAYS.values():
            function_answer = function(array)
            expected = activation(array)
            assert np.isclose(function_answer, expected).all
