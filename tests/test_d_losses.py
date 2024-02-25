import json

import numpy as np

# load test data from file
# arrays are stored and not generated for better reproducibility of any errors
with open("tests/test_data/ndarrays4losses.json") as f:
    NDARRAYS = json.load(f)
    for category in NDARRAYS.keys():
        for key in NDARRAYS[category].keys():
            NDARRAYS[category][key]["x"] = np.array(NDARRAYS[category][key]["x"])
            NDARRAYS[category][key]["y"] = np.array(NDARRAYS[category][key]["y"])


class TestClassDifferentiatedLosses:
    @staticmethod
    def match_d_functions(function, activation):
        """Go through all elements of each np.ndarray from all test
        sets and check activations at each point."""
        for category in NDARRAYS.values():
            for instance in category.values():
                function_answer = function(instance["x"], instance["y"])
                expected = activation(instance["x"], instance["y"])
                assert np.isclose(function_answer, expected).all
