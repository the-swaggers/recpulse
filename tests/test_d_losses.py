import json

import numpy as np

import recpulse.differentiated_losses as d_losses

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
    def match_d_functions(function, loss, category):
        """Go through all elements of each np.ndarray from all test
        sets and check activations at each point."""
        for instance in NDARRAYS[category].values():
            function_answer = function(instance["x"], instance["y"])
            expected = loss(instance["x"], instance["y"])
            assert np.isclose(function_answer, expected).all

    @staticmethod
    def d_mse(x, y):
        """Differentiated MSE loss function."""
        return (2 * (y - x)) / x.size

    def test_d_mse(self):
        """Python tester."""
        self.match_d_functions(self.d_mse, d_losses.mse, "regression")

    @staticmethod
    def d_mae(x, y):
        """Differentiated MAE loss function."""
        return np.sign(y - x) / x.size

    def test_d_mae(self):
        """Python tester."""
        self.match_d_functions(self.d_mae, d_losses.mae, "regression")

    @staticmethod
    def d_cross_entropy(x, y):
        return y / x

    def test_d_cross_entropy(self):
        """Python tester."""
        self.match_d_functions(
            self.d_cross_entropy, d_losses.cross_entropy, "multiclass_cross_entropy"
        )

    @staticmethod
    def d_binary_cross_entropy(x, y):
        return y / x - (1 - y) / (1 - x)

    def test_d_binary_cross_entropy(self):
        """Python tester."""
        self.match_d_functions(
            self.d_binary_cross_entropy, d_losses.binary_cross_entropy, "binary_cross_entropy"
        )
