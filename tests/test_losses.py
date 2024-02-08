import json

import numpy as np
import pytest

import classes.losses as losses

# load test data from file
# arrays are stored and not generated for better reproducibility of any errors
with open("tests/test_data/ndarrays4losses.json") as f:
    NDARRAYS = json.load(f)
    for category in NDARRAYS.keys():
        for key in NDARRAYS[category].keys():
            print(NDARRAYS[category][key])
            NDARRAYS[category][key]["x"] = np.array(NDARRAYS[category][key]["x"])
            NDARRAYS[category][key]["y"] = np.array(NDARRAYS[category][key]["y"])


class TestClassActivations:
    def test_mse(self):
        """Python tester."""
        for data in NDARRAYS["regression"].values():
            loss = (data["x"] - data["y"]) ** 2 / data["x"].size

            assert np.isclose(loss, losses.mse(data["x"], data["y"])).all

    def test_mae(self):
        """Python tester."""
        for data in NDARRAYS["regression"].values():
            loss = abs(data["x"] - data["y"]) / data["x"].size

            assert np.isclose(loss, losses.mse(data["x"], data["y"])).all

    @pytest.mark.filterwarnings("ignore: divide by zero encountered in log")
    def test_multiclass_cross_entropy(self):
        """Python tester."""
        for data in NDARRAYS["multiclass_cross_entropy"].values():
            indices = list(zip(*[axis.flatten() for axis in np.indices(data["x"].shape)]))

            loss = 0
            for index in indices:
                loss -= data["x"][index] * np.log(data["y"][index])

            assert np.isclose(loss, losses.mse(data["x"], data["y"])).all

    @pytest.mark.filterwarnings("ignore: divide by zero encountered in log")
    def test_binary_cross_entropy(self):
        """Python tester."""
        for data in NDARRAYS["binary_cross_entropy"].values():
            loss = data["x"][0] * np.log(data["y"][0]) + (1 - data["x"][0]) * np.log(
                (1 - data["y"][0])
            )

            assert np.isclose(loss, losses.mse(data["x"], data["y"])).all
