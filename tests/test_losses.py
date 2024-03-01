import json

import numpy as np
import pytest

import recpulse.losses as losses

# load test data from file
# arrays are stored and not generated for better reproducibility of any errors
with open("tests/test_data/ndarrays4losses.json") as f:
    NDARRAYS = json.load(f)
    for category in NDARRAYS.keys():
        for key in NDARRAYS[category].keys():
            NDARRAYS[category][key]["x"] = np.array(NDARRAYS[category][key]["x"])
            NDARRAYS[category][key]["y"] = np.array(NDARRAYS[category][key]["y"])


class TestClassActivations:
    def test_mse(self):
        """Python tester."""
        for data in NDARRAYS["regression"].values():
            loss = (data["x"] - data["y"]) ** 2 / data["x"].size

            assert np.isclose(loss, losses.mse(data["x"], data["y"])).all

    def test_mse_validation(self):
        """Python tester."""
        with pytest.raises(Exception):
            losses.mse(np.array([1, 2, 54]), np.array([0, 21]))
        with pytest.raises(Exception):
            losses.mse(np.array([2, 3]), np.array([[1, 2]]))

    def test_mae(self):
        """Python tester."""
        for data in NDARRAYS["regression"].values():
            loss = abs(data["x"] - data["y"]) / data["x"].size

            assert np.isclose(loss, losses.mae(data["x"], data["y"])).all

    def test_mae_validation(self):
        """Python tester."""
        with pytest.raises(Exception):
            losses.mae(np.array([1, 2, 3]), np.array([1, 1]))
        with pytest.raises(Exception):
            losses.mae(np.array([1, 1]), np.array([[0, 0]]))

    @pytest.mark.filterwarnings("ignore: divide by zero encountered in log")
    def test_multiclass_cross_entropy(self):
        """Python tester."""
        for data in NDARRAYS["multiclass_cross_entropy"].values():
            indices = list(zip(*[axis.flatten() for axis in np.indices(data["x"].shape)]))

            loss = 0
            for index in indices:
                loss -= data["x"][index] * np.log(data["y"][index])

            assert np.isclose(loss, losses.cross_entropy(data["x"], data["y"])).all

    def test_multiclass_cross_entropy_validation(self):
        """Python tester."""
        with pytest.raises(Exception):
            losses.cross_entropy(np.array([1, 2]), np.array([1, 1]))
        with pytest.raises(Exception):
            losses.cross_entropy(np.array([1, 1]), np.array([[0, 0]]))
        with pytest.raises(Exception):
            losses.cross_entropy(np.array([0.5, 1]), np.array([[-1, 0]]))

    @pytest.mark.filterwarnings("ignore: divide by zero encountered in log")
    def test_binary_cross_entropy(self):
        """Python tester."""
        for data in NDARRAYS["binary_cross_entropy"].values():
            loss = data["x"][0] * np.log(data["y"][0]) + (1 - data["x"][0]) * np.log(
                (1 - data["y"][0])
            )

            assert np.isclose(loss, losses.binary_cross_entropy(data["x"], data["y"])).all

    def test_binary_cross_entropy_validation(self):
        """Python tester."""
        with pytest.raises(Exception):
            losses.binary_cross_entropy(np.array([1, 0]), np.array([1, 1]))
        with pytest.raises(Exception):
            losses.binary_cross_entropy(np.array([1]), np.array([2]))
        with pytest.raises(Exception):
            losses.binary_cross_entropy(np.array([0.5]), np.array([[1]]))
