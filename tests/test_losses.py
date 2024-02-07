import json

import numpy as np

# load test data from file
# arrays are stored and not generated for better reproducibility of any errors
with open("tests/test_data/ndarrays4losses.json") as f:
    NDARRAYS = json.load(f)
    for key in NDARRAYS.keys():
        NDARRAYS[key] = np.array(NDARRAYS[key])


class TestClassActivations:
    def test_mse(self):
        """Python tester."""
        pass
