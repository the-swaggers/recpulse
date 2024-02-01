import numpy as np

import classes.activations as activations


def test_linear():
    x = np.array([1, 2, 3])
    assert (x == activations.linear(x)).all()
    x = np.array([[1, 2], [3, 4]])
    assert (x == activations.linear(x)).all()
