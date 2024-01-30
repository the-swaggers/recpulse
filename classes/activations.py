def linear(x: np.array) -> np.array:
    """Linear activation function."""
    return x


def relu(x: np.array) -> np.array:
    """ReLU activation function."""
    return np.where(x >= 0, x, 0)


def sigmoid(x: np.array) -> np.array:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def softmax(x: np.array) -> np.array:
    """Softmax activation function."""
    exponentiated = np.exp(x)
    return exponentiated / np.sum(exponentiated)
