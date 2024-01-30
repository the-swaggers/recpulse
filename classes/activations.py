def linear(x: np.array) -> np.array:
    """Linear activation function."""
    return x


def sigmoid(x: np.array) -> np.array:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def softmax(x: np.array) -> np.array:
    """Softmax activation function."""
    exponentiated = np.exp(x)
    return exponentiated / np.sum(exponentiated)


def relu(x: np.array) -> np.array:
    """ReLU activation function."""
    return np.where(x >= 0, x, 0)


def leaky_relu(x: np.array) -> np.array:
    """Leaky ReLU activation function."""
    return np.where(x >= 0, x, 0.1 * x)


def parametric_relu(x: np.array, alpha: float = 0) -> np.array:
    """Parametric ReLU activation function."""
    return np.where(x >= 0, x, alpha * x)


def binary_step(x: np.array) -> np.array:
    """Binary step activation function."""
    return np.where(x >= 0, 1, 0)


def tanh(x: np.array) -> np.array:
    """Tanh activation function."""
    return np.tanh(x)


def arctan(x: np.array) -> np.array:
    """Arctan activation function."""
    return np.arctan(x)


def elu(x: np.array, alpha: float = 1) -> np.array:
    """ELU activation function."""
    return np.where(x >= 0, x, alpha * (np.exp(-x) - 1))


def swish(x: np.array) -> np.array:
    """Swish activation function."""
    return x / (1 + np.exp(-x))
