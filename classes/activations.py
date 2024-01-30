import numpy as np


def linear(x: np.ndarray) -> np.ndarray:
    """Linear activation function."""
    return x


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax activation function."""
    exponentiated = np.exp(x)
    return exponentiated / np.sum(exponentiated)


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    return np.where(x >= 0, x, 0)


def leaky_relu(x: np.ndarray) -> np.ndarray:
    """Leaky ReLU activation function."""
    return np.where(x >= 0, x, 0.1 * x)


def parametric_relu(x: np.ndarray, alpha: float = 0) -> np.ndarray:
    """Parametric ReLU activation function."""
    return np.where(x >= 0, x, alpha * x)


def binary_step(x: np.ndarray) -> np.ndarray:
    """Binary step activation function."""
    return np.where(x >= 0, 1, 0)


def tanh(x: np.ndarray) -> np.ndarray:
    """Tanh activation function."""
    return np.tanh(x)


def arctan(x: np.ndarray) -> np.ndarray:
    """Arctan activation function."""
    return np.arctan(x)


def elu(x: np.ndarray, alpha: float = 1) -> np.ndarray:
    """ELU activation function."""
    return np.where(x >= 0, x, alpha * (np.exp(-x) - 1))


def swish(x: np.ndarray) -> np.ndarray:
    """Swish activation function."""
    return x / (1 + np.exp(-x))
