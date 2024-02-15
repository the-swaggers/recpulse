import numpy as np

data = np.load("tests/performance_requiring_tests/MNIST/mnist.npz")
print(data["train_x"].shape, data["train_x"].dtype)
print(data["train_y"].shape, data["train_y"].dtype)
print(data["test_x"].shape, data["test_x"].dtype)
print(data["test_y"].shape, data["test_y"].dtype)
