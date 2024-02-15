import numpy as np

from recpulse.models import Sequential
from recpulse.layers import Dense


data = np.load("tests/MNIST/mnist.npz")
print(data["train_x"].shape, data["train_x"].dtype)
print(data["train_y"].shape, data["train_y"].dtype)
print(data["test_x"].shape, data["test_x"].dtype)
print(data["test_y"].shape, data["test_y"].dtype)


def one_hot(val):
    """Designed to handle 10 classes only as it'll be used in this case only."""
    result = np.zeros(10)
    result[val] = 1
    return result


def train_data_generator(data):
    for i in range(len(data["train_x"])):
        yield data["train_x"][i].reshape((28**2,)), one_hot(data["train_y"][i])


def test_data_generator(data):
    for i in range(len(data["test_x"])):
        yield data["test_x"][i].reshape((28**2,)), one_hot(data["test_y"][i])


train_data = train_data_generator(data)
test_data = test_data_generator(data)

model = Sequential(
    input_shape=(784,),
    layers=[
        Dense(64, activation="relu"),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax"),
    ],
)

model.compile(loss="MSE", learning_rate=0.001, optimizer="SGD")

history = model.fit(train_data, epochs=10)
