import numpy as np

from recpulse.layers import Dense
from recpulse.models import Sequential

data = np.load("tests/MNIST/mnist.npz")
print("Data loaded successfully")


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

model.compile(loss="MAE", learning_rate=0.1, optimizer="SGD")

history = model.fit(test_data, epochs=10)
