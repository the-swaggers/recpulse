import numpy as np
from tqdm import tqdm

from recpulse.layers import Dense
from recpulse.models import Sequential

print("Load data")
data = np.load("tests/MNIST/mnist.npz")
print("Data loaded successfully")


def one_hot(val):
    """Designed to handle 10 classes only as it'll be used in this case only."""
    result = np.zeros(10)
    result[val] = 1
    return result


def data_generator(data, key_x: str, key_y: str):
    length = len(data[key_x])
    x = data[key_x].reshape(-1, 28**2)
    y = np.zeros(shape=(length, 10), dtype=np.bool_)
    msg = f"One hot encoding {key_y}"
    for i in tqdm(range(length), desc=msg):
        y[i] = one_hot(data[key_y][i])

    return x, y


print("Reformat the data")
# train_x, train_y = data_generator(data, "train_x", "train_y")
test_x, test_y = data_generator(data, "test_x", "test_y")
print("Data reformatted successfully")

model = Sequential(
    input_shape=(784,),
    layers=[
        Dense(16, activation="sigmoid"),
        Dense(10, activation="softmax"),
    ],
)

model.compile(loss="multiclass_cross_entropy", learning_rate=0.001, optimizer="SGD")


history = model.fit(test_x, test_y, epochs=10)

print(history)

print(model.evaluate(test_x, test_y, "accuracy"))


for i in range(100):
    print(model.predict(test_x[i]))
