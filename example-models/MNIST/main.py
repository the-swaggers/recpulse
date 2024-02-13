from recpulse.models import Sequential
from recpulse.layers import Dense

import zipfile
import os


# prepare data
if len(os.listdir("data")) == 1:
    with zipfile.ZipFile("data/dataset.zip", 'r') as zip_ref:
        zip_ref.extractall("data")


# build model
model = Sequential(
    input_shape=(784, ),
    layers=[
        Dense(64, activation="relu"),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax"),
    ]
)

model.compile(
    loss="multiclass_cross_enpropy",
    learning_rate=0.001,
    optimizer="SGD"
)


