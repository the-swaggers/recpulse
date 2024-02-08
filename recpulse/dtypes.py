from typing import Literal

import recpulse.activations as activations
import recpulse.losses as losses

ACTIVATIONS = Literal[
    "linear",
    "sigmoid",
    "softmax",
    "relu",
    "leaky_relu",
    "parametric_relu",
    "binary_step",
    "tanh",
    "arctan",
    "elu",
    "swish",
]
STR2ACTIVATION = {
    "linear": activations.linear,
    "sigmoid": activations.sigmoid,
    "softmax": activations.softmax,
    "relu": activations.relu,
    "leaky_relu": activations.leaky_relu,
    "parametric_relu": activations.parametric_relu,
    "binary_step": activations.binary_step,
    "tanh": activations.tanh,
    "arctan": activations.arctan,
    "elu": activations.elu,
    "swish": activations.swish,
}

LOSSES = Literal[
    "MSE",
    "MAE",
    "multiclass_cross_enpropy",
    "binary_cross_enpropy",
]
STR2LOSS = {
    "MSE": losses.mse,
    "MAE": losses.mae,
    "multiclass_cross_enrropy": losses.cross_entropy,
    "binary_cross_entropy": losses.binary_cross_entropy,
}

OPTIMIZERS = Literal["SGD"]
