from typing import Literal

import recpulse.activations as activations
import recpulse.differentiated_activations as d_activations
import recpulse.losses as losses

ACTIVATIONS = Literal[
    "linear",
    "sigmoid",
    "softmax",
    "relu",
    "leaky_relu",
    "parametric_relu",
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
    "tanh": activations.tanh,
    "arctan": activations.arctan,
    "elu": activations.elu,
    "swish": activations.swish,
}

STR2ACTIVATION = {
    "linear": d_activations.linear,
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
