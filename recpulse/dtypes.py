from typing import Any, Literal

import numpy as np
from pydantic import StrictInt

import recpulse.activations as activations
import recpulse.differentiated_activations as d_activations
import recpulse.differentiated_losses as d_losses
import recpulse.losses as losses

PRECISIONS = float | np.float16 | np.float32 | np.float64
TENSOR_TYPE = np.ndarray[PRECISIONS, Any]
SHAPE_TYPE = tuple[StrictInt]

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

STR2DACTIVATION = {
    "linear": d_activations.linear,
    "sigmoid": d_activations.sigmoid,
    "softmax": d_activations.softmax,
    "relu": d_activations.relu,
    "leaky_relu": d_activations.leaky_relu,
    "parametric_relu": d_activations.parametric_relu,
    "tanh": d_activations.tanh,
    "arctan": d_activations.arctan,
    "elu": d_activations.elu,
    "swish": d_activations.swish,
}

LOSSES = Literal[
    "MSE",
    "MAE",
    "multiclass_cross_entropy",
    "binary_cross_entropy",
]
STR2LOSS = {
    "MSE": losses.mse,
    "MAE": losses.mae,
    "multiclass_cross_entropy": losses.cross_entropy,
    "binary_cross_entropy": losses.binary_cross_entropy,
}
STR2DLOSS = {
    "MSE": d_losses.mse,
    "MAE": d_losses.mae,
    "multiclass_cross_entropy": d_losses.cross_entropy,
    "binary_cross_entropy": d_losses.binary_cross_entropy,
}

OPTIMIZERS = Literal["SGD"]

METRICS = Literal[
    "MSE",
    "MAE",
    "multiclass_cross_entropy",
    "binary_cross_entropy",
    "accuracy",
    "precision",
    "recall",
    "f1-score",
]
