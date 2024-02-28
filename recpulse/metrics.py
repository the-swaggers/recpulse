"""Metrics.
================================================================
Contains:
  Functions:
    `binary_class_confusion_matrix`: Calculates confusion
        matrix for a set of predictions and expected values
    `multiclass_confusion_matrix`: Calculates confusion
        matrix for a set of predictions and expected values
    `metrics`: Calculates the given metric for a set of
        predictions and expected values
================================================================
Authors: maksym-petrenko
Contact: maksym.petrenko.a@gmail.com
License: MIT
================================================================
"""
import numpy as np

from recpulse.dtypes import METRICS, STR2LOSS
from recpulse.np_dtypes import PRECISIONS, TENSOR_TYPE


def binary_class_confusion_matrix(preds: TENSOR_TYPE, outputs: TENSOR_TYPE) -> TENSOR_TYPE:
    """Creates a confusion matrix for binary classification tasks.

    Args:
        preds (TENSOR_TYPE): An array of model predictions (probabilities or class labels).
        outputs (TENSOR_TYPE): An array of true target labels.

    Returns:
        TENSOR_TYPE: An array representing the confusion matrix with shape (2, 2).
    """
    if preds.shape != outputs.shape:
        raise ValueError("Different sizes of inputs and outputs datasets!")

    preds = np.rint(preds)
    matrix = np.zeros(shape=(2, 2), dtype=int)
    data_len = len(preds)

    for sample in range(data_len):
        matrix[preds[sample]][outputs[sample]] += 1

    return matrix


def multiclass_confusion_matrix(preds: TENSOR_TYPE, outputs: TENSOR_TYPE) -> TENSOR_TYPE:
    """Creates a confusion matrix for multiclass classification tasks.

    Args:
        preds (TENSOR_TYPE): An array of model predictions (probabilities or class labels).
        outputs (TENSOR_TYPE): An array of true target labels.

    Returns:
        TENSOR_TYPE: An array representing the confusion matrix with
            shape (number_of_classes, number_of_classes).
    """
    if preds.shape != outputs.shape:
        raise ValueError("Different sizes of inputs and outputs datasets!")

    number_of_classes = preds.shape[1]

    matrix = np.zeros(shape=(number_of_classes, number_of_classes), dtype=int)

    data_len = len(preds)

    for sample in range(data_len):
        matrix[np.argmax(preds[sample])][np.argmax(outputs[sample])] += 1

    return matrix


def metric(preds: TENSOR_TYPE, outputs: TENSOR_TYPE, metric_type: METRICS) -> PRECISIONS:
    """Calculates a specified performance metric for model evaluation.

    Args:
        preds (TENSOR_TYPE): An array of model predictions.
        outputs (TENSOR_TYPE): An array of true target labels.
        metric_type: A string indicating the desired metric from the supported options
                     ('MSE', 'MAE', 'multiclass_cross_entropy', 'binary_cross_entropy',
                     'accuracy', 'precision', 'recall', 'f1-score').

    Returns:
        The calculated value of the specified metric.
    """
    if preds.shape[0] != outputs.shape[0]:
        raise ValueError("Different sizes of inputs and outputs datasets!")

    shape = preds.shape

    if metric_type in ["MSE", "MAE", "multiclass_cross_entropy", "binary_cross_entropy"]:
        data_len = len(preds)
        total: PRECISIONS = 0.0

        for sample in range(data_len):
            total += STR2LOSS[metric_type](preds[sample], outputs[sample])

        return total / data_len

    if len(shape) > 2 or len(shape) < 2:
        raise ValueError("This metric supports only a shape of (d, n)")

    if shape[1] == 1:
        matrix = binary_class_confusion_matrix(preds, outputs)
    else:
        matrix = multiclass_confusion_matrix(preds, outputs)

    number_of_classes = len(matrix)

    match metric_type:
        case "accuracy":
            return np.trace(matrix) / np.sum(matrix)
        case "precision":
            return np.sum(np.diag(matrix) / np.sum(matrix, axis=0)) / number_of_classes
        case "recall":
            return np.sum(np.diag(matrix) / np.sum(matrix, axis=1)) / number_of_classes
        case "f1-score":
            precision = metric(preds=preds, outputs=outputs, metric_type="precision")
            recall = metric(preds=preds, outputs=outputs, metric_type="recall")
            return 2 * precision * recall / (precision + recall)
        case _:
            raise ValueError("Incorrect metric is used.")
