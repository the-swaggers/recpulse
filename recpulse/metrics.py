import numpy as np

from recpulse.dtypes import METRICS, STR2LOSS

PRECISIONS = float | np.float16 | np.float32 | np.float64


def binary_class_confusion_matrix(preds: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    """Create a confusion matrix of shape (2, 2)."""

    if preds.shape != outputs.shape:
        raise ValueError("Different sizes of inputs and outputs datasets!")

    preds = np.rint(preds)

    matrix = np.zeros(shape=(2, 2), dtype=int)

    data_len = len(preds)

    for sample in range(data_len):
        matrix[preds[sample]][outputs[sample]] += 1

    return matrix


def multiclass_confusion_matrix(preds: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    """Create a confusion matrix of shape (number_of_classes, number_of_classes)."""

    if preds.shape != outputs.shape:
        raise ValueError("Different sizes of inputs and outputs datasets!")

    number_of_classes = preds.shape[1]

    matrix = np.zeros(shape=(number_of_classes, number_of_classes), dtype=int)

    data_len = len(preds)

    for sample in range(data_len):
        matrix[np.argmax(preds[sample])][outputs[sample]] += 1

    return matrix


def metric(preds: np.ndarray, outputs: np.ndarray, metric: METRICS) -> PRECISIONS:
    """Evaluate model using the given metrics."""

    if preds.shape != outputs.shape:
        raise ValueError("Different sizes of inputs and outputs datasets!")

    shape = preds.shape

    if metric in ["MSE", "MAE", "multiclass_cross_entropy", "binary_cross_entropy"]:
        data_len = len(preds)
        total: PRECISIONS = 0.0

        for sample in range(data_len):
            total += STR2LOSS[metric](preds[sample], outputs[sample])

        return total / data_len

    if len(shape) > 2 or len(shape) < 2:
        raise ValueError("This metric supports only a shape of (d, n)")

    if shape[1] == 1:
        matrix = binary_class_confusion_matrix(preds, outputs)
    else:
        matrix = multiclass_confusion_matrix(preds, outputs)

    match metric:
        case "accuracy":
            return np.trace(matrix) / np.sum(matrix)
        case "precision":
            pass
        case "recall":
            pass
        case "f1-score":
            pass
        case _:
            raise ValueError("Incorrect metric is used.")
