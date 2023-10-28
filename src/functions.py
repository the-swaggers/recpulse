import numpy as np


def extended_matmul(
    matrix_a: np.ndarray, matrix_b: np.ndarray, precision: np.dtype = np.float32
) -> np.ndarray:
    """Function that multiplies matrices of matrices."""

    result = np.zeros(shape=(matrix_a.shape[0], matrix_b.shape[1]), dtype=object)

    for x in range(matrix_a.shape[0]):
        for y in range(matrix_b.shape[1]):
            cell_result = np.zeros(
                shape=(matrix_a[x][0][0], matrix_b[0][y][1]), dtype=precision
            )

            assert matrix_a[x][0][1] == matrix_b[0][y][0], "Incompatible matrices"

            for index in range(matrix_a[x][0][1]):
                if matrix_a[x][index] is not None and matrix_b[index][y]:
                    cell_result += np.matmul(matrix_a[x][index], matrix_b[index][y])

            result[x][y] = cell_result

    return result
