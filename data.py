import numpy as np

C = 9
D = 7
E = 7
F = 7


def getMatrixSize():
    # index: 197797
    """
    Returns the size of the matrix based on students index number,
    in my case 1297
    """
    return 1200 + 10 * C + D


def getMatrix(size: int, a1: float, a2: float, a3: float):
    """
    Returns matrix with five diagonals with
    symmetrical values, a1, a2, a3
    """
    matrix = np.zeros((size, size))
    for i in range(size):
        matrix[i, i] = a1
        if i + 1 < size:
            matrix[i, i + 1] = a2
            matrix[i + 1, i] = a2
        if i + 2 < size:
            matrix[i, i + 2] = a3
            matrix[i + 2, i] = a3

    return matrix


def getMatrixTaskA(size: int):
    """
    Returns matrix with five diagonals with
    symmetrical values, a1, a2, a3
    a1 = 5+e
    a2 = a3 = -1
    """
    return getMatrix(size, 5 + E, -1, -1)


def getMatrixTaskC(size: int):
    """
    Returns matrix with five diagonals with
    symmetrical values, a1, a2, a3
    a1 = 3
    a2 = a3 = -1
    """
    return getMatrix(size, 3, -1, -1)


def getVectorTaskA(size: int):
    vector = np.zeros((size, 1))
    for i in range(size):
        vector[i] = np.sin(i * (F + 1))
    return vector
