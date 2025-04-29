import numpy as np
# import matplotlib.pyplot as plt
from scipy.linalg import lu

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


def getMatrixTaskA(size: int):
    """
    Returns matrix with five diagonals with
    symmetrical values, a1, a2, a3
    a1 = 5+e
    a2 = a3 = -1
    """
    matrix = np.zeros((size, size))
    a1 = 5 + E
    a2 = -1
    a3 = -1
    for i in range(size):
        matrix[i, i] = a1
        if i + 1 < size:
            matrix[i, i + 1] = a2
            matrix[i + 1, i] = a2
        if i + 2 < size:
            matrix[i, i + 2] = a3
            matrix[i + 2, i] = a3

    return matrix


def getVectorTaskA(size: int):
    vector = np.zeros((size, 1))
    for i in range(size):
        vector[i] = np.sin(i * (F + 1))
    return vector


def LU(A):
    """
    LU decomposition of a matrix A
    """
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        L[i, i] = 1
        for j in range(i, n):
            U[i, j] = A[i, j]
            for k in range(i):
                U[i, j] -= L[i, k] * U[k, j]
        for j in range(i + 1, n):
            L[j, i] = A[j, i]
            for k in range(i):
                L[j, i] -= L[j, k] * U[k, i]
            L[j, i] /= U[i, i]

    return L, U
