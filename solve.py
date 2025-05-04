import numpy as np


def LU(A):
    """
    LU decomposition of a matrix A
    """
    # p, l, u = lu(A)
    n = A.shape[0]

    # Inicjalizacja macierzy L i U
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    # Uzupełnienie diagonali macierzy L jedynkami
    for i in range(n):
        L[i, i] = 1.0

    # Realizacja eliminacji Gaussa
    for k in range(n):
        # Elementy macierzy U w k-tym wierszu
        for j in range(k, n):
            suma = 0
            for s in range(k):
                suma += L[k, s] * U[s, j]
            U[k, j] = A[k, j] - suma

        # Elementy macierzy L w k-tej kolumnie
        for i in range(k+1, n):
            suma = 0
            for s in range(k):
                suma += L[i, s] * U[s, k]
            L[i, k] = (A[i, k] - suma) / U[k, k]

    return L, U


def solveJacobi(A, b, tol=1e-9, max_iterations=1000):
    """
    Solves the system of equations Ax = b using the Jacobi method

    Parameters:
    - A: coefficient matrix
    - b: right-hand side vector
    - x0: initial guess (default: zero vector with same shape as b)
    - tol: convergence tolerance
    - max_iterations: maximum number of iterations

    Returns:
    - x: solution vector with same shape as b
    """

    x = np.zeros_like(b)

    D = np.diag(np.diag(A))
    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)

    D_inv = np.linalg.inv(D)
    M = -np.dot(D_inv, L + U)
    w = np.dot(D_inv, b)

    r_norm = np.linalg.norm(np.dot(A, x) - b)
    jacobi_rnorm = np.array([r_norm]*max_iterations)
    iter_count = 0

    while iter_count < max_iterations:
        jacobi_rnorm[iter_count] = r_norm
        x_new = np.dot(M, x) + w
        r_norm = np.linalg.norm(np.dot(A, x_new) - b)
        x = x_new
        iter_count += 1
        if r_norm < tol:
            break

    return x, jacobi_rnorm[:iter_count]


def solveGauss_Seidel(A, b, tol=1e-9, max_iterations=1000):
    """
    Solves the system of equations Ax = b using the Gauss-Seidel method

    Parameters:
    - A: coefficient matrix
    - b: right-hand side vector
    - tol: convergence tolerance
    - max_iterations: maximum number of iterations

    Returns:
    - x: solution vector with same shape as b
    """

    x = np.zeros_like(b)

    L = np.tril(A)
    U = np.triu(A, k=1)

    L_inv = np.linalg.inv(L)

    r_norm = np.linalg.norm(np.dot(A, x) - b)
    iter_count = 0

    gauss_seidel_rnorm = np.array([r_norm] * max_iterations)

    for i in range(max_iterations):
        x_new = np.dot(L_inv, b - np.dot(U, x))
        r_norm = np.linalg.norm(np.dot(A, x_new) - b)
        gauss_seidel_rnorm[iter_count] = r_norm
        if r_norm < tol:
            break
        x = x_new
        iter_count += 1

    return x, gauss_seidel_rnorm[:iter_count]


def solveLU(A, b):
    """
    Solves the system of equations Ax = b using LU decomposition

    Parameters:
    - A: coefficient matrix
    - b: right-hand side vector

    Returns:
    - x: solution vector with same shape as b
    """

    L, U = LU(A)

    y = np.linalg.inv(L).dot(b)
    x = np.linalg.inv(U).dot(y)

    r_norm = np.linalg.norm(np.dot(A, x) - b)

    return x, r_norm
