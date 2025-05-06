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
    - jacobi_rnorm: array of residual norms at each iteration
    - iter_count: number of iterations performed
    """

    x = np.zeros_like(b)

    D = np.diag(np.diag(A))
    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)

    D_inv = np.linalg.inv(D)

    r_norm = np.linalg.norm(np.dot(A, x) - b)
    jacobi_rnorm = np.array([r_norm]*max_iterations)
    iter_count = 0

    while iter_count < max_iterations:
        jacobi_rnorm[iter_count] = r_norm
        x_new = D_inv.dot(b - np.dot(L + U, x))
        r_norm = np.linalg.norm(np.dot(A, x_new) - b)
        x = x_new
        iter_count += 1
        if r_norm < tol:
            break

    return x, jacobi_rnorm[:iter_count], iter_count


def gauss_seidel(A, b, tol=1e-9, max_iter=1000):
    n = len(b)
    x = np.zeros_like(b)

    gauss_seidel_norms = np.zeros(max_iter)
    for iteration in range(max_iter):
        x_new = np.copy(x)

        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        gauss_seidel_norms[iteration] = np.linalg.norm(x_new - x, ord=np.inf)
        if gauss_seidel_norms[iteration] < tol:
            return x_new, gauss_seidel_norms[:iteration + 1], iteration + 1

        x = x_new

    print('Osiągnięto maksymalną liczbę iteracji')
    return x, gauss_seidel_norms[:max_iter], max_iter


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
    - gauss_seidel_rnorm: array of residual norms at each iteration
    - iter_count: number of iterations performed
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

    return x, gauss_seidel_rnorm[:iter_count], iter_count


def solveLU(A, b):
    """
    Solves the system of equations Ax = b using LU decomposition

    Parameters:
    - A: coefficient matrix
    - b: right-hand side vector

    Returns:
    - x: solution vector with same shape as b
    - r_norm: residual norm
    """

    L, U = LU(A)

    y = np.linalg.inv(L).dot(b)
    x = np.linalg.inv(U).dot(y)

    r_norm = np.linalg.norm(np.dot(A, x) - b)

    return x, r_norm
