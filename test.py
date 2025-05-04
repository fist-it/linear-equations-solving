import numpy as np
from solve import solveJacobi, solveGauss_Seidel, solveLU, LU
from data import getMatrixTaskA, getVectorTaskA


def test_lu():
    A = getMatrixTaskA(100)
    L, U = LU(A)
    # Check if the solution is close to the expected solution
    assert np.allclose(A, L @ U), \
        "LU decomposition did not reconstruct the original matrix"
    print("LU method passed the test")


def test_jacobi():
    A = getMatrixTaskA(100)
    b = getVectorTaskA(100)
    x_jacobi, residuum = solveJacobi(A, b)
    # Check if the solution is close to the expected solution
    x_expected = np.linalg.solve(A, b)
    assert np.allclose(x_jacobi, x_expected), \
        "Jacobi method did not converge to the expected solution"
    assert len(residuum) > 0, \
        "Residuum list is empty"

    print("Jacobi method passed the test")


def test_gauss_seidel():
    A = getMatrixTaskA(100)
    b = getVectorTaskA(100)
    x_gauss_seidel, residuum = solveGauss_Seidel(A, b)
    # Check if the solution is close to the expected solution
    x_expected = np.linalg.solve(A, b)
    assert np.allclose(x_gauss_seidel, x_expected), \
        "Gauss-Seidel method did not converge to the expected solution"
    assert len(residuum) > 0, \
        "Residuum list is empty"

    print("Gauss-Seidel method passed the test")


def test_direct_lu():
    A = getMatrixTaskA(100)
    b = getVectorTaskA(100)
    x_lu = solveLU(A, b)
    # Check if the solution is close to the expected solution
    x_expected = np.linalg.solve(A, b)
    assert np.allclose(x_lu, x_expected), \
        "LU method did not converge to the expected solution"

    print("Direct method passed the test")


def main():
    test_jacobi()
    test_gauss_seidel()
    test_lu()
    test_direct_lu()
    print("All tests passed!")


if __name__ == "__main__":
    main()
