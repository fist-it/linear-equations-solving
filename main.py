import numpy as np
import matplotlib.pyplot as plt
from solve import solveJacobi, solveLU, gauss_seidel
from data import getMatrixSize, getMatrixTaskA, getVectorTaskA, getMatrixTaskC
import time


def main():
    size = getMatrixSize()

    # task A and B {{{
    A = getMatrixTaskA(size)
    b = getVectorTaskA(size)

    x = np.linalg.solve(A, b)

    timer = time.time()
    x_jacobi, jacobi_residuum, jacobi_iterations = solveJacobi(A, b)
    timer = time.time() - timer
    jacobi_time = timer

    timer = time.time()
    x_gauss_seidel, gauss_residuum, gauss_iterations = gauss_seidel(A, b)
    timer = time.time() - timer
    gauss_seidel_time = timer

    assert x.shape == x_jacobi.shape, \
        "Shapes of x and x_jacobi do not match"
    assert x.shape == x_gauss_seidel.shape, \
        "Shapes of x and x_gauss_seidel do not match"

    # Check if the solutions are close
    assert np.allclose(
        x, x_jacobi), "Solution jacobi does not match"
    assert np.allclose(
        x, x_gauss_seidel), "Solution gauss seidel does not match"

    print(f"Jacobi Residuum length: {len(jacobi_residuum)}")
    print(f"Gauss-Seidel Residuum length: {len(gauss_residuum)}")

    print(f"Jacobi took {jacobi_time:.2f} seconds")
    print(f"Gauss-Seidel took {gauss_seidel_time:.2f} seconds")

    print(f"Jacobi iterations: {jacobi_iterations}")
    print(f"Gauss-Seidel iterations: {gauss_iterations}")

    # Plotting the residuals
    plt.plot(jacobi_residuum, label="Jacobi Residuum",
             marker='o', linewidth=0.8)
    plt.plot(gauss_residuum, label="Gauss-Seidel Residuum",
             marker='s', linewidth=0.8)

    plt.annotate(f"{jacobi_residuum[-1]:.2e}",
                 (len(jacobi_residuum)-1, jacobi_residuum[-1]),
                 textcoords="offset points", xytext=(-10, 10), ha='center')
    plt.annotate(f"{gauss_residuum[-1]:.2e}",
                 (len(gauss_residuum)-1, gauss_residuum[-1]),
                 textcoords="offset points", xytext=(-10, 10), ha='center')

    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel("Iteration")
    plt.ylabel("Residual Norm")
    plt.title("Residual Norms of Jacobi and Gauss-Seidel Methods")
    plt.legend()
    plt.savefig("./sprawozdanie/graphs/residuals_task_a.png", dpi=300)
    plt.show()
    # }}}

    # task C {{{
    A = getMatrixTaskC(size)

    timer = time.time()
    x_jacobi, jacobi_residuum, jacobi_iterations = solveJacobi(A, b)
    timer = time.time() - timer
    jacobi_time = timer
    print(f"Jacobi took {jacobi_time:.2f} seconds")

    # Solving with gauss method for non-diagonal dominant matrix takes
    # around 40 seconds, disable if not needed
    timer = time.time()
    x_gauss_seidel, gauss_residuum, iterations = gauss_seidel(A, b)
    timer = time.time() - timer
    gauss_seidel_time = timer
    print(f"Gauss-Seidel took {gauss_seidel_time:.2f} seconds")

    # Plotting the residuals
    plt.plot(jacobi_residuum, label="Jacobi Residuum",
             linewidth=0.8)
    plt.plot(gauss_residuum, label="Gauss-Seidel Residuum",
             linewidth=0.8)

    plt.annotate(f"{jacobi_residuum[-1]:.2e}",
                 (len(jacobi_residuum)-1, jacobi_residuum[-1]),
                 textcoords="offset points", xytext=(-10, 10), ha='center')
    plt.annotate(f"{gauss_residuum[-1]:.2e}",
                 (len(gauss_residuum)-1, gauss_residuum[-1]),
                 textcoords="offset points", xytext=(-10, 10), ha='center')

    plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xlabel("Iteration")
    plt.ylabel("Residual Norm")
    plt.title("Residual Norms of Jacobi and Gauss-Seidel Methods (Task C)")
    plt.legend()
    plt.savefig("./sprawozdanie/graphs/residuals_task_c.png", dpi=300)
    plt.show()
    # }}}


if __name__ == "__main__":
    main()
