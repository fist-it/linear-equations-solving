import matplotlib.pyplot as plt
from solve import solveJacobi, solveGauss_Seidel, solveLU, gauss_seidel
from data import getMatrixTaskA, getVectorTaskA
import time


def main():
    sizes = range(100, 1001, 100)

    times_jacobi = []
    times_gauss_seidel = []
    times_lu = []

    for size in sizes:
        A = getMatrixTaskA(size)
        b = getVectorTaskA(size)

        # Measure time for Jacobi method
        start_time = time.time()
        solveJacobi(A, b, max_iterations=1000000)
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        times_jacobi.append(1000 * (end_time - start_time))
        print(f"Jacobi method for size {size} took {
            total_time:.4f} milliseconds")

        start_time = time.time()
        gauss_seidel(A, b, max_iter=1000000)
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        times_gauss_seidel.append(1000 * (end_time - start_time))
        print(f"Gauss-Seidel method for size {size} took {
            total_time:.4f} milliseconds")

        start_time = time.time()
        solveLU(A, b)
        end_time = time.time()
        total_time = (end_time - start_time) * 1000
        times_lu.append(1000 * (end_time - start_time))
        print(f"LU method for size {size} took {total_time:.4f} milliseconds")
        print("--------------------------------------------------")

    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, times_jacobi, label='Jacobi Method', marker='o')
    plt.plot(sizes, times_gauss_seidel,
             label='Gauss-Seidel Method', marker='o')
    plt.plot(sizes, times_lu, label='LU Method (Direct)', marker='o')

    plt.subplots_adjust(bottom=0.2)
    plt.title('Time Complexity of Jacobi and Gauss-Seidel and Direct Methods')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time [ms]')
    plt.legend()
    plt.grid()
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.savefig("./sprawozdanie/graphs/time_complexity.png", dpi=300)
    plt.yscale('linear')
    plt.savefig("./sprawozdanie/graphs/time_complexity_linear.png", dpi=300)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
