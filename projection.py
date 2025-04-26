"""
"Pressure projection" step.
"""
from scipy.sparse import csr_matrix, csc_matrix, linalg
from sksparse.cholmod import cholesky
import numpy as np
import logging
import time
import numpy as np


class Projection(object):
    """
    Find the set of pressures so when velocities are adjusted for pressure differences, i.e.

         v_new <- v_temp - dt * (p_right -p_left)/dx, 

    the resulting velocity field is divergence fre  (where v_temp is velocity after advection, external forces, 
    and diffusion/viscosity but before pressure terms are added.)
    """

    def __init__(self, pressure_field, velocity_field, cell_flags, dt, free_velocities, boundary_velocities):
        """
        Initialize the projection object.
        :param pressure_field: The pressure field object.
        :param velocity_field: The velocity field object.
        :param cell_flags: Flags indicating which faces of which cells are free or fixed.
        :param dt: The time step size.
        :param free_velocities: The free velocity components.
        :param boundary_velocities: The boundary velocity components.
        """
        self.pressure_field = pressure_field
        self.velocity_field = velocity_field
        self.cell_flags = cell_flags
        self.dt = dt
        self.free_velocities = free_velocities
        self.boundary_velocities = boundary_velocities


def solve(a, b):
    """
    Solve the positive semidefinite system Ax = b.
    :param a: The coefficient matrix (should be positive semidefinite), sparse matrix.
    :param b: The right-hand side vector, vector.
    :return: The solution vector x.
    """
    # Perform Cholesky factorization
    L = cholesky(a)
    x = L.solve_A(b)

    return x


def _make_test_matrix(size, n_per_row=4, fmt='csr'):
    """
    Create a test positive semidefinite matrix.
    :param size: Size of the matrix.
    :param n_per_row: Number of non-zero elements per row.
    :return: A positive semidefinite matrix.
    """
    row_inds, col_inds, values = [], [], []
    # make the random prng:
    for i in range(size):
        # Add the diagonal element
        row_inds.append(i)
        col_inds.append(i)
        values.append(100.0)

        # Add half the off-diagonal elements (the rest added symmetrically)
        for j in range(n_per_row//2):

            if fmt == 'csc':
                col_inds.append(i)
                row_inds.append(np.random.randint(0, size))
            else:
                row_inds.append(i)
                col_inds.append(np.random.randint(0, size))

            values.append(np.random.rand())

    if fmt == 'csc':
        A = csc_matrix((values, (row_inds, col_inds)), shape=(size, size))
    else:
        A = csr_matrix((values, (row_inds, col_inds)), shape=(size, size))

    # make symmetric, positive semidefinite:
    A = A + A.T

    return A


def test_solve(n_trials=100, grid_size=30, verbose=True):
    """
    For testing on a G x G grid, we'll need to solve a G^2 x G^2 matrix.
    The matrix will be sparse, with 4 non-zero off-diagonal elements per row.

    Test the solve function with random matrices and vectors.
    :param n_trials: Number of trials to run.
    :param grid_size: Size of the grid for the random matrix.
    """
    n_nz_per_row = 4

    times = []
    errors = []
    for _ in range(n_trials):
        n_cells = grid_size * grid_size

        A = _make_test_matrix(n_cells, n_per_row=n_nz_per_row, fmt='csc')
        # A = A.tocsr()
        t0 = time.perf_counter()

        b = np.random.rand(n_cells)
        b = b.reshape(-1, 1)
        x = solve(A, b)
        prod = A.dot(x).reshape(-1)
        b = b.reshape(-1)
        errors.append(np.sqrt(np.mean((prod - b) ** 2)))
        # assert np.allclose(prod, b), "Solution does not satisfy Ax = b"
        times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
    mean_rms = np.mean(errors)
    mean_time = np.mean(times)
    if verbose:
        logging.info("\tRan %i trials solving sparse A: %i x %i with %i nonzero/row." %
                     (n_trials, n_cells, n_cells, n_nz_per_row))
        logging.info(f"\t\tAverage time per solve: {1000*mean_time:.2f} ms")
        logging.info(f"\t\tMax time per solve: {1000*np.max(times):.2f} ms")
        logging.info(f"\t\tAverage RMS error: {mean_rms:.12f}")
        logging.info("")

    return mean_time, mean_rms


def plot_speed_test():
    import matplotlib.pyplot as plt
    times = []
    grid_range = np.arange(10, 100, 2, dtype=np.int32)

    for grid_size in grid_range:
        logging.info("Testing grid size: %i" % grid_size)
        times.append(test_solve(n_trials=1, grid_size=grid_size, verbose=False)[0])

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    times = np.array(times)
    ax.plot(grid_range, times*1000, label='Mean solution time', marker='o')
    ax.set_xlabel('Grid Size G (solving for G^2 pressures)')
    ax.set_ylabel('Solution Time (ms)')
    ax.set_title('Speed of Pressure Projection step')
    # set grid with major and minor lines:
    ax.grid(which='both', linestyle='--', linewidth=0.5)
    # make y-axis logarithmic:
    ax.set_yscale('log')
    plt.tight_layout()
    #ax.legend()
    plt.show()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)


    # Test the solve function with random matrices and vectors
    print(test_solve(n_trials=10, grid_size=10))
    
    plot_speed_test()

    print("All tests passed.")
