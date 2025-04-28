import numpy as np
import logging
from fields import CenterScalarField
import matplotlib.pyplot as plt
from projection import solve
from gradients import gradient_upwind
from scipy.sparse import csc_matrix
from loop_timing.loop_profiler import LoopPerfTimer as LPT


class PressureField(CenterScalarField):
    def __init__(self, size_m, grid_size, p_atm=1.0, unit='atm', phase='gas'):
        """
        Initialize the pressure field.
        :param size_m: The size of the simulation domain in meters (width, height).
        :param grid_size: The number of cells in the x and y direction for the pressure field.
        :param p0: Initial pressure value.
        :param unit: The unit of pressure. (atm, Pa, etc., for labels)
        :param phase: The phase of matter ('gas' or 'liquid').
        """
        super().__init__(size_m, grid_size, values=p_atm, name="Pressure")
        self.p_atm = p_atm
        self.unit = unit
        self.phase = phase  # 'gas' or 'liquid'
        if phase == 'liquid':
            raise NotImplementedError("Liquid phase not implemented yet.")

    def plot(self, ax, **kwargs):
        return super().plot(ax, title="Pressure Field", **kwargs)

    @LPT.time_function
    def set_incompressible(self, v_new, v_const, dt, phase='gas', reg=1e-6):
        """
        Get the projection of the pressure field onto the velocity field: find p(x,y) so when velocities are adjusted 
        for pressure differences i.e. v_new += - dt * (p_right -p_left)/dx, the resulting velocity field is divergence free.
        (where v_temp is velocity after advection, external forces, and diffusion/viscosity but before pressure terms are added.)

        If the phase of matter is 'gas', every cell is made divergence free.
        If the phase of matter is 'liquid', only entirely submerged cells are made divergence free.

        NOTE: This returns the negative A matrix, to be positive definite, solution should be negated.

        Roadmap:
            TODO 2: Implement static objects other than the 4 walls.
            TODO 3: Implement moving objects
            TODO 1: Implement liquids
            TODO 4: Implement sources and sinks?

            TODO N: move to a c extension for speed.


        :param v_new: The new velocity field after advection, external forces, and diffusion/viscosity, before pressure terms are added.
        :param v_const: the VelocityConstraint object, which components are free/frozen (and eventually, updatable) for the pressure projection.
        :param dt: The time step size.
        :param phase: The phase of matter ('gas' or 'liquid') (when liquids are implemented, pass in the fluid object and determine phase from its type.)
        """
        dx = v_new.dx

        A_row_inds = []
        A_col_inds = []
        A_vals = []  # A matrix, 1 row per cell, solving for pressures.
        A_size = self.n_cells[0] * self.n_cells[1]  # number of cells in the grid.

        B_vals = []
        
        import ipdb; ipdb.set_trace()
        # Fix B vector tomorrow...
        dudx, dudy = gradient_upwind(v_new.h_vel, dx)
        dvdy, dvdx = gradient_upwind(v_new.v_vel, dx)

        if phase == 'gas':
            # Solving for all pressures.
            # B vector is negative velocity divergence (i.e. the flux) at each cell.
            # A matrix is 4-point laplacian stencil with diagonal of -4/dx^2 and off-diagonal of 1/dx^2.
            for row in range(self.n_cells[0]):
                for col in range(self.n_cells[1]):
                    i = col
                    j = row  # flip y axis since matrix direction is opposite cartesian direction?

                    # Get the index of the cell in the 1D array.
                    cell_index = row * self.n_cells[1] + col

                    n_neighbors = 0
                    vel_diverg = 0.0
                    # TODO:  Add v_const checking in these if's:
                    if i > 0:  # left neighbor
                        A_row_inds.append(cell_index)
                        A_col_inds.append(cell_index - 1)
                        A_vals.append(-1.0 / dx**2)
                        n_neighbors += 1
                        vel_diverg += v_new.h_vel[i, j]
                    if i < self.n_cells[1] - 1:  # right neighbor
                        A_row_inds.append(cell_index)
                        A_col_inds.append(cell_index + 1)
                        A_vals.append(-1.0 / dx**2)
                        n_neighbors += 1
                        vel_diverg -= v_new.h_vel[i, j+1]
                    if j > 0:  # top neighbor
                        A_row_inds.append(cell_index)
                        A_col_inds.append(cell_index - self.n_cells[1])
                        A_vals.append(-1.0 / dx**2)
                        n_neighbors += 1
                        vel_diverg += v_new.v_vel[i, j]
                    if j < self.n_cells[0] - 1:  # bottom neighbor
                        A_row_inds.append(cell_index)
                        A_col_inds.append(cell_index + self.n_cells[1])
                        A_vals.append(-1.0 / dx**2)
                        n_neighbors += 1
                        vel_diverg -= v_new.v_vel[i+1, j]

                    # Diagonal term:
                    A_row_inds.append(cell_index)
                    A_col_inds.append(cell_index)
                    A_vals.append(n_neighbors / dx**2 + reg)
                    # B vector term:
                    B_vals.append(vel_diverg / dx/dt)

        else:
            raise NotImplementedError("Liquid phase not implemented yet.")
        # Create the sparse matrix A:
        A = csc_matrix((A_vals, (A_row_inds, A_col_inds)), shape=(A_size, A_size), dtype=np.float64)

        # Create the B vector:
        B = np.array(B_vals, dtype=np.float64)
        p = solve(A, B)

        # Check solution:
        B_hat = A.dot(p)
        # Check the residual:
        residual = np.sqrt(np.mean((B_hat - B)**2))  # L2 norm of the residual.
        logging.info(f"Pressure projection residual RMSE: {residual:.2e}")

        # Set the pressure field to the negative solution since we are solving for -p:
        self.values = -p.reshape(self.n_cells[1], self.n_cells[0])


def _test_pressure(plot=True):

    pf = PressureField((1.0, 1.0), (20, 20), unit='atm')
    pf.randomize(scale=10)
    if plot:
        fig, ax = plt.subplots()
        pf.plot(ax, alpha=0.5)
        plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _test_pressure(plot=True)
