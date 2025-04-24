import numpy as np
# import jax.numpy as jnp
from jax import grad, jit, vmap
import logging
from fields import CenterScalarField
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import splu, spsolve
from projection import Projection


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

    def set_incompressible(self, temp_velocities, dt, velocity_masks):
        """
        Get the projection of the pressure field onto the velocity field: find p(x,y) so when velocities are adjusted 
        for pressure differences i.e. v_new <- v_temp - dt * (p_right -p_left)/dx, the resulting velocity field is divergence free.
        (where v_temp is velocity after advection, external forces, and diffusion/viscosity but before pressure terms are added.)

        If the phase of matter is 'gas', every cell is made divergence free.
        If the phase of matter is 'liquid', only entirely submerged cells are made divergence free (unimplemented).

        Roadmap:
            TODO 1: Implement liquids
            TODO 2: Implement static objects other than the 4 walls.
            TODO 3: Implement moving objects
            TODO 4: Implement sources and sinks?

        :param temp_velocities: Mostly-updated velocity field, just needs to be incompressible.
        :param dt: The time step size.
        :param velocity_masks:  (horizontal, vertical), each an self.n_cells sized array
          indicating which velocity components are free (i.e. not fixed by boundary conditions).
        """
        if self.dx != velocities.dx or self.size != velocities.size:
            raise ValueError("Velocity and pressure fields must have the same grid sizes & spacing.")

        # Determine which velocity components (defined on each cell's four edges) are free/frozen for boundary conditions.    
        cell_flags, boundary_velocities, free_velocities = self._calc_dof(velocities, dt, fluid=fluid)

        pressure_proj = Projection(self, velocities, cell_flags, dt, free_velocities, boundary_velocities)
        self.values = pressure_proj.get_pressures()


def _test_pressure(plot=True):

    pf = PressureField((1.0, 1.0), (20, 20), unit='atm')
    pf.randomize(seed=0, scale=10)
    if plot:
        fig, ax = plt.subplots()
        pf.plot(ax, alpha=0.5)
        plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _test_pressure(plot=True)
