#from jax import numpy as jnp
#from jax import random, grad, jit, vmap
import numpy as np
import matplotlib.pyplot as plt
import cv2
from velocity import VelocityField
from pressure import PressureField
from fluid import SmokeField #, LiquidField
from util import scale_y
import logging

class Simulator(object):
    """
    Simulate 2d fluid flow in a rectangular domain using Navier Stokes equations.

    Subclasses are for compressible/incompressible fluids.

    Roadmap:
    1. "sealed tank" model:  fluid interacting with walls, itself, and gravity.
    TODO: 2. Create sources and sinks.
    TODO: 3. Be pushed around by moving objects.


    For smoke, the fluid is represented by a density function.
    For liquid, the fluid is represented by a signed distance function (level set).
    """

    def __init__(self, size_m, n_cells_x_vel, fluid_cell_mult):
        """

        :param size_m: The size of the simulation domain in meters (width, height).
        :param n_cells_x_vel: The number of cells in the x direction for the velocity field. (y is in proportion)
        :param fluid_cell_mult:  How many fluid cells per velocity cell. (must be integer >=1)
        """

        self._size, vel_grid_size, _ = scale_y(size_m, n_cells_x_vel)
        _, fluid_grid_size, _ = scale_y(size_m, n_cells_x_vel * fluid_cell_mult)

        self._vel = VelocityField(size_m, vel_grid_size)
        self._pressure = PressureField(size_m, vel_grid_size)
        self._fluid = SmokeField(size_m, fluid_grid_size)  # value at x,y is density of smoke.

    #@jit
    def pixel_to_world(self, coords):
        """
        Convert pixel coordinates to world coordinates.
        :param coords: N x 2 array of pixel coordinates.
        :return: N x 2 array of world coordinates.
        """
        x, y = coords[:, 0], coords[:, 1]
        w_coords = (x / self._size[0]) * self._dims[0], (y / self._size[1]) * self._dims[1]
        return np.array(w_coords).T

    #@jit
    def world_to_pixel(self, coords):
        """
        Convert world coordinates to pixel coordinates.
        :param coords: N x 2 array of world coordinates.
        :return: N x 2 array of pixel coordinates.
        """
        x, y = coords[:, 0], coords[:, 1]
        p_coords = (x / self._dims[0]) * self._size[0], (y / self._dims[1]) * self._size[1]
        return np.array(p_coords).T


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    size_m = (1.0, 1.0)
    n_cells_x_vel = 10
    fluid_cell_mult = 3
    sim = Simulator(size_m, n_cells_x_vel, fluid_cell_mult)
    sim._pressure.plot(plt.gca())
    sim._vel.plot_grid(plt.gca())
    sim._vel.plot_velocities(plt.gca())
    plt.show()