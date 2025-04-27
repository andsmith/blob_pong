# from jax import numpy as jnp
# from jax import random, grad, jit, vmap
import numpy as np
import matplotlib.pyplot as plt
from velocity import VelocityField, VelocityConstraints
from pressure import PressureField
from fluid import SmokeField  # , LiquidField
from util import scale_y
import logging
import time

class Simulator(object):
    """
    Simulate 2d fluid flow in a rectangular domain using Navier Stokes equations.

    Roadmap:
    1. "sealed tank" model:  fluid interacting with walls, itself, and gravity.
    TODO: 2. Create sources and sinks.
    TODO: 3. Be pushed around by moving objects.

    (Current) For smoke, the fluid is represented by a density function.
    (Future) For liquid, the fluid is represented by a signed distance function (level set).
    """

    def __init__(self, size_m, n_cells_x_vel, fluid_cell_mult):
        """

        :param size_m: The size of the simulation domain in meters (width, height).
        :param n_cells_x_vel: The number of cells in the x direction for the velocity field. (y is in proportion)
        :param fluid_cell_mult:  How many fluid cells per velocity cell. (must be integer >=1)
        """

        self._size, vel_grid_size, _ = scale_y(size_m, n_cells_x_vel)
        _, fluid_grid_size, _ = scale_y(size_m, n_cells_x_vel * fluid_cell_mult)

        self._vel = VelocityField(size_m, vel_grid_size).add_wind(np.array([3.0, -1.0])) 
        self._pressure = PressureField(size_m, vel_grid_size)
        self._fluid = SmokeField(size_m, fluid_grid_size)  # value at x,y is density of smoke.
        self._colorbar = None

        self._timing = {}


    # @jit
    def pixel_to_world(self, coords):
        """
        Convert pixel coordinates to world coordinates.
        :param coords: N x 2 array of pixel coordinates.
        :return: N x 2 array of world coordinates.
        """
        x, y = coords[:, 0], coords[:, 1]
        w_coords = (x / self._size[0]) * self._dims[0], (y / self._size[1]) * self._dims[1]
        return np.array(w_coords).T

    # @jit
    def world_to_pixel(self, coords):
        """
        Convert world coordinates to pixel coordinates.
        :param coords: N x 2 array of world coordinates.
        :return: N x 2 array of pixel coordinates.  
        """
        x, y = coords[:, 0], coords[:, 1]
        p_coords = (x / self._dims[0]) * self._size[0], (y / self._dims[1]) * self._size[1]
        return np.array(p_coords).T
    
    def tick(self, dt):
        """
        Perform a simulation step:
          a) Update velocity & pressure field by dt using the Navier Stokes equations.
            1. External forces: gravity, buoyancy, etc.
            2. Momentum/advection term:  advect the velocity field using the current velocity field.
            3. TODO: Viscosity: Evaluate the stress tensor, compute viscous forces, add to velocity field.
            4. Pressure projection:  find the pressure field so the velocity field is divergence free.
          b) Advect the fluid for time dt using the new velocity field.
        """
        # a) Update velocity
        # TODO: Gravity here
        self._vel.advect(dt)
        #self._vel.diffuse(dt)

        # START HERE: 
        vel_constraints = VelocityConstraints(self._vel)
        self._pressure.set_incompressible(self._vel,vel_constraints, dt)
        self._vel.project(self._pressure, dt)

        # b) Move the fluid along the velocity field:
        self._fluid.advect(self._vel, dt)
    

    def plot_step(self, ax, dt):
        # Do step:
        self.tick(dt)

        # Plot step:
        #self._vel.plot_grid(ax)
        n_velocity_arrows = 50
        self._vel.plot_velocities(ax, show_faces=False, show_field=True, res=n_velocity_arrows)
        # img_artist=self._pressure.plot(ax, alpha=0.6,res=500)
        img_artist = self._fluid.plot(ax, alpha=0.6, res=200)

        if self._colorbar is None:
            # Create colorbar only once
            self._colorbar = plt.colorbar(img_artist, ax=ax, shrink=0.8)
            self._colorbar.set_label("Density")
        else:
            self._colorbar.update_normal(img_artist)

    def animate(self, dt):
        plt.ion()
        fig, ax = plt.subplots()
        n_frames=0
        t0 = time.perf_counter()
        while True:
            n_frames += 1

            self.plot_step(ax, dt)
            plt.pause(.1)
            plt.cla()
            plt.xlim(0, self._size[0])
            plt.ylim(0, self._size[1])
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title("Fluid Simulation")
            plt.draw()

            if n_frames % 10 == 0:
                t1 = time.perf_counter()
                logging.info(f"FPS: {n_frames / (t1 - t0):.2f}")
                n_frames = 0
                t0 = time.perf_counter()

        plt.ioff()

    def add_smoke(self):
        self._fluid.add_circle((0.5, 0.5), 0.16, 10.0)  # Add a smoke source at the center of the domain.
        # self._fluid.randomize(scale=3.0)  # Randomize the fluid density to create a more realistic initial condition.


def run(plot=True):
    size_m = (2.0, 2.0)
    n_cells_x_vel = 15  # Number of velocity cells in the x direction
    fluid_cell_mult = 10  # Number of fluid cells per velocity cell.
    sim = Simulator(size_m, n_cells_x_vel, fluid_cell_mult)
    sim.add_smoke()

    dt = 0.05  # Time step for the simulation.
    
    if plot:
       sim.animate(dt)
    else:
        n_ticks = 0
        t0= time.perf_counter()
        while True:
            sim.tick(dt)
            n_ticks += 1
            if n_ticks % 10 == 0:
                t1 = time.perf_counter()
                logging.info(f"Updates per second: {n_ticks / (t1 - t0):.2f}")
                n_ticks = 0
                t0 = time.perf_counter()
        
    # sim._pressure.plot(plt.gca())
    # sim._vel.plot_grid(plt.gca())
    # sim._vel.plot_velocities(plt.gca())
    # plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run(plot=True)