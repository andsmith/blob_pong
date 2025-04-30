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
import cv2

from loop_timing.loop_profiler import LoopPerfTimer as LPT


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

        self._vel = VelocityField(size_m, vel_grid_size).add_wind(np.array([1.0, 0.0]), h_min=0.333)
        self._pressure = PressureField(size_m, vel_grid_size)
        self._fluid = SmokeField(size_m, fluid_grid_size)  # value at x,y is density of smoke.
        self._colorbar = None
        self._d_max = 0.0  # rendering this density, scale color to full saturation (clip if above)
        self._timing = {'frame_no':0,
                        't_0': time.perf_counter(), 
                        't_fps_start': time.perf_counter(),
                        'update_interval_sec': 2.0}
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

    @LPT.time_function
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
        self._vel.advect(dt)
        # self._vel.gravity(dt, self._fluid, rel_density=1.0)  # Gravity is a force acting on the fluid.
        # self._vel.diffuse(dt)

        # START HERE:
        self._pressure.set_incompressible(self._vel, None, dt)
        self._vel.project(self._pressure, dt)

        # b) Move the fluid along the velocity field:
        self._fluid.advect(self._vel, dt)

    def _update_fps(self):
        self._timing['frame_no'] += 1
        now = time.perf_counter()
        t_since_last_update = now - self._timing['t_fps_start']
        if t_since_last_update >= self._timing['update_interval_sec']:
            logging.info(f"FPS: {self._timing['frame_no'] / t_since_last_update:.2f}")
            self._timing['frame_no'] = 0
            self._timing['t_fps_start'] = now
            
    def animate_cv2(self, dt, render_v_grid, render_f_grid):
        # use render instead of plot methods.
        # use cv2 to render the simulation
        size = 900, 900  # w, h
        margin = 10
        win_name = "Fluid Simulation"
        cv2.namedWindow(win_name)
        cv2.resizeWindow(win_name, size[0], size[1])

        blank = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        bkg_color = 246, 238, 227  # Background
        fluid_color = 0, 4, 51
        line_color = 0, 0, 0

        bbox = {'x': (margin, size[0]-margin), 'y': (margin, size[1]-margin)}
        LPT.reset(False, burn_in=50, display_after=10)
        while True:
            # Do step:
            LPT.mark_loop_start()
            self.tick(dt)

            frame = blank.copy()
            self._fluid.render(frame, self._d_max, bbox, fluid_color, bkg_color)
            # self._vel.render(frame, bbox, line_color)
            if render_f_grid:
                self._fluid.render_grid(frame, bbox, line_color)
            if render_v_grid:
                self._vel.render_grid(frame, bbox,  line_color)

            self._update_fps()

            cv2.imshow(win_name, frame[:, :, ::-1])
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Randomize the fluid density to create a more realistic initial condition.
                self._vel.randomize(scale=3.0)
            elif key == ord('c'):
                self._fluid.add_circle((0.5, 0.5), 0.16, 10.0)
            elif key == ord('s'):
                cv2.imwrite("fluid.png", frame)
                logging.info("Saved image to fluid.png")

    def plot_state(self, ax, show_pressure=True):
        # Do step:
        # Plot step:
        # self._vel.plot_grid(ax)

        n_velocity_arrows = 20
        if show_pressure:
            #import ipdb; ipdb.set_trace()
            img_artist=self._pressure.plot(ax, alpha=0.6,res=200, cmap_name='hot')
            cbar_title = "Relative Pressure"
            print(np.mean(self._pressure.values), np.std(self._pressure.values))
        else:
            img_artist = self._fluid.plot(ax, alpha=0.6, res=200, cmap_name='hot')  # use 'gray' for showing velocity faces
            cbar_title = "Density"
        self._vel.plot_velocities(ax, show_faces=True, show_field=False, res=n_velocity_arrows)
        

        if self._colorbar is None:
            # Create colorbar only onnce
            self._colorbar = plt.colorbar(img_artist, ax=ax, shrink=0.8)
            self._colorbar.set_label(cbar_title)
        else:
            self._colorbar.update_normal(img_artist)

    def animate(self, dt, wait=True):

        plt.ion()
        fig, ax = plt.subplots()

        def disp():
            plt.cla()
            self.plot_state(ax)
            plt.xlim(0, self._size[0])
            plt.ylim(0, self._size[1])
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title("Fluid Simulation")
            plt.draw()

            if wait:
                key = plt.waitforbuttonpress()
                if key == ord('q'):
                    return True
            else:
                plt.pause(.25)
            return False

        t0 = time.perf_counter()

        disp()  # show initial conditions

        while True:

            self.tick(dt)

            if disp():
                break
            self._update_fps()

        logging.info("Exiting simulation loop.")
        plt.ioff()

    def set_d_max(self, d_max):
        # rendering this density, scale color to full saturation (clip if above)
        self._d_max = d_max

    def add_smoke(self, density_max=10.0):
        self._fluid.add_circle((0.5, 0.5), 0.26, density_max)  # Add a smoke source at the center of the domain.
        # self._fluid.randomize(scale=3.0)  # Randomize the fluid density to create a more realistic initial condition.
        self.set_d_max(density_max)


def run(plot=True, matplotlib=False):
    size_m = (1.0, 1.0)
    n_cells_x_vel = 50  # Number of velocity cells in the x direction
    fluid_cell_mult = 10  # Number of fluid cells per velocity cell.
    sim = Simulator(size_m, n_cells_x_vel, fluid_cell_mult)
    sim.add_smoke(1.0)  # Add a smoke source at the center of the domain.

    dt = 0.02  # Time step for the simulation.

    if plot:
        if matplotlib:
            sim.animate(dt)
        else:
            sim.animate_cv2(dt, render_v_grid=False, render_f_grid=False)
    else:
        n_ticks = 0
        t0 = time.perf_counter()
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
    run()
