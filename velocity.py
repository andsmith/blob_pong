from jax import numpy as jnp
from jax import random, grad, jit, vmap
import numpy as np
import matplotlib.pyplot as plt
import cv2
from interpolation import Interp2d
import logging


class VelocityField(object):
    """
    Staggered MAC grid for velocity field.
    The horizontal component of a cell's velocity is defined at the midpoints of its vertical boundaries.
    The vertical components are defined at the midpoints of horizontal boundaries.
    """

    def __init__(self, size_m, grid_size):
        self.n_cells = grid_size
        self.dx = size_m[0] / grid_size[0]  # dy is the same
        self.size = size_m

        # Check that the grid size is valid:
        dy= (size_m[1] / grid_size[1])
        if (np.abs(self.dx - dy) > 1e-10):
            raise ValueError("Grid size is not valid. dx and dy must be equal.")
        logging.info("Initializing Velocity grid %i x %i (dx = %.3f m) spanning (%.3f, %.3f) meters."
                     % (self.n_cells[0], self.n_cells[1], self.dx, self.size[0], self.size[1]))
        self._init_grids()

        self._interp_h, self._interp_v = None, None
        self.finalize()

    def interp_at(self, points):
        """
        Get the velocity at the given points.
        """
        if self._interp_h is None:
            self.finalize()
        h_vel = self._interp_h.interp(points)
        v_vel = self._interp_v.interp(points)
        vel = np.stack((h_vel, v_vel), axis=1)
        return vel
    
    def finalize(self):
        self.enforce_free_slip()
        h_p0 = (self.h_x[0], self.h_y[0])
        v_p0 = (self.v_x[0], self.v_y[0])
        n_h_vel_cells = (self.n_cells[0] + 1, self.n_cells[1] )
        n_v_vel_cells = (self.n_cells[0], self.n_cells[1]+ 1)
        self._interp_h = Interp2d(h_p0, self.dx, size=n_h_vel_cells, value=self.h_vel)
        self._interp_v = Interp2d(v_p0, self.dx, size=n_v_vel_cells, value=self.v_vel)
        
    
    def mark_dirty(self):
        """
        Mark the interpolators as dirty
        """
        self._interp_h = None
        self._interp_v = None

    def enforce_free_slip(self):
        """
        Set the horizontal component of veolocities on vertical boundaries to zero,
        and set the vertical component of velocities on horizontal boundaries to zero.
        """
        self.v_vel[0, :] = 0.0
        self.v_vel[-1, :] = 0.0
        self.h_vel[:, 0] = 0.0
        self.h_vel[:, -1] = 0.0
        # Set the velocities at the corners to zero:

    def _init_grids(self):
        # Coordinates of grid centers:
        self.centers_x = np.linspace(
            0.0, self.size[0], self.n_cells[0] + 1)[1:-1] + 0.5 * (self.size[0] / self.size[0])
        self.centers_y = np.linspace(
            0.0, self.size[1], self.n_cells[1] + 1)[1:-1] + 0.5 * (self.size[1] / self.size[1])

        # Coordinates of grid face centers for the x-component of velocity:
        self.h_x = np.linspace(0.0, self.size[0], self.n_cells[0] + 1)
        self.h_y = np.linspace(0.0, self.size[1], self.n_cells[1] + 1)[:-1] + 0.5 * (self.size[1] / self.n_cells[1])


        # Coordinates of grid face centers for the y-component of velocity:
        self.v_x = np.linspace(0.0, self.size[0], self.n_cells[0] + 1)[:-1] + 0.5 * (self.size[0] / self.n_cells[0])
        self.v_y = np.linspace(0.0, self.size[1], self.n_cells[1] + 1)

        # Horizontal and vertical velocities (stored in numpy order):
        self.v_vel = np.zeros((self.n_cells[1] + 1, self.n_cells[0]), dtype=np.float32)
        self.h_vel = np.zeros((self.n_cells[1], self.n_cells[0] + 1), dtype=np.float32)
        if True:
            self.v_vel += np.random.randn(self.n_cells[1] + 1, self.n_cells[0]) * .1
            self.h_vel += np.random.randn(self.n_cells[1], self.n_cells[0] + 1) * .1


    def plot_grid(self, ax):
        """
        Plot the grid
        :param ax: matplotlib axis to plot on.
        """
        for i in range(self.h_x.shape[0]):
            ax.plot([self.h_x[i], self.h_x[i]], [0, self.size[1]], 'k-', lw=1)
        for i in range(self.v_y.shape[0]):
            ax.plot([0, self.size[0]], [self.v_y[i], self.v_y[i]], 'k-', lw=1)
        # plot the bounds
        ax.plot([0, self.size[0]], [0, 0], 'k-', lw=2)
        ax.plot([0, self.size[0]], [self.size[1], self.size[1]], 'k-', lw=2)  # Top edge
        ax.plot([0, 0], [0, self.size[1]], 'k-', lw=2)  # Left edge
        ax.plot([self.size[0], self.size[0]], [0, self.size[1]], 'k-', lw=2)  # Right edge

    def plot_velocities(self, ax, res=100, show_faces=True, show_field=False):

        def plot_component(x_coords, y_coords, vel,  plt_str, label, direction=(0, 1)):
            """
            Show as arrows on each face
            """
            vel_h, vel_v = vel * direction[0], vel * direction[1]
            print(label,x_coords.shape, y_coords.shape, vel.shape)
            ax.quiver(x_coords, y_coords, vel_h, vel_v, label=label, color=plt_str,
                      scale_units='xy', angles='xy', width=0.005, headwidth=3, headlength=5)
            ax.set_aspect('equal')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')

            # ax.set_xlim(0, self.size[0])
            # ax.set_ylim(0, self.size[1])

        def plot_field():
            aspect = self.size[0] / self.size[1]
            x_val = np.linspace(0.0,self.size[0], res)

            x_res, y_res = int(res), int(res/aspect)
            y_val = np.linspace(0.0,self.size[1], int(res/aspect))
            x_coords, y_coords = np.meshgrid(x_val, y_val)
            x_coords = x_coords.flatten()
            y_coords = y_coords.flatten()
            vel = self.interp_at(np.array([x_coords, y_coords]).T)

            # plot as a vector field
            vel_h, vel_v = vel[:, 0], vel[:, 1]
            vel_h = vel_h.reshape((y_res, x_res))
            vel_v = vel_v.reshape((y_res, x_res))
            x_coords = x_coords.reshape((y_res, x_res))
            y_coords = y_coords.reshape((y_res, x_res))
            ax.quiver(x_coords, y_coords, vel_h, vel_v, color='k', scale_units='xy', angles='xy',
                      width=0.005, headwidth=3, headlength=4)
            ax.set_aspect('equal')

        if show_faces:
            plot_component(self.h_x, self.h_y, self.h_vel, 'r', 'Horizontal velocity', direction=(1, 0))
            plot_component(self.v_x, self.v_y, self.v_vel, 'b', 'Vertical velocity', direction=(0, 1))
        if show_field: 
            plot_field()
        ax.legend()
        ax.set_title('Velocity field')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    vf = VelocityField((1.0, 1.0), (10, 10))
    fig, ax = plt.subplots()
    vf.plot_grid(ax)
    vf.plot_velocities(ax, res=50, show_faces=False,show_field=True)
    plt.show()
