import numpy as np
import matplotlib.pyplot as plt
# from interpolation import Interp2d
from interp_jax import Interp2d
import logging
from semi_lagrange import advect
from fields import InterpField
from projection import solve
from fluid import SmokeField
from loop_timing.loop_profiler import LoopPerfTimer as LPT


class VelocityConstraints(object):
    """
    Which components are constrained to specific values, which are free to move.
        Constraints in this class:
          - Boundary cells:  Velocity components through faces of the domain are fixed to zero (inmpermeable).
          - interior cells:  Velocity components are free to take any value.
          - TODO:  Sources and sinks on the boundary.
          - TODO:  Static/moving objects:  Normal component (to object) is fixed to object's velocity.
    """

    def __init__(self, velocity_field, fix_boundary=True):
        """
        :param velocity_field: The velocity field to constrain.
        """
        self.grid_size = velocity_field.n_cells

        self.h_vel_const = np.zeros(velocity_field.h_vel.shape)
        self.v_vel_const = np.zeros(velocity_field.v_vel.shape)

        self.h_fixed = np.zeros(velocity_field.h_vel.shape, dtype=bool)
        self.v_fixed = np.zeros(velocity_field.v_vel.shape, dtype=bool)

        if fix_boundary:
            self.fix_boundary()

    def fix_boundary(self):
        """
        Fix the boundary cells to zero velocity.
        :return: None
        """
        self.h_fixed[0, :] = True
        self.h_fixed[-1, :] = True
        self.v_fixed[:, 0] = True
        self.v_fixed[:, -1] = True


class VelocityField(InterpField):
    """
    Staggered MAC grid for velocity field.
    The horizontal component of a cell's velocity is defined at the midpoints of its vertical boundaries.
    The vertical components are defined at the midpoints of horizontal boundaries.
    """

    def __init__(self, size_m, grid_size):
        super().__init__(size_m, grid_size, name="Velocity")

        logging.info("Initializing Velocity grid %i x %i (dx = %.3f m) spanning (%.3f, %.3f) meters."
                     % (self.n_cells[0], self.n_cells[1], self.dx, self.size[0], self.size[1]))
        self._init_grids()
        self.finalize()  # prepare for interpolation (call this after changes to velocities)

    def add(self, delta_v):
        """
        Add a delta velocity field to the current velocity field.
        :param delta_v: The delta velocity field to add.
        """
        if self.h_vel.shape != delta_v.h_vel.shape or self.v_vel.shape != delta_v.v_vel.shape:
            raise ValueError("DeltaV objects must have the same shape.")
        self.h_vel += delta_v.h_vel
        self.v_vel += delta_v.v_vel
        self.mark_dirty()

    def _interp_at(self, points):
        h_vel = self._interp['horiz'].interpolate(points)
        v_vel = self._interp['vert'].interpolate(points)
        vel = np.stack((h_vel, v_vel), axis=-1)
        return vel

    def finalize(self):
        self.enforce_free_slip()
        super().finalize()

    def _get_interp(self):
        """
        Create the interpolation object for the velocity field.
        (finalize() calls this)
        """
        h_p0 = (self.h_x[0], self.h_y[0])
        v_p0 = (self.v_x[0], self.v_y[0])
        n_h_vel_cells = (self.n_cells[0] + 1, self.n_cells[1])
        n_v_vel_cells = (self.n_cells[0], self.n_cells[1] + 1)
        return {'horiz': Interp2d(h_p0, self.dx, size=n_h_vel_cells, value=self.h_vel),
                'vert': Interp2d(v_p0, self.dx, size=n_v_vel_cells, value=self.v_vel)}

    def enforce_free_slip(self):
        """
        Set the horizontal component of veolocities on vertical boundaries to zero,
        and set the vertical component of velocities on horizontal boundaries to zero.
        """
        self.v_vel[0, :] = 0.0
        self.v_vel[-1, :] = 0.0
        self.h_vel[:, 0] = 0.0
        self.h_vel[:, -1] = 0.0

        # TODO:  enforce free/no slip for objects

    def gravity(self, dt, fluid, rel_density=100.0):
        """
        Add gravity to the velocity field in proportion to local interpolated fluid density.
        :param fluid: The fluid field to add gravity to.
        :return: None
        """
        if not isinstance(fluid, SmokeField):
            raise ValueError("Gravity only works with smoke fields for now.")

        fluid_density = fluid.interp_at(self._v_points[1:-1, :, :]) * rel_density
        force = fluid_density * 9.81*dt
        self.v_vel[1:-1, :] -= force  # apply gravity to vertical velocity

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

        # All grid points (for advecting velocities):
        self._h_points = np.stack(np.meshgrid(self.h_x, self.h_y), axis=-1)
        self._v_points = np.stack(np.meshgrid(self.v_x, self.v_y), axis=-1)

        # Horizontal and vertical velocities (stored in numpy order):
        self.v_vel = np.zeros((self.n_cells[1] + 1, self.n_cells[0]))
        self.h_vel = np.zeros((self.n_cells[1], self.n_cells[0] + 1))

    def randomize(self, scale=1.0):

        self.v_vel += self._rng.normal(0, scale, self.v_vel.shape)
        self.h_vel += self._rng.normal(0, scale, self.h_vel.shape)
        self.finalize()
        return self

    def add_wind(self, wind, h_min=0.5):
        """
        Add a wind velocity to the velocity field.
        :param wind: The wind velocity to add.
        :param h_max:  only add to velocity field above this position.
        """
        if wind.shape != (2,):
            raise ValueError("Wind must be a 2D vector.")

        horiz_y_mask = self.h_y > h_min
        vert_y_mask = self.v_y > h_min

        self.h_vel[horiz_y_mask, :] += wind[0]
        self.v_vel[vert_y_mask, :] += wind[1]
        self.finalize()
        return self

    def get_cfl(self, dt, dx, C=0.5):
        """
        We need to step dt forward, but that might be too big.

        Break it into N steps of length dt_sub, where:

           dt_sub <= C * dx / max(|u| + |v|)

        and N * dt_sub = dt.

        :param dt: The time step to use for advection.
        :param dx: The grid spacing.
        :param C: The CFL number.  Default is 0.5.
        :return: The CFL condition.
        """
        max_vel = np.max(np.abs(self.v_vel)) + np.max(np.abs(self.h_vel))
        if max_vel == 0.0:
            return dt, 1
        dt_sub = C * dx / max_vel
        n_iter = int(np.ceil(dt / dt_sub))
        dt_sub = dt / n_iter
        return dt_sub, n_iter
    @LPT.time_function
    def advect(self, dt, fluid=None):
        """
        Approximate the momentum term of the Navier Stokes equation using semi-lagrangian advection.
        For each grid point, find the new velocity by moving the point backwards through the
        velocity field for a time step dt. Then interpolate the velocity at the new position.

        :param dt: The time step to use for advection.
        :param fluid: The fluid field to advect.  (Not used for gasses, TODO.)
        """
        def _get_v_at_pos(points):
            points_back = advect(points, self, dt, self.dx, self.size)
            return self.interp_at(points_back)

        h_vel = _get_v_at_pos(self._h_points)
        v_vel = _get_v_at_pos(self._v_points)
        self.h_vel = h_vel[:, :, 0]
        self.v_vel = v_vel[:, :, 1]
        self.finalize()

    def gradient(self):
        #  (row order is in DECREASING y direction)
        dx = (self.h_vel[:, 1:] - self.h_vel[:, :-1])/self.dx  # horizontal velocity divergence in x direction.
        dy = (self.v_vel[1:, :] - self.v_vel[:-1, :])/self.dx  # vertical velocity divergence in y direction
        return dx, dy

    @LPT.time_function
    def project(self, pressure, dt, v_const=None):
        """
        Project the velocity field onto the pressure field to make it divergence free.
        :param pressure: The pressure field to project onto.
        :param dt: The time step to use for projection.
        :param v_const: The velocity constraints to use for projection.
        """
        dpdx, dpdy = pressure.gradient()

        self.h_vel[:, 1:-1] -= dpdx * dt 
        self.v_vel[1:-1, :] -= dpdy * dt 

        # check for divergence:
        # div = self.gradient()
        # div = np.sum(div, axis=0)  # sum over x and y components
        # div = np.abs(div)  # take the absolute value of the divergence
        # div = np.max(div)
        # logging.info("Max cell divergence after projection: %f" % div)
        self.finalize()

    def diffuse(self, dt, fluid=None):
        """
        Diffuse the velocity field using a simple diffusion equation.
        :param dt: The time step to use for diffusion.
        :param fluid: The fluid field to diffuse.  (Not used for gasses, TODO.)
        """
        pass

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

        def plot_component(x_coords, y_coords, vel,  plt_str, label, direction=(0, 1), min_v=0.025):
            """
            Show as arrows on each face
            """
            vel_h, vel_v = vel * direction[0], vel * direction[1]

            vel_h[np.abs(vel_h) < min_v] = 0
            vel_v[np.abs(vel_v) < min_v] = 0

            ax.quiver(x_coords, y_coords, vel_h, vel_v, label=label, color=plt_str,
                      scale_units='xy', angles='xy', width=0.005, headwidth=3, headlength=5)

            ax.set_aspect('equal')
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')

            # ax.set_xlim(0, self.size[0])
            # ax.set_ylim(0, self.size[1])

        def plot_field(min_v=0.025):
            aspect = self.size[0] / self.size[1]
            x_val = np.linspace(0.0, self.size[0], res)

            x_res, y_res = int(res), int(res/aspect)
            y_val = np.linspace(0.0, self.size[1], int(res/aspect))
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

            vel_h[np.abs(vel_h) < min_v] = 0
            vel_v[np.abs(vel_v) < min_v] = 0

            ax.quiver(x_coords, y_coords, vel_h, vel_v, color='k', scale_units='xy', angles='xy',scale=res/2,
                      width=0.0025, headwidth=2.5, headlength=3.5)
            ax.set_aspect('equal')

        if show_faces:
            self.plot_grid(ax)
            plot_component(self.h_x, self.h_y, self.h_vel, 'r', 'Horizontal velocity', direction=(1, 0))
            plot_component(self.v_x, self.v_y, self.v_vel, 'b', 'Vertical velocity', direction=(0, 1))
            ax.legend()
        if show_field:
            plot_field()
        ax.set_title('Velocity field, free-slip BCs\n(boundary normal v = 0)')


def test_velocity():
    # Create a random 10x10 velocity field, verify boundary condition.
    size_m = (1.0, 1.0)
    grid_size = (10, 10)
    dx = .1
    vel = VelocityField(size_m, grid_size)
    vel.randomize(scale=1.0)
    # vel.enforce_free_slip()
    t = np.linspace(0, 1.0, 100)
    zero_v = np.zeros(t.shape)
    min_coord = np.zeros(t.shape)
    max_coord = np.ones(t.shape) * size_m[0]  # 1.0 m

    vals = vel.interp_at(np.stack((t, min_coord), axis=-1))  # vertical velocity at the bottom
    assert np.all(np.isclose(vals[:, 1], zero_v))  # y must be zero
    vals = vel.interp_at(np.stack((min_coord, t), axis=-1))  # horizontal velocity at the left edge
    assert np.all(np.isclose(vals[:, 0], zero_v))  # x must be zero

    vals = vel.interp_at(np.stack((t, max_coord), axis=-1))  # vertical velocity at the top
    assert np.all(np.isclose(vals[:, 1], zero_v))  # y must be 0.0
    vals = vel.interp_at(np.stack((max_coord, t), axis=-1))  # horizontal velocity at the right edge
    assert np.all(np.isclose(vals[:, 0], zero_v))    #


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_velocity()
    vf = VelocityField((1.0, 1.0), (6, 6))
    vf.randomize(scale=0.5)
    fig, ax = plt.subplots()
    vf.plot_grid(ax)
    vf.plot_velocities(ax, res=30, show_faces=True, show_field=True)
    plt.show()
