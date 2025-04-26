import numpy as np
import logging
from abc import ABC, abstractmethod
#from interpolation import Interp2d
from interp_jax import Interp2d

from util import scale_y


class InterpField(ABC):
    """
    Values (scalar/vector/etc) defined on a grid, might be needed somewhere else.
    Values can be modified but interpolation objects need to be re-initialized.

    For now, control this with these methods:
      * mark_dirty:  Call after values are modified (if used in mulitple places).
      * finalize:  Call to rebuild interpolation objects (use until values are modified again).

    (Alternatively, call finalize() after every modification, but this is slow.)
    """

    def __init__(self, size_m, grid_size, name, rnd_seed=0):
        """
        Initialize the grid.
        :param size_m: The size of the simulation domain in meters (width, height).
        :param grid_size: The number of cells in the x and y direction for the field.
        """
        self._rng = np.random.default_rng(rnd_seed)
        self.name = name
        self.n_cells = grid_size
        self.dx = size_m[0] / grid_size[0]
        dy = size_m[1] / grid_size[1]
        if (np.abs(self.dx - dy) > 1e-10):
            raise ValueError(
                "Grid spacing (span/n_cells) is not valid, must be equal for x and y, was: %.4f, %.4f" % (self.dx, dy))
        self.size = size_m
        self._interp = None
        logging.info("Created Interpolation field '%s':  %i x %i (dx = %.3f m) spanning (%.3f, %.3f) meters."
                     % (self.name, self.n_cells[0], self.n_cells[1], self.dx, self.size[0], self.size[1]))

    @abstractmethod
    def _interp_at(self, points):
        """
        Use the interpolation object to get the value at the given points.
        assume self._interp is not None.
        """
        pass

    @abstractmethod
    def _get_interp(self):
        """
        Create the interpolation object
        """
        pass

    def finalize(self):
        # subclasses should override & add anything else needed to finalize the field, e.g. boundary conditions.
        self._interp = self._get_interp()

    def mark_dirty(self):
        logging.info("Clearing interpolator:  %s" % self.name)
        self._interp = None

    def interp_at(self, points):
        """
        Get the velocity at the given points.
        """
        if self._interp is None:
            logging.info("No interpolator, re-initializing:  %s" % self.name)
            self.finalize()
        return self._interp_at(points)


class CenterScalarField(InterpField):
    """
    A scalar field is defined at the center of the cells & interpolated elsewhere.
      (pressure, density, etc.)
    """

    def __init__(self, size_m, grid_size, values=None, name="(scalar)"):
        """
        Initialize the scalar field.
        :param size_m: The size of the simulation domain in meters (width, height).
        :param grid_size: The number of cells in the x and y direction for the scalar field.
        :param values: Initial values for the field:
             If None, the field is initialized to zero.
             If a scalar, the field is initialized to that value.
             If a 2d arary of values, those values are used.  (Other shapes raise an error.)
        :param name: The name of the field.
        """
        super().__init__(size_m, grid_size, name=name)
        self._init_coord_grids(values)

    def randomize(self, scale=1.0):
        """
        Randomize the pressure field.
        :param scale: Scale of the random noise.
        """
        noise = self._rng.uniform(0, scale, self.values.shape)
        self.values += noise
        self.values = np.clip(self.values, 0, None)  # ensure non-negative pressure
        
    def _get_interp(self):
        """
        Create the interpolation object for the scalar field.
        The interpolation object is created using the center of the cells.
        """
        p0 = (self.centers_x[0], self.centers_y[0])
        return Interp2d(p0, self.dx, size=self.n_cells, value=self.values)

    def _interp_at(self, points):
        return self._interp.interpolate(points)

    def _init_coord_grids(self, values):

        # Coordinates of grid centers:
        self.centers_x = np.linspace(
            0.0, self.size[0], self.n_cells[0] + 1)[:-1] + 0.5 * (self.size[0] / self.n_cells[0])
        self.centers_y = np.linspace(
            0.0, self.size[1], self.n_cells[1] + 1)[:-1] + 0.5 * (self.size[1] / self.n_cells[1])

        # Initialize pressure field:
        self.values = np.zeros((self.n_cells[1], self.n_cells[0]), dtype=np.float32)

        if values is not None:
            self.values += values

    def plot(self, ax, alpha=0.4, res=200, title="(scalar)"):

        # draw boundary lines
        ax.plot([0, 0], [0, self.size[1]], color='black', lw=2)
        ax.plot([self.size[0], self.size[0]], [0, self.size[1]], color='black', lw=2)
        ax.plot([0, self.size[0]], [0, 0], color='black', lw=2)
        ax.plot([0, self.size[0]], [self.size[1], self.size[1]], color='black', lw=2)

        # show grid points
        #points = np.array([(x, y) for x in self.centers_x for y in self.centers_y])
        #ax.scatter(points[:, 0], points[:, 1], color='black', s=1, alpha=0.5)

        if res == 0:
            # show values
            img = ax.imshow(self.values[::-1,:], extent=(0, self.size[0], 0, self.size[1]), alpha=alpha, cmap='jet')
        else:
            n_x_points = res
            aspect = self.size[1] / self.size[0]
            n_y_points = int(n_x_points * aspect)
            x, y = np.meshgrid(np.linspace(0, self.size[0], n_x_points),
                               np.linspace(0, self.size[1], n_y_points))
            values = self.interp_at(np.array([x.flatten(), y.flatten()]).T).reshape((n_y_points, n_x_points))
            img = ax.imshow(values[::-1,:], extent=(0, self.size[0], 0, self.size[1]), alpha=alpha, cmap='jet')
        mass = np.sum(self.values) * self.dx * self.dx
        title = "%s\nmass = %.2f kg, range(%.3f, %.3f)" % (self.name, mass, self.values.min(), self.values.max())
        ax.set_title(title)
        return img
