import numpy as np
import logging
from abc import ABC, abstractmethod
from interp_jax import Interp2d

def scale_y(size, n_x):
    """
    Find the y spacing of the grid based on the x-spacing and the aspect ratio.
    The y-spacing is the same as the x-spacing, but the height of the domain may not be a multiple of it.
    The y-spacing is expanded to the next multiple of the x-spacing.

    1. Determine dx = dy = size[0] / n_x
    2. Expand size[1] to the next multiple of dy.
    3. Determine n_y = size[1] / dy


    :param size: The size of the domain in meters (width, height).
    :param n_x: The number of cells in the x direction for the velocity field.
    :return: new_size: The new size of the domain in meters (width, height).
             (n_x, n_y): The number of cells in the y direction for the velocity field.
             dx:  side length of a cell in meters.
    """
    dx = size[0] / n_x
    n_y = int(np.ceil(size[1] / dx))
    size = size[0], dx * n_y  # size of the simulation domain in meters (width, height).
    return size, (n_x, n_y), dx


class InterpField(ABC):
    """
    Values (scalar/vector/etc) defined on a grid, might be needed somewhere else.
    Values can be modified but interpolation objects need to be re-initialized.
    For now control this with these methods:
      * mark_dirty:  Call after values are modified.
      * finalize:  Call to rebuild interpolation objects (use until values are modified again).
    """

    def __init__(self, size_m, grid_size, name):
        """
        Initialize the grid.
        :param size_m: The size of the simulation domain in meters (width, height).
        :param grid_size: The number of cells in the x and y direction for the field.
        """
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
    A scalar field defined at the center of the cells.
    The field is defined on a grid of size (n_x, n_y).
    The value is defined at the center of each cell, interpolated elsewhere.
    The size is in meters.
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
            0.0, self.size[1], self.n_cells[1] + 1)[:-1] + 0.5 * (self.size[1] /   self.n_cells[1])

        # Initialize pressure field:
        self.values = np.zeros((self.n_cells[1], self.n_cells[0]), dtype=np.float32)

        if values is not None:
            self.values += values


    def plot(self, ax, alpha = 0.4):
        ax.imshow(self.values, cmap='jet', interpolation='bilinear', alpha=alpha, extent=(0, self.size[0], 0, self.size[1]))
        ax.set_title("%s field"% self.name)