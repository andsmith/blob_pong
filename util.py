import numpy as np
import logging


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


class CenterScalarField(object):
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
        self.name = name
        self.n_cells = grid_size
        self.dx = size_m[0] / grid_size[0]
        self.size = size_m
        self._init_grids(values)
        logging.info("Initializing %s grid %i x %i (dx = %.3f m) spanning (%.3f, %.3f) meters."
                     % (name, self.n_cells[0], self.n_cells[1], self.dx, self.size[0], self.size[1]))

    def _init_grids(self, values):

        # Coordinates of grid centers:
        self.centers_x = np.linspace(
            0.0, self.size[0], self.n_cells[0] + 1)[1:-1] + 0.5 * (self.size[0] / self.size[0])
        self.centers_y = np.linspace(
            0.0, self.size[1], self.n_cells[1] + 1)[1:-1] + 0.5 * (self.size[1] / self.size[1])

        # Initialize pressure field:
        self.values = np.zeros((self.n_cells[1], self.n_cells[0]), dtype=np.float32)

        if values is not None:
            self.values += values


    def plot(self, ax, alpha = 0.4):
        ax.imshow(self.values, cmap='jet', interpolation='bilinear', alpha=alpha, extent=(0, self.size[0], 0, self.size[1]))
        ax.set_title("%s field"% self.name)