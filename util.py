import numpy as np


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
