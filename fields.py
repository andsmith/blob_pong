import numpy as np
import logging
from abc import ABC, abstractmethod
# from interpolation import Interp2d
from interp_jax import Interp2d
from gradients import gradient_upwind
from util import scale_y
import cv2

from loop_timing.loop_profiler import LoopPerfTimer as LPT
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

    @abstractmethod
    def gradient(self, extent='valid'):
        """
        Get the gradient of the field at the grid points.
        :param extent: The extent of the gradient ('valid' or 'same').
        :return: The gradient of the field at all grid points.
        """
        pass

    def grid_to_pixel(self, bbox, coords):
        """
        Convert grid coordinates to pixel coordinates.
        :param bbox: The bounding box of where the grid will be drawn on an image,
            {'x': (x_left, x_right), 'y': (y_bottom, y_top)}
        :param coords: The coordinates to convert. [..., 2]
        :return: The pixel coordinates.
        """
        x_left, y_bottom = bbox['x'][0], bbox['y'][0]
        x_right, y_top = bbox['x'][1], bbox['y'][1]
        x_span = x_right - x_left
        y_span = y_top - y_bottom
        grid_x_max = self.n_cells[0] * self.dx
        grid_y_max = self.n_cells[1] * self.dx
        # Convert to pixel coordinates:
        pixel_coords = np.zeros_like(coords)
        pixel_coords[..., 0] = (coords[..., 0]/grid_x_max) * x_span + x_left
        pixel_coords[..., 1] = (coords[..., 1]/grid_y_max) * y_span + y_bottom

        return pixel_coords

    def pixel_to_grid(self, bbox, coords):
        """
        Convert pixel coordinates to grid coordinates.
        :param bbox: The bounding box of where the grid will be drawn on an image,
            {'x': (x_left, x_right), 'y': (y_bottom, y_top)}
        :param coords: The coordinates to convert. [..., 2]
        :return: The grid coordinates.
        """
        x_left, y_bottom = bbox['x'][0], bbox['y'][0]
        x_right, y_top = bbox['x'][1], bbox['y'][1]
        x_span = x_right - x_left
        y_span = y_top - y_bottom
        grid_x_max = self.n_cells[0] * self.dx
        grid_y_max = self.n_cells[1] * self.dx
        # Convert to pixel coordinates:
        grid_coords = np.zeros_like(coords)
        grid_coords[..., 0] = (coords[..., 0]-x_left)/x_span * grid_x_max
        grid_coords[..., 1] = (coords[..., 1]-y_bottom)/y_span * grid_y_max

        return grid_coords
    @LPT.time_function
    def render_grid(self, image, bbox, line_color, only_bbox=False):
        """
        Render the grid on the image w/thickness 1, bounding box with thickness 2.
        :param image: The image to render on.
        :param bbox: The bounding box of the grid.
        :param line_color: The color of the grid lines.
        """
        SHIFT_B = 6
        SHIFT_M = 2**SHIFT_B
        img_size_wh = image.shape[1], image.shape[0]
        grid_top, grid_left = self.n_cells[1] * self.dx, self.n_cells[0] * self.dx
        grid_bottom, grid_right = 0, 0
        smallest = self.grid_to_pixel(bbox, np.array((grid_left, grid_bottom)).reshape(1, -1)).reshape(-1)
        x_left, y_bottom = smallest[0], smallest[1]
        biggest = self.grid_to_pixel(bbox, np.array((grid_right, grid_top)).reshape(1, -1)).reshape(-1)
        x_right, y_top = biggest[0], biggest[1]
        # Draw the bounding box:
        cv2.rectangle(image,
                      (int(x_left*SHIFT_M), int(y_bottom*SHIFT_M)),
                      (int(x_right*SHIFT_M), int(y_top*SHIFT_M)), line_color, 2, cv2.LINE_AA, shift=SHIFT_B)
        if only_bbox:
            return
        # Draw the grid lines:
        for i in range(self.n_cells[0] + 1):    # vertical lines    
            x = i * self.dx
            x_pixel = self.grid_to_pixel(bbox, np.array((x, 0)).reshape(1, -1)).reshape(-1)[0]
            cv2.line(image, (int(x_pixel*SHIFT_M), int(y_bottom*SHIFT_M)),
                     (int(x_pixel*SHIFT_M), int(y_top*SHIFT_M)), line_color, 1, cv2.LINE_AA, shift=SHIFT_B)
        for i in range(self.n_cells[1] + 1):    # horizontal lines
            y = i * self.dx
            y_pixel = self.grid_to_pixel(bbox, np.array((0, y)).reshape(1, -1)).reshape(-1)[1]
            cv2.line(image, (int(x_left*SHIFT_M), int(y_pixel*SHIFT_M)),
                     (int(x_right*SHIFT_M), int(y_pixel*SHIFT_M)), line_color, 1, cv2.LINE_AA, shift=SHIFT_B)


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

    def gradient(self):
        """
        Get the gradient of the field at the grid points.
        :param method: The method to use for the gradient ('upwind' or 'central').
        :return: The gradient of the field at the grid points.
        """
        dx, dy = gradient_upwind(self.values, self.dx)
        # Return the gradient as a tuple of arrays
        return dx, dy

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
        # points = np.array([(x, y) for x in self.centers_x for y in self.centers_y])
        # ax.scatter(points[:, 0], points[:, 1], color='black', s=1, alpha=0.5)

        if res == 0:
            # show values
            img = ax.imshow(self.values[::-1, :], extent=(0, self.size[0], 0, self.size[1]), alpha=alpha, cmap='jet')
        else:
            n_x_points = res
            aspect = self.size[1] / self.size[0]
            n_y_points = int(n_x_points * aspect)
            x, y = np.meshgrid(np.linspace(0, self.size[0], n_x_points),
                               np.linspace(0, self.size[1], n_y_points))
            values = self.interp_at(np.array([x.flatten(), y.flatten()]).T).reshape((n_y_points, n_x_points))
            img = ax.imshow(values[::-1, :], extent=(0, self.size[0], 0, self.size[1]), alpha=alpha, cmap='jet')
        mass = np.sum(self.values) * self.dx * self.dx
        title = "%s\nmass = %.2f kg, range(%.3f, %.3f)" % (self.name, mass, self.values.min(), self.values.max())
        ax.set_title(title)
        return img
