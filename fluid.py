import numpy as np
from abc import ABC, abstractmethod
import logging
import matplotlib.pyplot as plt
from fields import CenterScalarField
from semi_lagrange import advect
from loop_timing.loop_profiler import LoopPerfTimer as LPT
import cv2


class FluidField(CenterScalarField, ABC):
    """
    Base class for fluid.
        subclasses:  Gasses will be density, liquids will be signed distance
    """

    def __init__(self, size_m, grid_size, values=None):
        super().__init__(size_m, grid_size, values=values, name=self.__class__.__name__)
        self._frame_no = 0


class SmokeField(FluidField):
    def __init__(self, size_m, grid_size):
        """
        Initialize the smoke field.
        self.values will be the density of the smoke.
        :param size_m: The size of the simulation domain in meters (width, height).
        :param grid_size: The number of cells in the x and y direction for the smoke field.
        """
        super().__init__(size_m, grid_size)

    def _get_cell_centers(self):
        points_x, points_y = np.meshgrid(self.centers_x, self.centers_y)
        points = np.stack((points_x, points_y), axis=-1)
        return points

    def add_circle(self, center, radius, density):
        """
        Set all points in a specified circle to a density value.
        :param center: The center of the circle in world coordinates.
        :param radius: The radius of the circle in world coordinates.
        :param density: The density value to set.
        """
        center = np.array(center).reshape(1, 1, 2) * self.size  # convert to grid coordinates
        points = self._get_cell_centers()
        distances = np.linalg.norm(points - center, axis=2)
        inside = np.where(distances < radius*self.size[0])
        self.values[inside] = density
        self.finalize()

    def plot(self, ax, res=1000, alpha=0.8):
        return super().plot(ax, res=res, alpha=alpha, title="Smoke density")

    @LPT.time_function
    def advect(self, velocity, dt, C=.5):  # _lagrange
        """
        For each grid point, find the new velocity by moving the point backwards through the 
        velocity field for a time step dt. Then interpolate the density at that position.
        """
        self._frame_no += 1
        
        points = self._get_cell_centers()
        points_moved = advect(points, velocity, dt, self.dx, self.size, C=C)

        old_values = self.interp_at(points_moved)

        self.values = old_values
        self.finalize()

    @LPT.time_function
    def render(self, img, d_max, bbox, fl_color, bkg_color):
        """
        Render the smoke field onto an image.
        :param d_max: The maximum density value for normalization.
        :param img: The image to render onto.
        :param bbox: The bounding box for the rendering. 
          {'x': (x_min, x_max), 'y': (y_min, y_max)}
        :param color: RGB triplet of maximum density color (interpolated between this and the image.)
        """
        
        # Normalize the density values
        normalized_vals = np.clip(self.values / d_max, 0, 1)

        # make a single-pixel image then scale it up and put it in the right place
        fl_color = np.array(fl_color, dtype=np.uint8).reshape(1, 1, 3)
        bkg_color = np.array(bkg_color, dtype=np.uint8).reshape(1, 1, 3)

        img_small = normalized_vals[...,np.newaxis] * fl_color + (1 - normalized_vals[...,np.newaxis]) * bkg_color  # blend with background color
        patch = cv2.resize(img_small, (bbox['x'][1] - bbox['x'][0], bbox['y'][1] - bbox['y'][0]), interpolation=cv2.INTER_NEAREST)
        img[bbox['y'][0]:bbox['y'][1], bbox['x'][0]:bbox['x'][1]] = patch


def test_smoke():
    # Matplotlib display
    smoke = SmokeField((1., 1.), (150, 150))
    smoke.add_circle((0.5, 0.5), 0.16, 10.0)
    fig, ax = plt.subplots()
    smoke.plot(ax, res=300, alpha=0.8)
    plt.show()

    # OpenCV render
    img = np.zeros((900,900, 3), dtype=np.uint8)
    bbox = {'x': (0, 900), 'y': (0, 900)}
    bkg_color = 246, 238, 227  # Background
    fluid_color = 0, 4, 51
    smoke.render(img, 10.0, bbox, fluid_color, bkg_color)
    cv2.imshow("Smoke", img[:,:, ::-1])
    cv2.waitKey(0)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_smoke()
