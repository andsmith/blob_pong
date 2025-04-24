import numpy as np
from abc import ABC, abstractmethod
import logging
import matplotlib.pyplot as plt
from fields import CenterScalarField
from gradients import gradient_central as gradient
from semi_lagrange import advect


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

    def add_sphere(self, center, radius, density):
        """
        Set all points in a specified sphere to a density value.
        :param center: The center of the sphere in world coordinates.
        :param radius: The radius of the sphere in world coordinates.
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

    def advect(self, velocity, dt):  # _lagrange
        """
        For each grid point, find the new velocity by moving the point backwards through the 
        velocity field for a time step dt. Then interpolate the density at the new position.
        """
        self._frame_no += 1
        # if self._frame_no==100:
        #    import ipdb; ipdb.set_trace()

        points = self._get_cell_centers()
        points_moved = advect(points, velocity, dt, self.dx, self.size,C=.5)
        old_values = self.interp_at(points_moved)

        self.values = old_values
        self.finalize()


def test_smoke():
    smoke = SmokeField((1., 1.), (50, 50))
    smoke.add_sphere((0.5, 0.5), 0.16, 10.0)
    fig, ax = plt.subplots()
    smoke.plot(ax, res=300, alpha=0.8)
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_smoke()
