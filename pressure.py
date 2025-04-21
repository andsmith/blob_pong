import numpy as np
# import jax.numpy as jnp
# from jax import grad, jit, vmap
import logging
from util import CenterScalarField
import matplotlib.pyplot as plt


class PressureField(CenterScalarField):
    def __init__(self, size_m, grid_size, p0=1.0, unit='atm'):
        """
        Initialize the pressure field.
        :param size_m: The size of the simulation domain in meters (width, height).
        :param grid_size: The number of cells in the x and y direction for the pressure field.
        :param p0: Initial pressure value.
        """
        super().__init__(size_m, grid_size, values=p0, name="Pressure")
        self.p0 = p0
        self.unit = unit

    def randomize(self, seed=0, scale=1.0):
        """
        Randomize the pressure field.
        :param seed: Random seed.
        :param scale: Scale of the random noise.
        """
        rng = np.random.default_rng(seed)
        noise = rng.uniform(0, scale, self.values.shape)
        self.values += noise
        self.values = np.clip(self.values, 0, None)  # ensure non-negative pressure

    def plot(self, ax, alpha=0.4, res=200):

        # draw boundary lines
        ax.plot([0, 0], [0, self.size[1]], color='black', lw=2)
        ax.plot([self.size[0], self.size[0]], [0, self.size[1]], color='black', lw=2)
        ax.plot([0, self.size[0]], [0, 0], color='black', lw=2)
        ax.plot([0, self.size[0]], [self.size[1], self.size[1]], color='black', lw=2)

        # show grid points
        points = np.array([(x,y) for x in self.centers_x for y in self.centers_y])
        ax.scatter(points[:, 0], points[:, 1], color='black', s=1, alpha=0.5)

        if res == 0:
            # show pressures
            img = ax.imshow(self.values, extent=(0, self.size[0], 0, self.size[1]), alpha=alpha, cmap='jet')
        else:
            n_x_points = res
            aspect = self.size[1] / self.size[0]
            n_y_points = int(n_x_points * aspect)
            x, y = np.meshgrid(np.linspace(0, self.size[0], n_x_points),
                               np.linspace(0, self.size[1], n_y_points))
            values = self.interp_at(np.array([x.flatten(), y.flatten()]).T).reshape((n_y_points, n_x_points))
            img = ax.imshow(values, extent=(0, self.size[0], 0, self.size[1]), alpha=alpha, cmap='jet')
        ax.set_title("Pressure Field")
        # add colorbar
        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label('Pressure (%s)' % self.unit)


def _test_pressure(plot=True):

    pf = PressureField((1.0, 1.0), (20, 20), p0=1.0, unit='atm')
    pf.randomize(seed=0, scale=10)
    if plot:
        fig, ax = plt.subplots()
        pf.plot(ax, alpha=0.5)
        plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _test_pressure(plot=True)
