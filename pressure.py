import numpy as np
# import jax.numpy as jnp
# from jax import grad, jit, vmap
import logging
from fields import CenterScalarField
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


    def plot(self, ax,**kwargs):
        return super().plot(ax,title="Pressure Field",**kwargs)

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
