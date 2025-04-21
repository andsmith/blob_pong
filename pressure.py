import numpy as np
# import jax.numpy as jnp
# from jax import grad, jit, vmap
import logging
from util import CenterScalarField


class PressureField(CenterScalarField):
    def __init__(self, size_m, grid_size, p0=1.0):
        """
        Initialize the pressure field.
        :param size_m: The size of the simulation domain in meters (width, height).
        :param grid_size: The number of cells in the x and y direction for the pressure field.
        :param p0: Initial pressure value.
        """
        super().__init__(size_m, grid_size, values=p0, name="Pressure")
        self.p0 = p0
        
