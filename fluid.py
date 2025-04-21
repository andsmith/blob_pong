import numpy as np
from abc import ABC, abstractmethod
from util import CenterScalarField

class FluidField(CenterScalarField, ABC):
    """
    Base class for fluid.
        subclasses:  Gasses will be density, liquids will be signed distance
    """
    def __init__(self, size_m, grid_size, values=None):
        super().__init__(size_m, grid_size, values=values, name=self.__class__.__name__)
    

class SmokeField(FluidField):
    def __init__(self, size_m, grid_size):
        """
        Initialize the smoke field.
        self.values will be the density of the smoke.
        :param size_m: The size of the simulation domain in meters (width, height).
        :param grid_size: The number of cells in the x and y direction for the smoke field.
        """
        super().__init__(size_m, grid_size)
        