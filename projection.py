"""
"Pressure projection" step.
"""

from sksparse.cholmod import cholesky
import numpy as np
import logging


class Projection(object):
    """
    Find the set of pressures so when velocities are adjusted for pressure differences, i.e.

         v_new <- v_temp - dt * (p_right -p_left)/dx, 

    the resulting velocity field is divergence fre  (where v_temp is velocity after advection, external forces, 
    and diffusion/viscosity but before pressure terms are added.)
    """

    def __init__(self, pressure_field, velocity_field, cell_flags, dt, free_velocities, boundary_velocities):
        """
        Initialize the projection object.
        :param pressure_field: The pressure field object.
        :param velocity_field: The velocity field object.
        :param cell_flags: Flags indicating which faces of which cells are free or fixed.
        :param dt: The time step size.
        :param free_velocities: The free velocity components.
        :param boundary_velocities: The boundary velocity components.
        """
        self.pressure_field = pressure_field
        self.velocity_field = velocity_field
        self.cell_flags = cell_flags
        self.dt = dt
        self.free_velocities = free_velocities
        self.boundary_velocities = boundary_velocities
