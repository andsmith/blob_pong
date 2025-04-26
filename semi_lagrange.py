"""
Semi-lagrangian method for 2d advection of a scalar field using a velocity field.
This is a numerical approximation to the equation:
     dC/dt = - v * grad(C)
where C is the scalar field (e.g. density, temperature, x-component of velocity, etc.) and v is the velocity field.
"""

import numpy as np
import logging


def advect(points, velocity, dt, dx, size, C=.5):
    """
    :param points:  ... x 2 array of points to advect.
    :param velocity:  Velocity field object.
    :param dt:  Time step size.
    :param dx:  Grid spacing.
    :param size:  Upper limits of the simulation domain (x_max, y_max)
    """
    
    dt_sub, n_iter = velocity.get_cfl(dt, dx, C=C)
    #logging.info("\tadvection time step (%.3f) -> (%i x %.3f)."
    #                % (dt, n_iter, dt_sub))
    
    for iter in range(n_iter):
        velocities = velocity.interp_at(points)
        points -= dt_sub * velocities
        points[..., 0] = np.clip(points[..., 0], 0, size[0])
        points[..., 1] = np.clip(points[..., 1], 0, size[1])

    return points
        