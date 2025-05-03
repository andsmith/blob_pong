"""
Semi-lagrangian method for 2d advection of a scalar field using a velocity field.
This is a numerical approximation to the equation:
     dC/dt = - v * grad(C)
where C is the scalar field (e.g. density, temperature, x-component of velocity, etc.) and v is the velocity field.
"""

#import numpy as np
import jax.numpy as np
import numpy
import logging
from jax import grad, jit, vmap

from loop_timing.loop_profiler import LoopPerfTimer as LPT


@LPT.time_function


def advect(points, velocity, dt, dx, size, C=.5):
    """
    :param points:  ... x 2 array of points to advect.
    :param velocity:  Velocity field object.
    :param dt:  Time step size.
    :param dx:  Grid spacing.
    :param size:  Upper limits of the simulation domain (x_max, y_max)
    """

    dt_sub, n_iter = velocity.get_cfl(dt, dx, C=C)
    # logging.info("\tadvection time step (%.3f) -> (%i x %.3f)."
    #                % (dt, n_iter, dt_sub))

    for iter in range(n_iter):
        velocities = velocity.interp_at(points)
        points -= dt_sub * velocities
        points[..., 0] = np.clip(points[..., 0], 0, size[0])
        points[..., 1] = np.clip(points[..., 1], 0, size[1])

    return points

@jit
def advect_jax(points, velocity, dt, dx, size, C=.5):
    """
    TODO:  FIX THIS (need to move interpolation of velocity to jax)
    :param points:  ... x 2 array of points to advect.
    :param velocity:  Velocity field object.
    :param dt:  Time step size.
    :param dx:  Grid spacing.
    :param size:  Upper limits of the simulation domain (x_max, y_max)
    """

    dt_sub, n_iter = velocity.get_cfl(dt, dx, C=C)
    # logging.info("\tadvection time step (%.3f) -> (%i x %.3f)."
    #                % (dt, n_iter, dt_sub))

    for iter in range(n_iter):
        velocities = velocity.interp_at(points)
        points -= dt_sub * velocities
        points_x= np.clip(points[..., 0], 0, size[0])
        points_y = np.clip(points[..., 1], 0, size[1])

    return np.stack((points_x, points_y), axis=-1)


def test_advect(n_times=10):
    from velocity import VelocityField
    import time
    n_grid = 10
    span = (1.0, 1.0)
    v = VelocityField(size_m=span, grid_size=(n_grid, n_grid)).randomize(scale=3.0)

    n_test_points = 100
    times = 0.
    for _ in range(n_times):
        points = numpy.random.rand(n_test_points**2, 2) * np.array(span)
        t0 = time.perf_counter()
        new_points = advect(points, v, dt=0.1, dx=1./n_grid, size=span)
        t1 = time.perf_counter()
        times += t1 - t0
    logging.info("")
    logging.info(f"Average time for {n_times} advect calls({n_test_points**2} points): {times/n_times*1000:.3f} ms.")





if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    test_advect()