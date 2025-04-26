"""
This module does simple bilinear interpolation using JAX.
No-slip / free-slip condtions can be enforced at the boundaries by clamping
outside values to zero for the relevant components.
The interpolation is done using the following method:
1. values are defined on a cartesian grid of size (n_x, n_y).

2. THe interpolated value at a point (x, y) is calculated by
   finding the four nearest grid points and using bilinear interpolation. 

3. If (x,y) falls outside the grid on a ZERO-boundary, it is linearly interpolated 
   between the weighted average of the two nearest grid points and zero. 
   If it falls on a FREE-boundary, it is just the weighted average of the two nearest grid points.

TODO: 4. Include a signed-distance function and signed-motion function so velocities near
    objects can have no-slip or free-slip conditions.  (without this, the velocities on 
    the boundary must be manually set to that boundary's velocity.  Ideally both should happen.)

Jax is use for just-in-time compilation (jit) and vectorization (vmap).

"""
import numpy as np
from scipy.interpolate import RectBivariateSpline
import logging
from multiprocessing import cpu_count, Pool
import time
#from interp_jax import Interp2d as Interp2d_jax


class Interp2d(object):
    """
    Bilinear interpolation on a 2D grid.
    """

    def __init__(self, p0, dp, size=None, value=0.):
        """
        Initialize the Interp2d object.
        :param p0: (float, float)The x- and y-coordinates of the lower left corner of the grid.
        :param dp:  float,  The x and y spacing of the grid
        :param size: The shape of the grid (n_x, n_y). If None, it will be determined from the values.
        :param value: initial value(s) of the grid. If None, the grid will be initialized to zero.
            if scalar float, size must be not None.
            if 2D array, size is ignored.
        """
        self.p_min = np.array(p0)
        self.dp = float(dp)
        if size is None:
            if value is None:
                raise ValueError("Shape must be provided if values are not.")
            else:
                self.size = value.shape[1], value.shape[0]
                self.values = value
        else:
            self.size = size
            self.values = np.zeros((size[1], size[0]), dtype=np.float32) + value
        self.p_max = self.p_min + (np.array(self.size)-1) * np.array(self.dp)
        #logging.info("Initialized interpolation field %i x %i (dx = %.3f m) spanning (x = [%.3f, %.3f], y = [%.3f, %.3f]) meters, with grid shaped %s."
        #             % (self.size[0], self.size[1], self.dp, self.p_min[0], self.p_max[0], self.p_min[1], self.p_max[1], self.values.shape))

    def upper_left_inds(self, x, y):
        """
        Return the indices of the upper-left corner of the grid cell containing (x, y).
        """
        i0 = np.floor((x - self.p_min[0]) / self.dp).astype(int)
        j0 = np.floor((y - self.p_min[1]) / self.dp).astype(int)
        return i0, j0

    def interpolate(self, points):
        """
        Interpolate the values at the given points.
        If out of bounds, clip to bounds
        """
        points = np.array(points)
        x = points[..., 0]
        y = points[..., 1]
        i0, j0 = self.upper_left_inds(x, y)
        i1 = i0 + 1
        j1 = j0 + 1

        # Clip the indices to the grid bounds
        i0 = np.clip(i0, 0, self.size[0] - 1)
        j0 = np.clip(j0, 0, self.size[1] - 1)
        i1 = np.clip(i1, 0, self.size[0] - 1)
        j1 = np.clip(j1, 0, self.size[1] - 1)

        # Get the relative offsets within each cell
        t_x = (x - (self.p_min[0] + i0 * self.dp)) / self.dp
        t_y = (y - (self.p_min[1] + j0 * self.dp)) / self.dp

        # Get the values at the grid points
        v00 = self.values[(j0, i0)]  # V01        V11
        v01 = self.values[(j1, i0)]  # <-tx->p
        v10 = self.values[(j0, i1)]  # .     ^ (ty)
        v11 = self.values[(j1, i1)]  # V00...v    V10

        # Get the areas of the four rectangles around the point p in the cell:
        a_lower_left = t_x * t_y
        a_lower_right = (1 - t_x) * t_y
        a_upper_left = t_x * (1 - t_y)
        a_upper_right = (1 - t_x) * (1 - t_y)

        # Perform bilinear interpolation
        interp_values = (v11 * a_lower_left + v10 * a_upper_left +
                         v01 * a_lower_right + v00 * a_upper_right)
        return interp_values

def test_interp_diff(plot=False):
    """
    Show the difference between our implementation and RectBivariateSpline.
    :param plot: If True, show the images, otherwise just print the values.
    """
    x = np.array((0, 1))
    y = np.array((0, 1))
    z = np.array([[2.0, 2.0],  # function to be interpolated at (x,y)
                  [2.0, 1.0]])
    dx = 1.0
    p0 = (x[0], y[0])

    interp_test = Interp2d(p0, dx, size=None, value=z)
    interp_control = RectBivariateSpline(x, y, z.T, kx=1, ky=1)

    test_points = [(0.5, 0.75),
                   (0.25, 0.25),
                   (0.75, 0.25),
                   (0.75, 0.75),
                   (0.5, 0),
                   (0, 0.5),
                   (1, 1),
                   (1, 0),
                   (0, 1)]

    for test_point in test_points:
        control_value = interp_control(test_point[0], test_point[1], grid=False)
        interp_value = interp_test.interpolate(test_point)
        print(f"Test point: {test_point}, Interpolated value: {interp_value}, Control value: {control_value}")
        assert np.isclose(interp_value, control_value), f"Interpolation=?=control failed for point {test_point}"

    if plot:
        import matplotlib.pyplot as plt
        extent = (0, 1, 0, .7)  # must be same for dx to be the same
        n_points = 10
        x_coords = np.linspace(extent[0], extent[1], n_points)
        y_coords = x_coords[x_coords < extent[3]]
        print("mean dx: %.5f, mean_dy: %.5f" % (np.mean(np.diff(x_coords)), np.mean(np.diff(y_coords))))
        dx = np.mean((np.mean(np.diff(x_coords)), np.mean(np.diff(y_coords))))
        p0 = (x[0], y[0])
        x, y = np.meshgrid(x_coords, y_coords)
        vals = np.sin(x*20) + np.cos(15*y)
        interp_test = Interp2d(p0, dx, size=None, value=vals)
        control_test = RectBivariateSpline(x_coords, y_coords, vals.T, kx=1, ky=1)

        # interpolate at these points:
        nx_test = n_points*10
        ny_test = nx_test
        m = .2
        test_extent = (extent[0]-m, extent[1]+m, extent[2]-m, extent[3]+m)
        xt, yt = np.linspace(test_extent[0], test_extent[1], nx_test), np.linspace(
            test_extent[2], test_extent[3], ny_test)
        x_test, y_test = np.meshgrid(xt, yt)
        x_test = x_test.reshape(-1)
        y_test = y_test.reshape(-1)

        # interpolate the image and a bit of the margin outside to see the boundary conditions.
        interp_img = interp_test.interpolate(np.stack((x_test, y_test), axis=-1)).reshape(ny_test, nx_test)
        interp_img_control = control_test(x_test, y_test, grid=False).reshape(ny_test, nx_test)

        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
        ax = ax.flatten()
        # ax[0].imshow(values, cmap='jet', interpolation='none', extent=extent)
        # ax[1].imshow(interp_img, cmap='jet', interpolation='none', extent=test_extent)
        # ax[2].imshow(interp_img_control, cmap='jet', interpolation='none', extent=test_extent)

        def _show_img(ax, img, title):
            img = ax.imshow(img, cmap='jet', interpolation='none', extent=test_extent)
            plt.colorbar(img, ax=ax)
            # draw box around extent
            ax.add_patch(plt.Rectangle((extent[0], extent[2]), extent[1]-extent[0], extent[3]-extent[2],
                                       fill=False, edgecolor='green', linewidth=2))
            ax.set_title(title)
            ax.set_xlim(test_extent[0], test_extent[1])
            ax.set_ylim(test_extent[2], test_extent[3])

        _show_img(ax[0], vals, "Original")
        _show_img(ax[1], interp_img_control, "Control Interpolation")
        _show_img(ax[2], interp_img, "Test Interpolation")
        _show_img(ax[3], interp_img - interp_img_control, "Difference Image")
        plt.show()


def _trial(image, n_interp, n_test_points, interpolator):
    times = {}
    t_start = time.perf_counter()
    interp_test = interpolator((0, 0), 1.0, size=None, value=image)
    now = time.perf_counter()
    times['t_interp_create'] = (now - t_start)
    t_start = now
    interp_control = RectBivariateSpline(np.arange(image.shape[1]), np.arange(image.shape[0]), image.T, kx=1, ky=1)
    now = time.perf_counter()
    times['t_control_create'] = (now - t_start)
    t_start = now
    image_size = np.array((image.shape[1], image.shape[0]))
    times['t_interp'] = 0.0
    times['t_control'] = 0.0
    times['max_diff'] = 0.0
    trial_start = time.perf_counter()
    for iter in range(n_interp):
        test_points = np.random.rand(n_test_points, 2) * image_size
        interp_vals = interp_test.interpolate(test_points)
        now = time.perf_counter()
        times['t_interp'] += (now - t_start)
        t_start = now
        control_vals = interp_control(test_points[:, 0], test_points[:, 1], grid=False)
        now = time.perf_counter()
        times['t_control'] += (now - t_start)
        t_start = now
        max_diff = np.max(np.abs(interp_vals - control_vals))   
        times['max_diff'] = max_diff if max_diff > times['max_diff'] else times['max_diff']
    print("\tCompleted %i tests in %.3f seconds" % (n_interp, time.perf_counter() - trial_start))
    times['t_interp'] /= n_interp
    times['t_control'] /= n_interp
    return times


def speed_test(image_size=(640, 480), n_trials=100, n_interp=20,interpolator=None, n_cores=14):
    """
    Speed test for the interpolation, can it run in parallel?

    Run the naive/jax comparison test in parallel on n_cores.

    :param image_size: The size of the image to be interpolated.
    :param n_trials: The number of trials to run.
    :param n_interp: The number of times to re-interpolate the image for each trial
    :param interpolator: The interpolator class to be used. If None, use the default Interp2d class.
    """
    from multiprocessing import Pool
    if interpolator is None:
        from interp_jax import Interp2d as Interp2d_jax
        interpolator = Interp2d_jax
        #raise Exception("JAX not installed, please install it to use this feature.")
    image_size = np.array(image_size)
    n_test_points = image_size[0]*image_size[1]

    times = {'t_interp_create': [],
             't_interp': [],
             't_control_create': [],
             't_control': [],
             'max_diff': []}
    work = []
    for t_num in range(n_trials):
        image = np.random.rand(image_size[1], image_size[0])
        work.append((image, n_interp, n_test_points, interpolator))
    
    n_cores = cpu_count() if n_cores==0 else n_cores

    if n_cores==1:
        print("Computing single core, %i tasks, testing class: %s." % (len(work), interpolator.__name__))
        results = [_trial(*args) for args in work]
    else:
        print("Computing multi-core(%i), %i tasks, testing class: %s." % (n_cores, len(work), interpolator.__name__))
        with Pool(processes=n_cores) as pool:
            results = pool.starmap(_trial, work)
    for result_times in results:
        for key in times.keys():
            times[key].append(result_times[key])

    print("============\nAverage times:")
    for key, value in times.items():
        print(f"{key}: {np.mean(value):.6f}")
    print("============\nSpeedup:  %.3f" % (np.mean(times['t_control']) / np.mean(times['t_interp'])))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_interp_diff(plot=False)
    #speed_test()
    logging.info("All tests passed.")
    
