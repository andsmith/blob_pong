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
import jax.numpy as jnp
from jax import grad, jit, vmap
from numba import jit, float64, int32
from scipy.interpolate import RectBivariateSpline
import logging


class Interp2d(object):
    """
    Bilinear interpolation on a 2D grid.
    """

    def __init__(self, p0, dp, size=None, value=0., bounds_h_v=('zero', 'zero')):
        """
        Initialize the Interp2d object.
        :param p0: (float, float)The x- and y-coordinates of the lower left corner of the grid.
        :param dp:  float,  The x and y spacing of the grid
        :param size: The shape of the grid (n_x, n_y). If None, it will be determined from the values.
        :param value: initial value(s) of the grid. If None, the grid will be initialized to zero.
            if scalar float, size must be not None.
            if 2D array, size is ignored.


        :param bounds_hv: The boundary conditions for extrapolating verticaly (points above/below the grid)
          and horizontally (points to the left/right of the grid): 
            'zero' for no-slip (clamp to zero a half space outside the grid, interpolate linearly in that band)
            'free' for free-slip (constant equal to value at closest point on the grid edge whenever outside the grid)
        """
        self.p_min = np.array(p0)
        self.dp = float(dp)
        self.bounds_h_v = bounds_h_v
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
        logging.info("Initialized interpolation field %i x %i (dx = %.3f m) spanning (x = [%.3f, %.3f], y = [%.3f, %.3f]) meters, with grid shaped %s."
                     % (self.size[0], self.size[1], self.dp, self.p_min[0], self.p_max[0], self.p_min[1], self.p_max[1], self.values.shape))

    def inside_horizontal(self, x):
        """
        Check if the x-coordinates are inside the grid.
        :param x: The x-coordinates of the points to check.
        :return: A boolean array indicating if each point is inside the grid.
        """
        return (x >= self.p_min[0]) & (x < self.p_max[0])

    def inside_vertical(self, y):
        """
        Check if the y-coordinates are inside the grid.
        :param y: The y-coordinates of the points to check.
        :return: A boolean array indicating if each point is inside the grid.
        """
        return (y >= self.p_min[1]) & (y < self.p_max[1])

    def upper_left_inds(self, x, y):
        """
        Return the indices of the upper-left corner of the grid cell containing (x, y).
        """
        i0 = np.floor((x - self.p_min[0]) / self.dp).astype(int)
        j0 = np.floor((y - self.p_min[1]) / self.dp).astype(int)
        return i0, j0

    def interp(self, points):
        """
        Interpolate the values at the given points.
        For points inside the grid use bilinear interpolation.
        For points outside the grid apply the appropriate boundary rule interpolation.
        (Note: points can be outside both horizontally and vertically, both rules will be applied.)
        :param points: a (..., 2) element array of points to interpolate.
        :return: a (..., 1) element array of interpolated values at the points.
        """
        points = np.array(points)
        if len(points.shape) == 1:
            points = points.reshape(1, -1)
        outside_v_mask = ~self.inside_vertical(points[:, 1])
        outside_h_mask = ~self.inside_horizontal(points[:, 0])
        inside_mask = ~(outside_v_mask | outside_h_mask)

        interp_values = np.zeros_like(points[:, 0])
        interp_values[inside_mask] = self._interp(points[inside_mask])
        if np.any(outside_h_mask):
            interp_values[outside_h_mask] = self._interp_horiz_outside(points[outside_h_mask])
        if np.any(outside_v_mask):
            interp_values[outside_v_mask] = self._interp_vert_outside(points[outside_v_mask])

        return interp_values

    def _interp(self, points):
        """
        These points are inside the grid, just do regular bilinear interpolation.
        :param points: an N x 2 element array of points to interpolate.
        :return: an N x 2 element array of interpolated values at the points.
        """
        x = points[:, 0]
        y = points[:, 1]
        i0, j0 = self.upper_left_inds(x, y)
        i1 = i0 + 1
        j1 = j0 + 1

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


def test_interp_free(plot=True):
    x = np.array((0, 1, 2))
    y = np.array((0, 1, 2))
    dx = 1.0
    z = np.array([[1, 1, 1],
                  [2, 3, 4],
                  [.1, 1.2, 2.2]])
    inter_ctrl = RectBivariateSpline(x, y, z.T, kx=1, ky=1)
    interp_test = Interp2d((x[0], y[0]), dx, size=None, value=z, bounds_h_v=('zero', 'zero'))

    # test single values
    tests = [{'point': (0.0, 0.5), 'value': 1.5},  # left edge midway up, between values 1 and 2
             {'point': (2.0-1e-8, 0.5), 'value': 2.5},  # right edge midway up, between values 1 and 4
             # TODO: Add more edges & interior checks
             ]  # outside grid in both directions
    for test in tests:
        test_point = test['point']
        expected_value = test['value']
        
        interp_value = interp_test.interp(test_point)[0]
        control_value = inter_ctrl(test_point[0], test_point[1], grid=False)
        print("Test point: %s, Expected value: %.3f, Interpolated value:%.3f, Control value:%.3f" %
              (test_point, expected_value, interp_value, control_value))
        assert np.isclose(interp_value, expected_value, rtol=1e-5, atol=1e-6), f"Interpolation failed for point {test_point}"

    if plot:
        import matplotlib.pyplot as plt
        
        x_lim = x.min(), x.max()-1e-8
        y_lim = y.min(), y.max()-1e-8
        x = np.linspace(x_lim[0], x_lim[1], 200)
        y = np.linspace(y_lim[0], y_lim[1], 200)
        print(x.shape, y.shape, z.shape)
        test_x, test_y = np.meshgrid(x, y)
        img_ctrl = inter_ctrl(test_x, test_y, grid=False)
        coords = np.array((test_x.flatten(), test_y.flatten())).T
        img_test = interp_test.interp(coords)
        img_test = img_test.reshape(test_x.shape)
        fig, ax = plt.subplots(ncols=2, nrows=2)
        ax = ax.flatten()
        # show original
        img = ax[0].imshow(z, cmap='jet', interpolation='none', extent=(x[0], x[-1], y[0], y[-1]))
        plt.colorbar(img, ax=ax[0])
        ax[0].set_title("Original")

        # show control interpolation
        img = ax[1].imshow(img_ctrl, cmap='jet', interpolation='none', extent=(x[0], x[-1], y[0], y[-1]))
        plt.colorbar(img, ax=ax[1])
        ax[1].set_title("Control Interpolation")

        # show test interpolation
        img = ax[2].imshow(img_test, cmap='jet', interpolation='none', extent=(x[0], x[-1], y[0], y[-1]))
        plt.colorbar(img, ax=ax[2])
        ax[2].set_title("Test Interpolation")

        # show difference image
        img = ax[3].imshow(img_test - img_ctrl, cmap='jet', interpolation='none', extent=(x[0], x[-1], y[0], y[-1]))
        plt.colorbar(img, ax=ax[3])
        ax[3].set_title("Difference Image")
        plt.show()


'''
        

    def _interp2d(self, x, y):
        """
        These points are inside the grid, just do regular bilinear interpolation.

        :param x: N-element array, the x-coordinate of the point to interpolate.
        :param y: N-element array, the y-coordinate of the point to interpolate.
        :return: N-element array, the interpolated values at the (x, y) locations.
        """
        print("Interior-interpolating %i points." % len(x))
        #import ipdb; ipdb.set_trace()

        # Find the indices of the grid points surrounding (x, y)
        i0 = np.sum(x.reshape(-1,1) >= self.gx.reshape(1, -1), axis=1) - 1
        # Since image rows are opposite cartesian y-coordinates, we need to reverse the y-coordinates.
        j0 = np.sum(y.reshape(-1,1) >= self.gy.reshape(1, -1), axis=1) - 1  
        #j0 = self.v.shape[0] - 2 - j0 # reverse the y-coordinates
        #print(y,j0)
        
        i1 = i0 + 1
        j1 = j0 + 1

        # Get the coordinates of the grid points
        x0 = self.gx[i0]
        y0 = self.gy[j0]
        #print(x0, y0,"\n")
        x1 = self.gx[i1]
        y1 = self.gy[j0]

        # Get the values at the grid pointsq
        v00 = self.v[j0, i0]
        v01 = self.v[j0, i1]
        v10 = self.v[j1, i0]
        v11 = self.v[j1, i1]

        # Perform bilinear interpolation
        alpha = (x - x0) * (y - y0) / self._dxdy  # area nearest v00, opposite v11
        beta = (x1 - x) * (y - y0) / self._dxdy,# # area nearest v01, opposite v10
        gamma = (x - x) * (y1 - y) / self._dxdy # area nearest v10, opposite v01
        delta = (x1 - x) * (y1 - y) / self._dxdy # area nearest v11,     opposite v00

        interp_values = (v01 * alpha + v00 * beta + v11 * gamma + v10 * delta).reshape(-1)
        #interp_values = (v00 * alpha + v10 * beta + v01 * gamma + v11 * delta).reshape(-1)
        
        return interp_values
    
    def interp2d(self, x, y):
        """
        Interpolate the values at the given (x, y) coordinates.

        :param x: N-element array, the x-coordinate of the point to interpolate.
        :param y: N-element array, the y-coordinate of the point to interpolate.
        :return: N-element array, the interpolated values at the (x, y) locations.
        """
        x,y= np.array(x).reshape(-1), np.array(y).reshape(-1)
        interp_values = np.zeros_like(x)
        
        # Check if the points are inside the grid
        outside_vertically = (y < self.gy[0]) | (y >= self.gy[-1])
        outside_horizontally = (x < self.gx[0]) | (x >= self.gx[-1])
        inside = ~(outside_vertically | outside_horizontally).reshape(-1)
        print("Interpolating %i points (%i inside, %i outside):" % (x.size, np.sum(inside), x.size - np.sum(inside)))

        # Handle points inside the grid:
        interp_values[inside] = self._interp2d(x[inside], y[inside])
        
        # Handle points outside the grid
        
        return interp_values



    #@jit(float64[:](float64[:], float64[:]))
    def _interp_free_horizontal(self, x, y):
        """
        Use the x-coordinate to determine if the first or last column of grid values are to be used.
        Use the y-coordinates to do linear interpolation between the two values.
        :param x: The x-coordinates of the point to interpolate.
        :param y: The y-coordinates of the point to interpolate.
        :return: The interpolated values at (x, y) positions.
        """
        # left = x < self._center[0]
        right = x >= self._center[0]
        y_values = self.values[:, 0]
        y_values[right] = self.values[right, -1]

        # now use y to interpolate the y-values (with constant boundary conditions)
        y0 = jnp.searchsorted(self.gy, y) - 1
        y1 = y0 + 1
        y0 = jnp.clip(y0, 0, self.v.shape[0] - 1)
        y1 = jnp.clip(y1, 0, self.v.shape[0] - 1)

        alpha = (y - self.gy[y0]) / self._dy
        alpha = jnp.clip(alpha, 0, 1)
        beta = 1 - alpha
        interp_values = beta * y_values[y0] + alpha * y_values[y1]
        return interp_values

    #@jit
    def _interp_free_vertical(self, x, y):
        """
        Use the y-coordinate to determine if the first or last row of grid values are to be used.
        Use the x-coordinates to do linear interpolation between the two values.
        :param x: The x-coordinates of the point to interpolate.
        :param y: The y-coordinates of the point to interpolate.
        :return: The interpolated values at (x, y) positions.
        """
        # top = y < self._center[1]
        bottom = y >= self._center[1]
        x_values = self.values[0, :]
        x_values[bottom] = self.values[-1, :]

        # now use x to interpolate the x-values (with constant boundary conditions)
        x0 = jnp.searchsorted(self.gx, x) - 1
        x1 = x0 + 1
        x0 = jnp.clip(x0, 0, self.v.shape[1] - 1)
        x1 = jnp.clip(x1, 0, self.v.shape[1] - 1)

        alpha = (x - self.gx[x0]) / self._dx
        alpha = jnp.clip(alpha, 0, 1)
        beta = 1 - alpha
        interp_values = beta * x_values[x0] + alpha * x_values[x1]
        return interp_values

    def _interp_free(self, x, y, orient='vertical'):
        """
        All of these points are outside the grid and are free-slip.
        Use the value at the closest point on the grid edge.
        :param x: The x-coordinate of the point to interpolate.
        :param y: The y-coordinate of the point to interpolate.
        :param orient: 'vertical' means the points lie above or below the grid,
            'horizontal' means the points lie to the left or right of the grid.
        """
        if orient == 'horizontal':
            return self._interp_free_horizontal(x, y)
        elif orient == 'vertical':
            return self._interp_free_vertical(x, y)


def test_interp_free(plot=True):

    # function to be interpolated
    extent = (0, 1, 0, 1)
    nx = 2
    ny = 3
    x = np.linspace(0, 1.0, nx)
    y = np.linspace(0, 1.0, ny)
    ta,tb = np.meshgrid(x,y)

    """
    x = np.array((0,1))
    y = np.array((0,1))
    z = np.array([[2.0, 2.0],
                  [2.0, 1.0]])
    
    interp_test = Interp2d(x, y, z, bounds_h_v=('free', 'free'))

    test_points = {(0.5,0.75): 1.5,
                    (0.25,0.25): 1.5,
                    (0.75,0.25): 2.0,
                    (0.75,0.75): 1.5,
                    (0.5,0): 2.0,
                    (0,0.5): 2.0,
                    (1,1): 1.0,
                    (1,0): 2.0,
                    (0,1): 2.0}
    for test_point, expected_value in test_points.items():
        x_test, y_test = test_point
        
        interp_value = interp_test.interp2d(x_test, y_test)
        print(f"Test point: {test_point}, Expected value: {expected_value}, Interpolated value: {interp_value}")
        #assert np.isclose(interp_value, expected_value), f"Interpolation failed for point {test_point}"
    """



    values = np.sin(ta) * np.cos(tb)+1
    control_test = RectBivariateSpline(x,y, values.T, kx=1, ky=1)

    print("X:", x)
    print("Y:", y)

    interp_test = Interp2d(x, y, values, bounds_h_v=('zero', 'free'))
    interp_testb = Interp2d(x, y, values, bounds_h_v=('free', 'zero'))

    import matplotlib.pyplot as plt

 

    # do the interpolation of a test point
    test_p =np.array([0.85,.85])
    plt.ion()
    fig, ax = plt.subplots()
    # draw grid for x,y coords
    for i in range(len(x)):
        ax.plot([x[i], x[i]], [y[0], y[-1]], 'k--', lw=0.5)
    for i in range(len(y)):
        ax.plot([x[0], x[-1]], [y[i], y[i]], 'k--', lw=0.5)
    ax.plot(test_p[0], test_p[1], 'ro', markersize=10)
    ax.axis('equal')    
    plt.draw()
    plt.pause(0.1)

    ctrl_val = control_test(test_p[0], test_p[1], grid=False)
    #import ipdb; ipdb.set_trace()
    interp_val = interp_test.interp2d(test_p[0], test_p[1])
    print("Control value: ", ctrl_val)
    print("Interp value: ", interp_val)
    print("Difference: ", interp_val - ctrl_val)

    if plot:
        # interpolate at these points:
        nx_test = nx*20
        ny_test = ny*20
        m=-.01
        test_extent = (-m, 1+m, -m, 1+m)
        xt, yt =np.linspace(test_extent[0], test_extent[1], nx_test), np.linspace(test_extent[2], test_extent[3], ny_test)
        print("X: ", xt)
        print("Y: ", yt)
        x_test, y_test = np.meshgrid(xt, yt)
        x_test = x_test.reshape(-1)
        y_test = y_test.reshape(-1)

        # interpolate the image and a bit of the margin outside to see the boundary conditions.
        interp_img = interp_test.interp2d(x_test, y_test).reshape(ny_test, nx_test)
        interp_imgb = interp_testb.interp2d(x_test, y_test).reshape(ny_test, nx_test)
        interp_img_control = control_test(x_test, y_test,grid=False).reshape(ny_test, nx_test)


        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
        ax = ax.flatten()
        #ax[0].imshow(values, cmap='jet', interpolation='none', extent=extent)
        #ax[1].imshow(interp_img, cmap='jet', interpolation='none', extent=test_extent)
        #ax[2].imshow(interp_img_control, cmap='jet', interpolation='none', extent=test_extent)
        plt.colorbar(ax[0].imshow(values, cmap='jet', interpolation='none', extent=extent), ax=ax[0])
        plt.colorbar(ax[1].imshow(interp_img_control, cmap='jet', interpolation='none', extent=test_extent), ax=ax[2])
        plt.colorbar(ax[2].imshow(interp_img, cmap='jet', interpolation='none', extent=test_extent), ax=ax[1])
        plt.colorbar(ax[3].imshow(interp_imgb, cmap='jet', interpolation='none', extent=test_extent), ax=ax[3])
        for ax_ind in range(4):
            ax[ax_ind].plot(test_p[0], test_p[1], 'ro', markersize=10)
            
        ax[0].set_title("Original")
        ax[2].set_title("Interp-Test (free-slip, no-slip)")
        ax[3].set_title("Interp-Test (no-slip, free-slip)")
        ax[1].set_title("Interp-SciPy (extrapolated bounds)")
        plt.show()
        plt.waitforbuttonpress()
'''
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_interp_free()
