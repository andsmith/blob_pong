"""
Class to represent solid objects.  The set of objects has several representations:

  - Collection of polygons (for now rectangles).
  - Position field (level set): A signed distance function defined on a 2d grid.  The values are defined as, at every interior point, the
      negative distance to the nearest point on the object boundary, exterior is positive distance.
  - Velocity field:  defined constant in the voronoi region around an object, set to that object's velocity. 

"""
import numpy as np
import skfmm  # for the level set functions
import cv2  # for rendering
from abc import ABC, abstractmethod
from colors import LINE_COLOR, BKG_COLOR, FLUID_COLOR, OBJECT_COLORS
import matplotlib.pyplot as plt


def rgb_int_to_float(rgb_int):
    """
    Convert an RGB integer to a float in the range [0, 1].
    :param rgb_int: RGB integer value.
    :return: RGB float value.
    """
    return (rgb_int[0] / 255.0, rgb_int[1] / 255.0, rgb_int[2] / 255.0)


class Solid(ABC):
    """
    Base class for a single solid/rigid object.
    """

    def __init__(self, points, origin, velocity, color, inverse=False):
        """
        points: list of points defining the solid in the local coordinate system.
        origin: position of the solid in the global coordinate system.
        velocity: velocity of the solid in the global coordinate system.
        color: color of the solid for rendering.
        """
        self.points = np.array(points).reshape((-1, 2))  # convert to 2D array of points
        self.origin = np.array(origin).reshape((-1, 2))  # convert to 2D array of points
        self.velocity = velocity
        self.inverse = inverse
        self.color = color
        self._mpl_color = rgb_int_to_float(color)  # convert to float for matplotlib
        self._artists = {}

    def render_border(self, grid_shape, dx):
        """
        Render the border of the solid as a binary image (0/1) in the grid defined by x_pts and y_pts.
        (Typically, the cell centerpoints)

        Return an image that is 1 everywhere except the border, which is 0.

        :param grid_shape: shape of the grid (width, height).
        :param dx: size of a pixel in the real world (in meters).
        :return: binary image of the solid (size h x w).   All cells through which the border passes are 0.
        """
        # import ipdb; ipdb.set_trace()  # breakpoint 1

        PREC_BITS = 4
        PREC_MUL = 2 ** PREC_BITS  # precision multiplier
        # Scale points up to the grid size (pixel/integers coords):
        pts = ((self.points+self.origin) / dx - .5) * PREC_MUL  # scale points to pixel coordinates
        pts = pts.astype(np.int32)
        # Create a mask for the solid:
        mask = np.zeros(grid_shape, dtype=np.uint8)
        cv2.polylines(mask, [pts], isClosed=True, color=(1, 1, 1), thickness=1, shift=PREC_BITS)
        return mask.astype(np.float32)  # convert to float for distance function

    def plot(self, ax):
        """
        Plot the solid on the given axes.
        :param ax : axes to plot on.
        """
        # Plot the solid:
        pts = self.points + self.origin
        pts = pts.reshape((-1, 1, 2))
        last_pt = pts[0, 0, :].reshape(1, 1, 2)

        pts = np.concatenate((pts, last_pt), axis=0)  # close the polygon
        if self._artists.get('solid') is None:
            self._artists['border'] = ax.plot(pts[:, 0, 0], pts[:, 0, 1], color=self._mpl_color, linewidth=1)
            self._artists['solid'] = ax.fill(pts[:, 0, 0], pts[:, 0, 1], color=self._mpl_color, alpha=0.5)

        else:
            self._artists['border'].set_data(pts[:, 0, 0], pts[:, 0, 1])
            self._artists['solid'].set_xy(pts[:, 0, :])

    def is_inside(self, points):
        """
        winding number test.
        """
        line_points = np.concatenate((self.points, self.points[0:1]), axis=0)  # close the polygon

        w_num = np.zeros(points.shape[0], dtype=np.int32)  # winding number for each point
        test_points = points - self.origin  # shift points to the origin
        for line_ind in range(len(self.points)):

            line_start = line_points[line_ind, :]
            line_end = line_points[line_ind + 1, :]

            # Compute the cross product with this segment & a line from the start of the segment to the test point:
            cross_prod = np.cross(line_end - line_start, test_points - line_start)
            left_edge_pts = (cross_prod > 0) & (line_start[1] < test_points[:, 1]) & (
                line_end[1] > test_points[:, 1])
            right_edge_pts = (cross_prod < 0) & (line_start[1] > test_points[:, 1]) & (
                line_end[1] < test_points[:, 1])
            wn_num_inc = left_edge_pts | right_edge_pts  # increment winding number
            w_num[wn_num_inc] += 1

        return np.mod(w_num, 2) == 1  # odd winding number means inside


def plot_grid(ax, n_cells, dx, line_width=1):
    """
    Plot a grid on the given axes 
    :param ax: axes to plot on.
    :param n_cells: number of cells in the grid (width, height).
    :param dx: size of a pixel in the real world (in meters).
    :param line_width: width of the grid lines.
    """
    # Plot the grid:
    x_pts = np.linspace(0, n_cells[0] * dx, n_cells[0] + 1)
    y_pts = np.linspace(0, n_cells[1] * dx, n_cells[1] + 1)
    for x in x_pts:
        ax.axvline(x=x, color=LINE_COLOR, linewidth=line_width)
    for y in y_pts:
        ax.axhline(y=y, color=LINE_COLOR, linewidth=line_width)

    # Set the axis limits:
    ax.set_xlim(0, n_cells[0] * dx)
    ax.set_ylim(0, n_cells[1] * dx)


def test_solid():
    """
    Test the Solid class.
    """
    n_cells = 80

    def plot_test(ax):
        # Create a solid:
        points = np.random.rand(3*2).reshape(-1, 2)  # np.array([[.4, 0], [1, .14], [.3, 1], [0, 1]]) *.8
        points = points.reshape((-1, 1, 2))
        origin = (.1, .1)
        solid = Solid(points, origin, velocity=(0, 0), color=OBJECT_COLORS[0])

        # Create a grid:
        grid_size = (n_cells, n_cells)
        dx = 1/n_cells  # size of a pixel in the real world (in meters).

        # Render the solid:
        img = 1.0-solid.render_border(grid_size, dx)
        # randomize image:
        # img = np.random.rand(*img.shape) * img

        # Plot the solid:
        extent = (0, grid_size[0] * dx, 0, grid_size[1] * dx)  # extent of the image in the real world
        ax.imshow(img, cmap='gray', origin='lower', extent=extent)
        plot_grid(ax, grid_size, dx, line_width=0.5)
        solid.plot(ax)

        n_test = 0
        if n_test>0:
            test_x, test_y = np.meshgrid(np.linspace(-.5, 1.5, n_test), np.linspace(-.5, 1.5, n_test))
            test_points = np.stack((test_x, test_y), axis=-1).reshape((-1,  2))
            in_state = solid.is_inside(test_points)

            # Plot the points:

            ax.scatter(test_points[in_state, 0], test_points[in_state, 1], color='red', s=1)
            ax.scatter(test_points[~in_state, 0], test_points[~in_state, 1], color='blue', s=1)

        ax.set_xlim(0,1)
        ax.set_ylim(0, 1)
        
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.set_facecolor(rgb_int_to_float(BKG_COLOR))
        ax.set_aspect('equal')
        plot_test(ax)

    plt.show()


class ObjectSet(object):
    """
    Represent all the objects in the simulation.
    Interpolate as necessary from one frame to the next as objects positions are updated.
    """

    def __init__(self, grid_size, dx):
        """
        The set of objects is represented by:
          * the grid of signed distances (to nearest object boundary)
          * the grid of velocities extended to objects' voronoi regions
          * the grid of regions (which object is closest to each point in the grid)

        grid_size: size of the grid in pixels (width, height).
        dx: size of a pixel in the real world (in meters).
        """
        self.n_cells = grid_size
        self.dx = dx
        self._init_grids()
        self._init_objects()
        self._dists_blank = self._dists.copy()  # Add objects by modifying this blank grid.
        self._refresh_dists()

        self._artists = {}

    def _init_objects(self):
        """
        Init list of objects.
        Set walls as an object.
        Get signed distance function for walls, to be modified for each frame.
        """
        self._solids = []

        # add walls as an "inverse" solid.
        grid_ex = (self.n_cells[0] - 1) * self.dx, (self.n_cells[1] - 1) * self.dx
        walls_pts = np.array([[0, 0], [grid_ex[0], 0], [grid_ex[0], grid_ex[1]], [0, grid_ex[1]]])
        walls_pts = walls_pts.reshape((-1, 1, 2))
        self._walls = Solid(walls_pts, (0, 0), (0, 0), (255, 255, 255), inverse=True)

        border = self._walls.render_border(self._x_pts, self._y_pts)
        s_dist = skfmm.distance(border, dx=self.dx)  # signed distance function for the walls
        self._dists = -s_dist

    def _init_grids(self):
        # Signed distances:  self._dist[]
        self._x_pts = np.linspace(0, (self.n_cells[0]-1) * self.dx, self.n_cells[0])
        self._y_pts = np.linspace(0, (self.n_cells[1]-1) * self.dx, self.n_cells[1])

        # Extended velocities:
        self._vel_x = np.zeros((self.n_cells[1], self.n_cells[0]), dtype=np.float32)
        self._vel_y = np.zeros((self.n_cells[1], self.n_cells[0]), dtype=np.float32)

        # Regions:
        self._regions = np.zeros((self.n_cells[1], self.n_cells[0]), dtype=np.int32)

    def _refresh_dists(self):
        """
        Recompute the signed distance function and the regions for all solids.
        """
        dists = self._dists_blank.copy()  # walls adistances
        self._regions.fill(-1)  # wall index
        self._vel_x.fill(0)  # no motion in the walls
        self._vel_y.fill(0)

        # Compute the signed distance function and regions for each solid:
        for i, solid in enumerate(self._solids):
            # Compute the signed distance function for this solid:
            border = solid.render_border(self._x_pts, self._y_pts)
            s_dist = skfmm.distance(border, dx=self.dx, order=2)
            obj_mask = s_dist < dists  # mask for region of influence of this solid
            self._regions[obj_mask] = i
            dists[obj_mask] = s_dist[obj_mask]

            # Set the velocity field for this solid:
            vel_x, vel_y = solid.velocity
            self._vel_x[obj_mask] = vel_x
            self._vel_y[obj_mask] = vel_y

    def add_solid(self, solid):
        """
        Add a solid to the set of solids.
        solid: Solid object to add.
        """
        self._solids.append(solid)
        self._refresh_dists()

    def plot_state(self, axes):
        """
        Three plots side by side:
        1. Signed distance function (level set) for the solids. (color mapped)
        2. Region map: which solid is closest to each point in the grid. (color coded)
        3. Velocity field: velocity of the solids in the grid. (vector field)
        :param axes: axes to plot on (list of 3).
        """
        # Plot the signed distance function:
        if self._artists['dist'] is None:
            self._artists['dist'] = axes[0].imshow(self._dists, cmap='jet', origin='lower')
            axes[0].set_title('Signed distance function')

        else:
            self._artists['dist'].set_array(self._dists)

        # Plot the region map:
        if self._artists['regions'] is None:
            self._artists['regions'] = axes[1].imshow(self._regions, cmap='jet', origin='lower')
            axes[1].set_title('Region map')
        else:
            self._artists['regions'].set_array(self._regions)

        # Plot the velocity field:
        if self._artists['vel'] is None:
            axes[2].quiver(self._x_pts, self._y_pts, self._vel_x, self._vel_y, color='black')
            axes[2].set_title('Velocity field')
        else:
            self._artists['vel'].set_UVC(self._vel_x, self._vel_y)

        # Plot the solids:
        for solid in self._solids:
            solid.plot(axes[0], axes[1], axes[2])

        # Plot the walls:
        self._walls.plot(axes[0], axes[1], axes[2])

        # Set the axis limits:
        axes[0].set_xlim(0, self.n_cells[0] * self.dx)
        axes[0].set_ylim(0, self.n_cells[1] * self.dx)


def test_solids():
    """
    Test the solids class.
    """
    # Create a set of solids:
    grid_size = (100, 100)
    dx = 0.01
    solids = ObjectSet(grid_size, dx)

    # Create a rect in the middle:
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * .1534
    points = points.reshape((-1, 1, 2))
    origin = (.4, .6)
    solid = Solid(points, origin, velocity=(0, 0), color=OBJECT_COLORS[0])

    # Add the solid to the set of solids:
    solids.add_solid(solid)

    # Plot the solids:
    fig, axes = plt.subplots(1, 3)
    solids.plot_state(axes)
    plt.show()
    # Test the signed distance function:


if __name__ == '__main__':
    test_solid()
