import numpy as np
from abc import ABC, abstractmethod
import logging
import matplotlib.pyplot as plt
from fields import CenterScalarField
from semi_lagrange import advect
from loop_timing.loop_profiler import LoopPerfTimer as LPT
import cv2
import skfmm


class FluidField(CenterScalarField, ABC):
    """
    Base class for fluid.
        subclasses:  Gasses will be density, liquids will be signed distance
    """

    def __init__(self, size_m, grid_size, values=None):
        super().__init__(size_m, grid_size, values=values, name=self.__class__.__name__)
        self._frame_no = 0


class SmokeField(FluidField):
    def __init__(self, size_m, grid_size):
        """
        Initialize the smoke field.
        self.values will be the density of the smoke.
        :param size_m: The size of the simulation domain in meters (width, height).
        :param grid_size: The number of cells in the x and y direction for the smoke field.
        """
        super().__init__(size_m, grid_size)

    def _get_cell_centers(self):
        points_x, points_y = np.meshgrid(self.centers_x, self.centers_y)
        points = np.stack((points_x, points_y), axis=-1)
        return points

    def add_circle(self, center, radius, density):
        """
        Set all points in a specified circle to a density value.
        :param center: The center of the circle in world coordinates.
        :param radius: The radius of the circle in world coordinates.
        :param density: The density value to set.
        """
        center = np.array(center).reshape(1, 1, 2) * self.size  # convert to grid coordinates
        points = self._get_cell_centers()
        distances = np.linalg.norm(points - center, axis=2)
        inside = np.where(distances < radius*self.size[0])
        circle=np.zeros(self.values.shape, dtype=self.values.dtype)
        circle[inside]=density
        self.values+=circle
        self.finalize()
    def add_square(self, center, size, density):    
        """
        Set all points in a specified square to a density value.
        :param center: The center of the square in world coordinates.
        :param size: The size of the square in world coordinates.
        :param density: The density value to set.
        """
        center = np.array(center).reshape(1, 1, 2) * self.size
        points = self._get_cell_centers()
        distances = np.abs(points - center)
        inside = np.where(np.all(distances < size*self.size[0]/2, axis=2))
        square = np.zeros(self.values.shape, dtype=self.values.dtype)
        square[inside] = density
        self.values += square
        self.finalize()
        
    def plot(self, ax, res=1000, alpha=0.8, **kwargs):
        return super().plot(ax, res=res, alpha=alpha, title="Smoke density", **kwargs)

    @LPT.time_function
    def advect(self, velocity, dt, C=2.0):  # _lagrange
        """
        For each grid point, find the new velocity by moving the point backwards through the 
        velocity field for a time step dt. Then interpolate the density at that position.
        """
        self._frame_no += 1
        
        points = self._get_cell_centers()
        points_moved = advect(points, velocity, dt, self.dx, self.size, C=C)

        old_values = self.interp_at(points_moved)

        self.values = old_values
        self.finalize()

    @LPT.time_function
    def render(self, img, d_max, bbox, fl_color, bkg_color):
        """
        Render the smoke field onto an image.
        :param d_max: The maximum density value for normalization.
        :param img: The image to render onto.
        :param bbox: The bounding box for the rendering.
          {'x': (x_min, x_max), 'y': (y_min, y_max)}
        :param color: RGB triplet of maximum density color (interpolated between this and the image.)
        """

        # Normalize the density values
        normalized_vals = np.clip(self.values / d_max, 0, 1)

        # make a single-pixel image then scale it up and put it in the right place
        fl_color = np.array(fl_color, dtype=np.uint8).reshape(1, 1, 3)
        bkg_color = np.array(bkg_color, dtype=np.uint8).reshape(1, 1, 3)

        img_small = normalized_vals[...,np.newaxis] * fl_color + (1 - normalized_vals[...,np.newaxis]) * bkg_color  # blend with background color
        patch = cv2.resize(img_small, (bbox['x'][1] - bbox['x'][0], bbox['y'][1] - bbox['y'][0]), interpolation=cv2.INTER_LINEAR)
        img[bbox['y'][0]:bbox['y'][1], bbox['x'][0]:bbox['x'][1]] = patch

    @LPT.time_function
    def render_additive(self, img, d_max, bbox, fl_color):
        """
        Additively blend this smoke field on top of the existing image.
        Wherever density is nonzero, tint the pixel toward fl_color.
        """
        if d_max <= 0:
            return
        normalized_vals = np.clip(self.values / d_max, 0, 1)  # (H, W)
        h = bbox['y'][1] - bbox['y'][0]
        w = bbox['x'][1] - bbox['x'][0]
        alpha = cv2.resize(normalized_vals.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        region = img[bbox['y'][0]:bbox['y'][1], bbox['x'][0]:bbox['x'][1]].astype(np.float32)
        fl = np.array(fl_color, dtype=np.float32).reshape(1, 1, 3)
        blended = region * (1 - alpha[..., np.newaxis]) + fl * alpha[..., np.newaxis]
        img[bbox['y'][0]:bbox['y'][1], bbox['x'][0]:bbox['x'][1]] = np.clip(blended, 0, 255).astype(np.uint8)


class LiquidField(FluidField):
    """
    Free-surface liquid represented as a level-set (signed distance function φ).
      φ < 0  →  inside liquid
      φ > 0  →  outside liquid (air)
      φ = 0  →  free surface interface

    Lives at velocity-grid resolution so the pressure solver can directly index cells.
    """

    def __init__(self, size_m, grid_size):
        """
        Initialize with all-air (φ = +dx everywhere, slightly positive).
        :param size_m: domain size in meters (width, height).
        :param grid_size: (n_x, n_y) number of cells — should match velocity grid.
        """
        super().__init__(size_m, grid_size)
        self.values = np.ones((grid_size[1], grid_size[0]), dtype=np.float64) * self.dx

    def add_pool(self, center, radius):
        """
        Add a circular pool of liquid centered at `center` (in [0,1]² world coords)
        with world-space `radius`. Sets φ = signed distance to the circle boundary.
        :param center: (cx, cy) in world coordinates (metres).
        :param radius: radius in metres.
        """
        cx, cy = center[0] * self.size[0], center[1] * self.size[1]
        pts = self._get_cell_centers()          # (H, W, 2)
        dist_from_center = np.linalg.norm(pts - np.array([cx, cy]).reshape(1, 1, 2), axis=2)
        phi_new = dist_from_center - radius     # negative inside, positive outside
        # Take the element-wise min so multiple pools can be added
        self.values = np.minimum(self.values, phi_new)
        self.finalize()

    def _get_cell_centers(self):
        points_x, points_y = np.meshgrid(self.centers_x, self.centers_y)
        return np.stack((points_x, points_y), axis=-1)

    def is_liquid(self):
        """Return boolean mask at native resolution: True where φ < 0 (inside liquid)."""
        return self.values < 0

    def phi_at(self, n_cells):
        """Return φ resampled to the given (n_x, n_y) resolution as a float32 array."""
        if (n_cells[0], n_cells[1]) == (self.n_cells[0], self.n_cells[1]):
            return self.values.astype(np.float32)
        return cv2.resize(self.values.astype(np.float32),
                          (n_cells[0], n_cells[1]),
                          interpolation=cv2.INTER_LINEAR)

    def is_liquid_at(self, n_cells):
        """
        Return a liquid mask resampled to the given grid resolution (n_x, n_y).
        Used by the pressure solver and velocity extrapolation which work at velocity-grid resolution.
        Bilinear interpolation of φ before thresholding gives a smooth coarse mask.
        """
        return self.phi_at(n_cells) < 0

    @LPT.time_function
    def reinitialize(self):
        """
        Restore the signed-distance property of φ using the Fast Marching Method.
        Called after each advection step.
        """
        try:
            phi_new = skfmm.distance(self.values, dx=float(self.dx))
            if isinstance(phi_new, np.ma.MaskedArray):
                # masked entries are far from the interface — keep old values there
                self.values[~phi_new.mask] = phi_new[~phi_new.mask]
            else:
                self.values = phi_new
        except Exception as e:
            logging.warning(f"LiquidField.reinitialize skfmm failed: {e}")
        self.finalize()

    @LPT.time_function
    def advect(self, velocity, dt, C=2.0):
        """
        Advect the level-set φ with the given velocity field, then reinitialize.
        """
        self._frame_no += 1
        points = self._get_cell_centers()
        points_moved = advect(points, velocity, dt, self.dx, self.size, C=C)
        self.values = np.asarray(self.interp_at(points_moved))  # ensure plain numpy (interp returns JAX array)
        self.finalize()
        self.reinitialize()

    def plot_isocontour(self, ax, res=200, color='#39ff14', linewidth=2):
        """Draw the φ=0 free-surface iso-contour on an existing axes."""
        x = np.linspace(0, self.size[0], res)
        y = np.linspace(0, self.size[1], res)
        xx, yy = np.meshgrid(x, y)
        phi_grid = np.asarray(
            self.interp_at(np.stack([xx.ravel(), yy.ravel()], axis=-1))
        ).reshape(res, res)
        ax.contour(xx, yy, phi_grid, levels=[0.0], colors=color, linewidths=linewidth)

    def plot(self, ax, res=200, alpha=0.85, **kwargs):
        """
        Matplotlib plot of the level-set field.
        Cold (blue) inside liquid (φ < 0), warm (red) outside (φ > 0),
        with a solid white iso-contour at φ = 0.
        """
        import matplotlib.colors as mcolors

        x = np.linspace(0, self.size[0], res)
        y = np.linspace(0, self.size[1], res)
        xx, yy = np.meshgrid(x, y)
        phi_grid = self.interp_at(np.stack([xx.ravel(), yy.ravel()], axis=-1))
        phi_grid = np.asarray(phi_grid).reshape(res, res)

        abs_max = max(abs(float(phi_grid.min())), abs(float(phi_grid.max())), 1e-6)
        norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

        img = ax.contourf(xx, yy, phi_grid, levels=64, cmap='RdBu_r', norm=norm, alpha=alpha)
        ax.contour(xx, yy, phi_grid, levels=[0.0], colors='white', linewidths=2)

        # domain boundary
        for (x0, x1), (y0, y1) in [((0, self.size[0]), (0, 0)),
                                     ((0, self.size[0]), (self.size[1], self.size[1])),
                                     ((0, 0), (0, self.size[1])),
                                     ((self.size[0], self.size[0]), (0, self.size[1]))]:
            ax.plot([x0, x1], [y0, y1], color='black', lw=2)

        liquid_vol = np.sum(self.values < 0) * self.dx ** 2
        ax.set_title(f"Level set φ  (white = interface)\nliquid area = {liquid_vol:.4f} m²,"
                     f"  φ ∈ [{self.values.min():.3f}, {self.values.max():.3f}]")
        return img

    @LPT.time_function
    def render(self, img, bbox, fl_color, bkg_color):
        """
        Render the liquid: fill liquid cells with fl_color, air cells with bkg_color,
        and draw a bright contour at the free surface.
        :param img: uint8 BGR image to draw onto.
        :param bbox: {'x': (x_min, x_max), 'y': (y_min, y_max)}.
        :param fl_color: RGB triplet for liquid.
        :param bkg_color: RGB triplet for air/background.
        """
        h = bbox['y'][1] - bbox['y'][0]
        w = bbox['x'][1] - bbox['x'][0]

        # Upscale the continuous φ field with bilinear interpolation, then threshold.
        # This gives smooth sub-cell edges rather than block-sized steps.
        phi_up = cv2.resize(self.values.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        patch_liquid = (phi_up < 0).astype(np.float32)

        fl = np.array(fl_color, dtype=np.uint8).reshape(1, 1, 3)
        bkg = np.array(bkg_color, dtype=np.uint8).reshape(1, 1, 3)

        canvas = np.where(patch_liquid[..., np.newaxis] > 0.5, fl, bkg).astype(np.uint8)

        # Draw free-surface contour on the upscaled mask
        mask_u8 = (patch_liquid * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        surface_color = tuple(int(c) for c in fl_color)
        cv2.drawContours(canvas, contours, -1, surface_color, 2)

        img[bbox['y'][0]:bbox['y'][1], bbox['x'][0]:bbox['x'][1]] = canvas[:, :, ::-1]  # RGB→BGR


def test_smoke():
    # Matplotlib display
    smoke = SmokeField((1., 1.), (150, 150))
    smoke.add_circle((0.5, 0.5), 0.16, 10.0)
    fig, ax = plt.subplots()
    smoke.plot(ax, res=300, alpha=0.8)
    plt.show()

    # OpenCV render
    img = np.zeros((900,900, 3), dtype=np.uint8)
    bbox = {'x': (0, 900), 'y': (0, 900)}
    bkg_color = 246, 238, 227  # Background
    fluid_color = 0, 4, 51
    smoke.render(img, 10.0, bbox, fluid_color, bkg_color)
    cv2.imshow("Smoke", img[:,:, ::-1])
    cv2.waitKey(0)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_smoke()
