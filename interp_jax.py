import jax.numpy as jnp
from jax import jit, vmap

class Interp2d:
    def __init__(self, p0, dp, size=None, value=0.):
        self.p_min = jnp.array(p0)
        self.dp = float(dp)
        if size is None:
            if value is None:
                raise ValueError("Shape must be provided if values are not.")
            else:
                self.size = value.shape[1], value.shape[0]
                self.values = jnp.array(value)
        else:
            self.size = size
            self.values = jnp.zeros((size[1], size[0]), dtype=jnp.float32) + value
        self.p_max = self.p_min + (jnp.array(self.size) - 1) * self.dp

    @staticmethod
    @jit
    def _interp_jit(points, values, p_min, dp, size):
        points = jnp.array(points).reshape(-1, 2)
        x = points[:, 0]
        y = points[:, 1]

        # Compute indices
        i0 = jnp.floor((x - p_min[0]) / dp).astype(int)
        j0 = jnp.floor((y - p_min[1]) / dp).astype(int)
        i1 = i0 + 1
        j1 = j0 + 1

        # Clip indices to grid bounds
        i0 = jnp.clip(i0, 0, size[0] - 1)
        j0 = jnp.clip(j0, 0, size[1] - 1)
        i1 = jnp.clip(i1, 0, size[0] - 1)
        j1 = jnp.clip(j1, 0, size[1] - 1)

        # Compute relative offsets
        t_x = (x - (p_min[0] + i0 * dp)) / dp
        t_y = (y - (p_min[1] + j0 * dp)) / dp

        # Get values at grid points
        v00 = values[j0, i0]
        v01 = values[j1, i0]
        v10 = values[j0, i1]
        v11 = values[j1, i1]

        # Perform bilinear interpolation
        interp_values = (
            v11 * t_x * t_y +
            v10 * t_x * (1 - t_y) +
            v01 * (1 - t_x) * t_y +
            v00 * (1 - t_x) * (1 - t_y)
        )
        return interp_values

    def interp(self, points):
        return self._interp_jit(points, self.values, self.p_min, self.dp, self.size)