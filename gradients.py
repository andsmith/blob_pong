
#import jax.numpy as np
import numpy as np
#from jax import grad, jit, vmap
import matplotlib.pyplot as plt


#@jit
def gradient_central(values, spacing):
    """
    Get the spatial gradient of a function defined on an evenly spaced grid.
    Use the central difference for the interior points, and forward/backward difference for the edges.
    # TODO: Higher order
    :param values: M x N array of function values.
    :param spacing: The spacing between grid points.  The function is defined over 
        x=[0, (N-1)*spacing], and
        y=[0, (M-1)*spacing].
    :returns: Tuple with the x and y spatial first derivatives (same shape arrays):
      (d/dx: M x N,
       d/dy: M x N)
    """
    inner_dx = (values[:, 2:] - values[:, :-2]) / (2*spacing)
    inner_dy = (values[2:, :] - values[:-2, :]) / (2*spacing)
    # Edge cases:

    left_dx = (values[:, 1] - values[:, 0]) / spacing
    right_dx = (values[:, -1] - values[:, -2]) / spacing
    top_dy = (values[1, :] - values[0, :]) / spacing
    bottom_dy = (values[-1, :] - values[-2, :]) / spacing

    # Combine the results:
    dx = np.concatenate((left_dx[:, None], inner_dx, right_dx[:, None]), axis=1)
    dy = np.concatenate((top_dy[None, :], inner_dy, bottom_dy[None, :]), axis=0)

    return dx, dy


#@jit
def gradient_upwind(values, spacing):
    """
    Just use the upwind scheme for all but the last edges (which are downwind/copied)
    """
    inner_dx = (values[:, 1:] - values[:, :-1]) / spacing
    inner_dy = (values[1:, :] - values[:-1, :]) / spacing
    # Edge cases:
    right_dx = inner_dx[:, -1]  # Copy the last value
    bottom_dy = inner_dy[-1, :]

    # Combine the results:
    dx = np.concatenate((inner_dx, right_dx[:, None]), axis=1)
    dy = np.concatenate((inner_dy, bottom_dy[None, :]), axis=0)

    return dx, dy


def test_gradient(res=100):
    """
    Test the gradient function by plotting the gradient of a simple function.
    :param res: The resolution of the grid.
    """
    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-X**2 - Y**2)

    # import ipdb; ipdb.set_trace()

    dx, dy = gradient_upwind(Z, x[1]-x[0])

    _, axs = plt.subplots(1, 3, figsize=(15, 5))
    img = axs[0].imshow(Z, extent=(0, 2*np.pi, 0, 2*np.pi), origin='lower')
    axs[0].set_title('Function')
    plt.colorbar(img, ax=axs[0])

    img = axs[1].imshow(dx, extent=(0, 2*np.pi, 0, 2*np.pi), origin='lower')
    axs[1].set_title('Gradient in x')
    plt.colorbar(img, ax=axs[1])

    img = axs[2].imshow(dy, extent=(0, 2*np.pi, 0, 2*np.pi), origin='lower')
    axs[2].set_title('Gradient in y')
    plt.colorbar(img, ax=axs[2])

    plt.show()


if __name__ == "__main__":
    # Test the gradient function
    test_gradient()
