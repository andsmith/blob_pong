
import jax.numpy as np
#import numpy as np
from jax import grad, jit, vmap
import matplotlib.pyplot as plt



@jit
def gradient_upwind(values, spacing, flip_y=True):
    """
    :param values: 2D array of values to compute the gradient for.
    :param spacing: The spacing between grid points.
    :param flip_y: If True, flip the y-axis for the gradient calculation (y-downwind)
    """
    dx = (values[:, 1:] - values[:, :-1]) / spacing
    dy = (values[1:, :] - values[:-1, :]) / spacing 
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
