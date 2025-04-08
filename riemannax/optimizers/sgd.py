"""Riemannian gradient descent optimization algorithm.

This module implements the Riemannian gradient descent (RGD) algorithm for
optimization on Riemannian manifolds.
"""

from .state import OptState


def riemannian_gradient_descent(learning_rate=0.1, use_retraction=False):
    """Riemannian gradient descent optimizer.

    Implements the Riemannian gradient descent algorithm, which moves along the
    negative Riemannian gradient direction and then projects back to the manifold
    using either the exponential map or retraction.

    Args:
        learning_rate: Step size for updates.
        use_retraction: Whether to use retraction instead of exponential map.

    Returns:
        A tuple (init_fn, update_fn) for initialization and updates.
    """

    def init_fn(x0):
        """Initialize optimizer state.

        Args:
            x0: Initial point on the manifold.

        Returns:
            Initial optimizer state.
        """
        return OptState(x=x0)

    def update_fn(gradient, state, manifold):
        """Update optimizer state using Riemannian gradient.

        Args:
            gradient: Riemannian gradient at current point.
            state: Current optimizer state.
            manifold: Manifold on which to optimize.

        Returns:
            Updated optimizer state.
        """
        x = state.x

        # Compute descent direction
        v = -learning_rate * gradient

        # Move along manifold
        new_x = manifold.retr(x, v) if use_retraction else manifold.exp(x, v)

        return OptState(x=new_x)

    return init_fn, update_fn
