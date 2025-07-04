"""Riemannian momentum optimization algorithm.

This module implements Riemannian gradient descent with momentum, which
maintains a momentum term that is transported along the manifold using
parallel transport.
"""

from typing import Any

import jax.numpy as jnp
from jax import tree_util

from .state import OptState


class MomentumState(OptState):
    """Momentum optimizer state for Riemannian optimization.

    Extends OptState to include a momentum term that is transported
    along the manifold.

    Attributes:
        x: Current point on the manifold.
        momentum: Momentum term in the tangent space.
    """

    def __init__(self, x, momentum=None):
        """Initialize momentum state.

        Args:
            x: Current point on the manifold.
            momentum: Momentum term. If None, initialized to zeros.
        """
        super().__init__(x)
        self.momentum = jnp.zeros_like(x) if momentum is None else momentum

    def tree_flatten(self):
        """Flatten the MomentumState for JAX."""
        children = (self.x, self.momentum)
        aux_data: dict[str, Any] = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten the MomentumState for JAX."""
        return cls(x=children[0], momentum=children[1])


# Register the MomentumState class as a PyTree node
tree_util.register_pytree_node_class(MomentumState)


def riemannian_momentum(
    learning_rate=0.1,
    momentum=0.9,
    use_retraction=False
):
    """Riemannian gradient descent with momentum.

    Implements Riemannian gradient descent with momentum, where the momentum
    term is maintained in the tangent space and transported using parallel
    transport when moving to new points on the manifold.

    The momentum helps accelerate convergence and can help escape local minima.

    Args:
        learning_rate: Step size for updates.
        momentum: Momentum coefficient (typically between 0 and 1).
        use_retraction: Whether to use retraction instead of exponential map.

    Returns:
        A tuple (init_fn, update_fn) for initialization and updates.

    References:
        Ring, W., & Wirth, B. (2012). Optimization methods on Riemannian manifolds
        and their application to shape space. SIAM Journal on Optimization.
    """

    def init_fn(x0):
        """Initialize momentum optimizer state.

        Args:
            x0: Initial point on the manifold.

        Returns:
            Initial momentum state with zero momentum.
        """
        return MomentumState(x=x0)

    def update_fn(gradient, state, manifold):
        """Update momentum state using Riemannian gradient.

        Args:
            gradient: Riemannian gradient at current point.
            state: Current momentum state.
            manifold: Manifold on which to optimize.

        Returns:
            Updated momentum state.
        """
        x = state.x
        m = state.momentum

        # Update momentum term
        m_new = momentum * m - learning_rate * gradient

        # Move along manifold using momentum
        x_new = manifold.retr(x, m_new) if use_retraction else manifold.exp(x, m_new)

        # Transport momentum to new point
        m_transported = manifold.transp(x, x_new, m_new)

        return MomentumState(x=x_new, momentum=m_transported)

    return init_fn, update_fn
