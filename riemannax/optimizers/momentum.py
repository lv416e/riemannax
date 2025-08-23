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


def riemannian_momentum(learning_rate=0.1, momentum=0.9, use_retraction=False):
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

        # Update momentum term (combine old momentum with new gradient)
        m_new = momentum * m + gradient

        # Ensure momentum is in tangent space
        m_new = manifold.proj(x, m_new)

        # Compute the step direction
        step_direction = -learning_rate * m_new

        # Clip step size for numerical stability
        step_norm = jnp.linalg.norm(step_direction)
        max_step = 0.5  # Maximum step size
        step_direction = jnp.where(
            step_norm > max_step, step_direction * (max_step / (step_norm + 1e-8)), step_direction
        )

        # Move along manifold using the step direction
        try:
            x_new = manifold.retr(x, step_direction) if use_retraction else manifold.exp(x, step_direction)
        except Exception:
            # Fallback to retraction if exponential map fails
            x_new = manifold.retr(x, step_direction)

        # Ensure the new point is on the manifold (only for sphere-like manifolds)
        if hasattr(manifold, "proj") and hasattr(manifold, "__class__") and "Sphere" in manifold.__class__.__name__:
            x_new = manifold.proj(x_new, jnp.zeros_like(x_new))  # Project to manifold

        # Transport momentum to new point
        m_transported = manifold.transp(x, x_new, m_new)

        # Ensure transported momentum is in tangent space
        m_transported = manifold.proj(x_new, m_transported)

        return MomentumState(x=x_new, momentum=m_transported)

    return init_fn, update_fn
