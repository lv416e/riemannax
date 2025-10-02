"""Riemannian Adam optimization algorithm.

This module implements the Riemannian Adam algorithm, which adapts the popular
Adam optimizer to Riemannian manifolds using parallel transport for momentum
and second moment estimation.
"""

from typing import Any

import jax.numpy as jnp
from jax import tree_util

from .state import OptState


class AdamState(OptState):
    """Adam optimizer state for Riemannian optimization.

    Extends OptState to include momentum and second moment estimates
    that are transported along the manifold.

    Attributes:
        x: Current point on the manifold.
        m: First moment estimate (momentum).
        v: Second moment estimate (adaptive learning rate).
        step: Current step number.
    """

    def __init__(self, x: Any, m: Any = None, v: Any = None, step: int = 0) -> None:
        """Initialize Adam state.

        Args:
            x: Current point on the manifold.
            m: First moment estimate. If None, initialized to zeros.
            v: Second moment estimate. If None, initialized to zeros.
            step: Current step number.
        """
        super().__init__(x)
        self.m = jnp.zeros_like(x) if m is None else m
        self.v = jnp.zeros_like(x) if v is None else v
        self.step = step

    def tree_flatten(self) -> tuple[tuple[Any, Any, Any, int], dict[str, Any]]:
        """Flatten the AdamState for JAX."""
        children = (self.x, self.m, self.v, self.step)
        aux_data: dict[str, Any] = {}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: dict[str, Any], children: tuple[Any, Any, Any, int]) -> "AdamState":
        """Unflatten the AdamState for JAX."""
        return cls(x=children[0], m=children[1], v=children[2], step=children[3])


# Register the AdamState class as a PyTree node
tree_util.register_pytree_node_class(AdamState)


def riemannian_adam(
    learning_rate: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    use_retraction: bool = False,
) -> tuple[Any, Any]:
    """Riemannian Adam optimizer.

    Implements the Riemannian Adam algorithm, which adapts the Adam optimizer
    to Riemannian manifolds by using parallel transport to move momentum and
    second moment estimates along the manifold.

    The algorithm maintains first and second moment estimates in the tangent space
    and transports them using parallel transport when moving to new points.

    Args:
        learning_rate: Step size for updates.
        beta1: Exponential decay rate for first moment estimates.
        beta2: Exponential decay rate for second moment estimates.
        eps: Small constant for numerical stability.
        use_retraction: Whether to use retraction instead of exponential map.

    Returns:
        A tuple (init_fn, update_fn) for initialization and updates.

    References:
        Bécigneul, G., & Ganea, O. E. (2019). Riemannian adaptive optimization.
        International Conference on Learning Representations.
    """

    def init_fn(x0: Any) -> AdamState:
        """Initialize Adam optimizer state.

        Args:
            x0: Initial point on the manifold.

        Returns:
            Initial Adam state with zero momentum and second moments.
        """
        return AdamState(x=x0)

    def update_fn(gradient: Any, state: AdamState, manifold: Any) -> AdamState:
        """Update Adam state using Riemannian gradient.

        Args:
            gradient: Riemannian gradient at current point.
            state: Current Adam state.
            manifold: Manifold on which to optimize.

        Returns:
            Updated Adam state.
        """
        x = state.x
        m = state.m
        v = state.v
        step = state.step + 1

        # Update biased first moment estimate
        m_new = beta1 * m + (1 - beta1) * gradient

        # Update biased second raw moment estimate (element-wise)
        v_new = beta2 * v + (1 - beta2) * (gradient * gradient)

        # Compute bias-corrected first moment estimate
        m_hat = m_new / (1 - beta1**step)

        # Compute bias-corrected second raw moment estimate
        v_hat = v_new / (1 - beta2**step)

        # Compute Adam direction in tangent space using standard Adam formula
        # Following Kingma & Ba (2014) Algorithm 1: epsilon outside sqrt
        direction = m_hat / (jnp.sqrt(v_hat) + eps)

        # Scale by learning rate
        v = -learning_rate * direction

        # Clip the step size to prevent overshooting and NaN issues
        step_norm = jnp.linalg.norm(v)
        max_step = 0.1  # More conservative maximum step size
        v = jnp.where(step_norm > max_step, v * (max_step / (step_norm + 1e-8)), v)

        # Ensure the update is in the tangent space
        v = manifold.proj(x, v)

        # Move along manifold with numerical stability check
        try:
            x_new = manifold.retr(x, v) if use_retraction else manifold.exp(x, v)
        except Exception:
            # Fallback to retraction if exponential map fails
            x_new = manifold.retr(x, v)

        # Ensure the new point is on the manifold
        if hasattr(manifold, "__class__") and "Sphere" in manifold.__class__.__name__:
            # For sphere manifolds, normalize to ensure unit length
            x_norm = jnp.linalg.norm(x_new)
            x_new = jnp.where(x_norm > 1e-8, x_new / x_norm, x)  # Fallback to original if norm too small

        # Transport momentum estimates to new point
        m_transported = manifold.transp(x, x_new, m_new)
        v_transported = manifold.transp(x, x_new, v_new)

        # Ensure transported values are in tangent space
        m_transported = manifold.proj(x_new, m_transported)
        v_transported = manifold.proj(x_new, v_transported)

        return AdamState(x=x_new, m=m_transported, v=v_transported, step=step)

    return init_fn, update_fn
