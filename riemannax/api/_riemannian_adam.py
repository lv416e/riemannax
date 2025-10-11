"""Internal utilities for Riemannian Adam optimizer.

This module provides shared helper functions used by both RiemannianOptaxAdapter
and RiemannianOptimizer to avoid code duplication. These functions implement the
core Adam update logic adapted for Riemannian manifolds.

Note:
    This is an internal module (prefix _) and should not be imported directly by
    end users. It exists solely to share implementation details between the public
    API adapters.
"""

import jax.numpy as jnp
from jaxtyping import Array

from riemannax.manifolds.base import Manifold


def compute_adam_step(
    riemannian_grad: Array,
    m: Array,
    v: Array,
    step: int,
    learning_rate: float,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[Array, Array, Array]:
    """Compute Adam update step in tangent space.

    This function implements the standard Adam algorithm in the tangent space
    of a Riemannian manifold. It updates the first and second moment estimates
    and computes the tangent step direction.

    Args:
        riemannian_grad: Riemannian gradient (already projected to tangent space).
        m: First moment estimate (velocity).
        v: Second moment estimate (element-wise variance).
        step: Current optimization step (0-indexed).
        learning_rate: Learning rate.
        b1: Exponential decay rate for first moment (default: 0.9).
        b2: Exponential decay rate for second moment (default: 0.999).
        eps: Small constant for numerical stability (default: 1e-8).

    Returns:
        Tuple of (tangent_step, m_new, v_new) where:
        - tangent_step: Step direction in tangent space
        - m_new: Updated first moment
        - v_new: Updated second moment

    Note:
        The epsilon is placed in the denominator following the standard Adam
        formula for numerical stability.
    """
    # Update moments
    m_new = b1 * m + (1 - b1) * riemannian_grad
    v_new = b2 * v + (1 - b2) * jnp.square(riemannian_grad)

    # Bias correction
    m_hat = m_new / (1 - b1 ** (step + 1))
    v_hat = v_new / (1 - b2 ** (step + 1))

    # Compute tangent step using standard Adam formula
    # Epsilon in denominator for numerical stability
    tangent_step = -learning_rate * m_hat / (jnp.sqrt(v_hat) + eps)

    return tangent_step, m_new, v_new


def transport_adam_state(
    manifold: Manifold,
    x_old: Array,
    x_new: Array,
    m: Array,
    v: Array,
) -> tuple[Array, Array]:
    """Transport Adam first moment after retraction onto the manifold.

    After moving from x_old to x_new via retraction, the first moment (velocity)
    is transported to the new tangent space. The second moment (variance) is NOT
    transported as it represents coordinate-dependent element-wise variance.

    This follows the approach of geoopt (the reference Riemannian optimization library):
    - First moment m is a tangent vector and should be parallel transported
    - Second moment v contains element-wise variances (coordinate-dependent) and
      should remain in the local coordinate representation

    Args:
        manifold: The Riemannian manifold.
        x_old: Previous point on the manifold.
        x_new: New point on the manifold (after retraction).
        m: First moment estimate (velocity) at x_old.
        v: Second moment estimate (element-wise variance) at x_old.

    Returns:
        Tuple of (m_transported, v) where:
        - m_transported: First moment in tangent space at x_new
        - v: Second moment unchanged (coordinate-dependent state)

    Note:
        The first moment (velocity) is a tangent vector that should be parallel
        transported and projected to correct numerical errors. The second moment
        is NOT transported because it represents coordinate-dependent element-wise
        variance computed via jnp.square(grad), not a geometric tangent vector.

    References:
        - Bécigneul & Ganea (2019): "Riemannian Adaptive Optimization Methods"
        - geoopt library: https://github.com/geoopt/geoopt
    """
    # Parallel transport first moment (velocity) - it's a tangent vector
    m_transported = manifold.transp(x_old, x_new, m)

    # Project first moment to tangent space to correct numerical errors
    m_transported = manifold.proj(x_new, m_transported)

    # Second moment is coordinate-dependent variance, not transported
    # It remains in the local coordinate representation
    return m_transported, v
