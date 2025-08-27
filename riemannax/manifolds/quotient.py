"""Abstract quotient manifold implementation.

This module provides the abstract base class for quotient manifolds M/G,
where a Lie group G acts on a manifold M. Quotient manifolds arise when
points are considered equivalent under group actions.

Key concepts:
- Equivalence classes: Points x and g·x are equivalent for any group element g
- Horizontal space: Tangent vectors orthogonal to group action directions
- Quotient operations: Geometric operations respecting equivalence classes

Mathematical foundation:
For a quotient manifold M/G, the tangent space splits as:
T_x M = H_x ⊕ V_x
where H_x is the horizontal space (orthogonal to group action)
and V_x is the vertical space (tangent to group orbit).
"""

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from ..core.jit_decorator import jit_optimized
from .base import Manifold


class QuotientManifold(Manifold, ABC):
    """Abstract base class for quotient manifolds M/G.

    A quotient manifold is formed when a Lie group G acts on a manifold M,
    creating equivalence classes of points under the group action.

    This class provides the framework for:
    1. Horizontal space projections (orthogonal to group action)
    2. Quotient-aware geometric operations (exp, log, distance)
    3. Group action handling and equivalence class management
    4. Numerical stability for near-singular configurations
    """

    def __init__(self, has_quotient_structure: bool = True):
        """Initialize quotient manifold.

        Args:
            has_quotient_structure: Whether to enable quotient-specific operations
        """
        super().__init__()
        self.has_quotient_structure = has_quotient_structure

    @property
    @abstractmethod
    def total_space_dim(self) -> int:
        """Dimension of the total space M (before quotient)."""
        pass

    @property
    @abstractmethod
    def group_dim(self) -> int:
        """Dimension of the group G acting on M."""
        pass

    @abstractmethod
    def horizontal_proj(self, x: Array, v: Array) -> Array:
        """Project vector v to horizontal space at point x.

        The horizontal space H_x is orthogonal to the tangent space of the
        group orbit through x, i.e., orthogonal to {d/dt g(t)·x|_{t=0} : g(t) ∈ G}.

        Args:
            x: Point on the manifold
            v: Vector in ambient space

        Returns:
            Projection of v onto horizontal space at x
        """
        pass

    @abstractmethod
    def group_action(self, x: Array, g: Array) -> Array:
        """Apply group element g to point x.

        For quotient manifold M/G, this computes g·x where g ∈ G.

        Args:
            x: Point on the manifold
            g: Group element

        Returns:
            Result of group action g·x
        """
        pass

    @abstractmethod
    def random_group_element(self, key: PRNGKeyArray) -> Array:
        """Generate random element from the group G.

        Args:
            key: JAX PRNG key

        Returns:
            Random group element
        """
        pass

    def is_horizontal(self, x: Array, v: Array, atol: float = 1e-10) -> bool:
        """Check if vector v is in horizontal space at x.

        Args:
            x: Point on the manifold
            v: Vector to check
            atol: Absolute tolerance

        Returns:
            True if v is horizontal at x
        """
        v_proj = self.horizontal_proj(x, v)
        return bool(jnp.allclose(v, v_proj, atol=atol))

    def are_equivalent(self, x: Array, y: Array, atol: float = 1e-10) -> bool:
        """Check if points x and y are in the same equivalence class.

        Args:
            x: First point
            y: Second point
            atol: Absolute tolerance

        Returns:
            True if x and y represent the same equivalence class
        """
        # Default implementation: check if quotient distance is zero
        return bool(self.quotient_dist(x, y) < atol)

    @jit_optimized(static_args=(0,))
    def quotient_exp(self, x: Array, v: Array) -> Array:
        """Quotient-aware exponential map.

        Computes exponential map in quotient manifold, ensuring the result
        respects equivalence classes. For many quotient manifolds, this
        reduces to the regular exponential map followed by projection.

        Args:
            x: Base point on quotient manifold
            v: Tangent vector (should be horizontal)

        Returns:
            Point on quotient manifold reached by exponential map
        """
        # Default implementation: project to horizontal space and use regular exp
        v_horizontal = self.horizontal_proj(x, v)
        return self.exp(x, v_horizontal)

    @jit_optimized(static_args=(0,))
    def quotient_log(self, x: Array, y: Array) -> Array:
        """Quotient-aware logarithmic map.

        Computes logarithmic map in quotient manifold, returning a horizontal
        tangent vector at x pointing toward the equivalence class of y.

        Args:
            x: Base point on quotient manifold
            y: Target point on quotient manifold

        Returns:
            Horizontal tangent vector at x
        """
        # Default implementation: compute regular log and project to horizontal
        v = self.log(x, y)
        return self.horizontal_proj(x, v)

    @jit_optimized(static_args=(0,))
    def quotient_dist(self, x: Array, y: Array) -> Array:
        """Quotient-aware distance computation.

        Computes the distance between equivalence classes represented by x and y.
        This is the infimum of distances between representatives of the classes.

        Args:
            x: First point on quotient manifold
            y: Second point on quotient manifold

        Returns:
            Distance between equivalence classes
        """
        # Default implementation: use horizontal logarithmic map
        v = self.quotient_log(x, y)
        return jnp.sqrt(self.inner(x, v, v))

    def lift_tangent(self, x: Array, v_quotient: Array) -> Array:
        """Lift tangent vector from quotient to total space.

        Given a tangent vector in the quotient manifold T_[x](M/G),
        lift it to a horizontal vector in T_x M.

        Args:
            x: Point on total space M
            v_quotient: Tangent vector in quotient space

        Returns:
            Horizontal tangent vector in total space
        """
        # Default implementation: horizontal projection (assuming input is already in total space)
        return self.horizontal_proj(x, v_quotient)

    def project_tangent(self, x: Array, v_total: Array) -> Array:
        """Project tangent vector from total space to quotient space.

        Given a tangent vector in T_x M, project it to the quotient
        tangent space T_[x](M/G) by horizontal projection.

        Args:
            x: Point on total space M
            v_total: Tangent vector in total space

        Returns:
            Tangent vector in quotient space (horizontal component)
        """
        return self.horizontal_proj(x, v_total)

    def quotient_inner(self, x: Array, u: Array, v: Array) -> Array:
        """Inner product in quotient manifold.

        Computes inner product of horizontal tangent vectors u and v at x.

        Args:
            x: Point on quotient manifold
            u: First horizontal tangent vector
            v: Second horizontal tangent vector

        Returns:
            Inner product in quotient space
        """
        # Ensure vectors are horizontal
        u_h = self.horizontal_proj(x, u)
        v_h = self.horizontal_proj(x, v)
        return self.inner(x, u_h, v_h)

    def quotient_proj(self, x: Array, v: Array) -> Array:
        """Project vector to quotient tangent space.

        This is equivalent to horizontal projection for quotient manifolds.

        Args:
            x: Point on quotient manifold
            v: Vector to project

        Returns:
            Projection to quotient tangent space
        """
        return self.horizontal_proj(x, v)

    def validate_horizontal(self, x: Array, v: Array, atol: float = 1e-6) -> bool:
        """Validate that vector v is horizontal at point x.

        Args:
            x: Point on manifold
            v: Vector to validate
            atol: Absolute tolerance

        Returns:
            True if vector is horizontal
        """
        return self.is_horizontal(x, v, atol=atol)

    def random_horizontal_tangent(self, key: PRNGKeyArray, x: Array, *shape: int) -> Array:
        """Generate random horizontal tangent vector at x.

        Args:
            key: JAX PRNG key
            x: Base point
            *shape: Shape of output

        Returns:
            Random horizontal tangent vector
        """
        # Generate random vector and project to horizontal space
        target_shape = (*shape, *x.shape) if shape else x.shape

        v_random = jr.normal(key, target_shape)

        if shape:
            # Handle batch case
            def proj_fn(vi: Array) -> Array:
                return self.horizontal_proj(x, vi)
            return jax.vmap(proj_fn)(v_random)
        else:
            return self.horizontal_proj(x, v_random)

    # Enhanced numerical stability methods

    def _safe_horizontal_proj(self, x: Array, v: Array, regularization: float = 1e-12) -> Array:
        """Numerically stable horizontal projection.

        Uses regularization to handle near-singular configurations.

        Args:
            x: Point on manifold
            v: Vector to project
            regularization: Regularization parameter

        Returns:
            Stable horizontal projection
        """
        # Default implementation delegates to subclass
        return self.horizontal_proj(x, v)

    def _handle_near_equivalence(self, x: Array, y: Array, threshold: float = 1e-12) -> Array:
        """Handle cases where points are nearly equivalent.

        Args:
            x: First point
            y: Second point
            threshold: Threshold for near-equivalence

        Returns:
            Stable computation result
        """
        # Check if points are nearly equivalent
        if self.quotient_dist(x, y) < threshold:
            # Return zero vector for nearly equivalent points
            return jnp.zeros_like(x)

        # Otherwise use regular quotient log
        return self.quotient_log(x, y)

    # Batch operation support

    def batch_horizontal_proj(self, x_batch: Array, v_batch: Array) -> Array:
        """Batch horizontal projection using vmap.

        Args:
            x_batch: Batch of base points
            v_batch: Batch of vectors to project

        Returns:
            Batch of horizontal projections
        """
        return jax.vmap(self.horizontal_proj, in_axes=(0, 0))(x_batch, v_batch)

    def batch_quotient_exp(self, x_batch: Array, v_batch: Array) -> Array:
        """Batch quotient exponential map using vmap.

        Args:
            x_batch: Batch of base points
            v_batch: Batch of tangent vectors

        Returns:
            Batch of exponential map results
        """
        return jax.vmap(self.quotient_exp, in_axes=(0, 0))(x_batch, v_batch)

    def batch_quotient_log(self, x_batch: Array, y_batch: Array) -> Array:
        """Batch quotient logarithmic map using vmap.

        Args:
            x_batch: Batch of base points
            y_batch: Batch of target points

        Returns:
            Batch of logarithmic map results
        """
        return jax.vmap(self.quotient_log, in_axes=(0, 0))(x_batch, y_batch)

    def batch_quotient_dist(self, x_batch: Array, y_batch: Array) -> Array:
        """Batch quotient distance computation using vmap.

        Args:
            x_batch: Batch of first points
            y_batch: Batch of second points

        Returns:
            Batch of distances
        """
        return jax.vmap(self.quotient_dist, in_axes=(0, 0))(x_batch, y_batch)

    def __repr__(self) -> str:
        """String representation of quotient manifold."""
        if hasattr(self, 'total_space_dim') and hasattr(self, 'group_dim'):
            return f"{self.__class__.__name__}(total_dim={self.total_space_dim}, group_dim={self.group_dim})"
        return f"{self.__class__.__name__}(quotient_structure={self.has_quotient_structure})"
