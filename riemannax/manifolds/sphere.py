"""Implementation of the sphere manifold S^n with its Riemannian geometry.

This module provides operations for optimization on the unit sphere, a fundamental
manifold in Riemannian geometry with applications in directional statistics,
rotation representations, and constrained optimization.
"""

from typing import Any

import jax.numpy as jnp
import jax.random as jr
from jax import lax
from jaxtyping import Array

from ..core.constants import NumericalConstants
from ..core.jit_decorator import jit_optimized
from ..core.type_system import ManifoldPoint, TangentVector
from .base import Manifold


class Sphere(Manifold):
    """Sphere manifold S^n embedded in R^(n+1) with the canonical Riemannian metric.

    The n-dimensional sphere S^n consists of all unit vectors in R^(n+1), i.e.,
    all points x ∈ R^(n+1) such that ||x|| = 1.
    """

    def __init__(self, n: int = 2):
        """Initialize sphere manifold S^n.

        Args:
            n: Dimension of the sphere (default: 2 for S^2 embedded in R^3)

        Raises:
            ValueError: If n < 1 (sphere dimension must be positive)
        """
        if n < 1:
            raise ValueError(f"Sphere dimension must be positive, got {n}")
        self._n = n
        self._ambient_dim = n + 1

    @jit_optimized(static_args=(0,))
    def proj(self, x: Array, v: Array) -> Array:
        """Project vector v onto the tangent space of the sphere at point x.

        The tangent space at x consists of all vectors orthogonal to x.
        The projection removes the component of v parallel to x.

        Args:
            x: Point on the sphere (unit vector).
            v: Vector in the ambient space R^(n+1).

        Returns:
            The orthogonal projection of v onto the tangent space at x.
        """
        # Remove the component of v parallel to x
        dot_product = jnp.sum(x * v, axis=-1, keepdims=True)
        return v - dot_product * x

    @jit_optimized(static_args=(0,))
    def exp(self, x: Array, v: Array) -> Array:
        """Compute the exponential map on the sphere.

        For the sphere, the exponential map corresponds to following a great circle
        in the direction of the tangent vector v.

        Args:
            x: Point on the sphere (unit vector).
            v: Tangent vector at x (orthogonal to x).

        Returns:
            The point on the sphere reached by following the geodesic from x in direction v.
        """
        # Compute the norm of the tangent vector
        v_norm = jnp.linalg.norm(v)
        # Handle numerical stability for small vectors
        safe_norm = jnp.maximum(v_norm, NumericalConstants.EPSILON)
        # Follow the great circle
        return jnp.cos(safe_norm) * x + jnp.sin(safe_norm) * v / safe_norm

    @jit_optimized(static_args=(0,))
    def log(self, x: Array, y: Array) -> Array:
        """Compute the logarithmic map on the sphere.

        For two points x and y on the sphere, this finds the tangent vector v at x
        such that following the geodesic in that direction for distance ||v|| reaches y.

        Args:
            x: Starting point on the sphere (unit vector).
            y: Target point on the sphere (unit vector).

        Returns:
            The tangent vector at x that points toward y along the geodesic.
        """
        # Project y-x onto the tangent space at x
        v = self.proj(x, y - x)
        # Compute the norm of the projected vector
        v_norm = jnp.linalg.norm(v)
        # Handle numerical stability
        safe_norm = jnp.maximum(v_norm, NumericalConstants.EPSILON)
        # Compute the angle between x and y (geodesic distance)
        theta = jnp.arccos(jnp.clip(jnp.dot(x, y), -1.0, 1.0))
        # Scale the direction vector by the geodesic distance
        return jnp.asarray(theta * v / safe_norm)

    @jit_optimized(static_args=(0,))
    def retr(self, x: Array, v: Array) -> Array:
        """Compute the retraction on the sphere.

        For the sphere, a simple retraction is normalization of x + v.

        Args:
            x: Point on the sphere (unit vector).
            v: Tangent vector at x (orthogonal to x).

        Returns:
            The point on the sphere reached by the retraction.
        """
        # Simple retraction by normalization
        y = x + v
        return jnp.asarray(y / jnp.linalg.norm(y))

    @jit_optimized(static_args=(0,))
    def transp(self, x: Array, y: Array, v: Array) -> Array:
        """Parallel transport on the sphere from x to y.

        Parallel transport preserves the inner product and the norm of the vector.

        Args:
            x: Starting point on the sphere (unit vector).
            y: Target point on the sphere (unit vector).
            v: Tangent vector at x to be transported.

        Returns:
            The transported vector in the tangent space at y.
        """
        # Get the tangent vector that takes x to y
        log_xy = self.log(x, y)
        log_xy_norm = jnp.linalg.norm(log_xy)

        # Handle the case when x and y are very close or antipodal
        is_small = log_xy_norm < NumericalConstants.EPSILON

        def small_case() -> Array:
            # If x and y are close, approximate with projection
            return jnp.asarray(self.proj(y, v))

        def normal_case() -> Array:
            # Normal case: compute parallel transport
            u = log_xy / log_xy_norm
            return jnp.asarray(v - (jnp.dot(v, u) / (1 + jnp.dot(x, y))) * (u + y))

        return jnp.asarray(lax.cond(is_small, small_case, normal_case))

    @jit_optimized(static_args=(0,))
    def inner(self, x: Array, u: Array, v: Array) -> Array:
        """Compute the Riemannian inner product on the sphere.

        On the sphere, the Riemannian metric is simply the Euclidean inner product
        in the ambient space restricted to the tangent space.

        Args:
            x: Point on the sphere (unit vector).
            u: First tangent vector at x.
            v: Second tangent vector at x.

        Returns:
            The inner product <u, v>_x in the Riemannian metric.
        """
        return jnp.dot(u, v)

    @jit_optimized(static_args=(0,))
    def dist(self, x: Array, y: Array) -> Array:
        """Compute the geodesic distance between points on the sphere.

        The geodesic distance is the length of the shortest path along the sphere,
        which is the arc length of the great circle connecting x and y.

        Args:
            x: First point on the sphere (unit vector).
            y: Second point on the sphere (unit vector).

        Returns:
            The geodesic distance between x and y.
        """
        # The geodesic distance is the arc length, which is the angle between x and y
        cos_angle = jnp.clip(jnp.dot(x, y), -1.0, 1.0)
        return jnp.arccos(cos_angle)

    def random_point(self, key: Array, *shape: int) -> Array:
        """Generate random point(s) on the sphere.

        Points are sampled uniformly from the sphere using the standard normal
        distribution in the ambient space followed by normalization.

        Args:
            key: JAX PRNG key.
            *shape: Shape of the output array of points.

        Returns:
            Random point(s) on the sphere with specified shape.
        """
        if not shape:
            # Default to generating a single point with correct ambient dimension
            shape = (self.ambient_dimension,)

        # Sample from standard normal distribution
        samples = jr.normal(key, shape)

        # Normalize to get points on the sphere
        return jnp.asarray(samples / jnp.linalg.norm(samples, axis=-1, keepdims=True))

    def random_tangent(self, key: Array, x: Array, *shape: int) -> Array:
        """Generate random tangent vector(s) at point x.

        Tangent vectors are sampled from a normal distribution in the ambient
        space and then projected onto the tangent space at x.

        Args:
            key: JAX PRNG key.
            x: Point on the sphere (unit vector).
            *shape: Shape of the output array of tangent vectors.

        Returns:
            Random tangent vector(s) at x with specified shape.
        """
        if not shape:
            # Default to the shape of x
            shape = x.shape

        # Sample from standard normal distribution
        ambient_vectors = jr.normal(key, shape)

        # Project onto the tangent space at x
        tangent_vectors = self.proj(x, ambient_vectors)

        return jnp.asarray(tangent_vectors)

    def validate_point(self, x: ManifoldPoint, atol: float = 1e-6) -> bool:
        """Validate that x is a valid point on the sphere."""
        # Check that x is a unit vector
        norm = jnp.linalg.norm(x)
        return bool(jnp.allclose(norm, 1.0, atol=atol))

    def validate_tangent(self, x: ManifoldPoint, v: TangentVector, atol: float = 1e-6) -> bool:
        """Validate that v is in the tangent space at x."""
        if not self.validate_point(x, atol):
            return False
        # Check that v is orthogonal to x
        dot_product = jnp.dot(x, v)
        return bool(jnp.allclose(dot_product, 0.0, atol=atol))

    @property
    def dimension(self) -> int:
        """Dimension of the sphere (n for S^n)."""
        return self._n  # Default to S^2 (dimension 2)

    @property
    def ambient_dimension(self) -> int:
        """Ambient space dimension (n+1 for S^n)."""
        return self._ambient_dim  # Default to R^3 for S^2

    # JIT-optimized implementation methods

    def _proj_impl(self, x: Array, v: Array) -> Array:
        """JIT-optimized projection implementation.

        Numerical stability improvements:
        - Safe normalization to prevent division by zero
        - Appropriate handling of small norms
        - Batch processing support
        """
        # Special case: if v is zero, project x onto sphere (normalization)
        v_norm = jnp.linalg.norm(v, axis=-1)
        is_zero_v = v_norm < NumericalConstants.EPSILON

        # Normalize x onto sphere (batch-aware)
        x_norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
        safe_x_norm = jnp.maximum(x_norm, NumericalConstants.EPSILON)  # Prevent division by zero
        normalized_x = x / safe_x_norm

        # Project tangent (batch-aware)
        dot_product = jnp.sum(x * v, axis=-1, keepdims=True)
        # Numerical stabilization through clipping
        clipped_dot = jnp.clip(dot_product, -1e10, 1e10)
        tangent_projection = v - clipped_dot * x

        # Use jnp.where for batch-compatible conditional processing
        is_zero_v = jnp.expand_dims(is_zero_v, axis=-1)
        return jnp.where(is_zero_v, normalized_x, tangent_projection)

    def _exp_impl(self, x: Array, v: Array) -> Array:
        """JIT-optimized exponential map implementation.

        Numerical stability improvements:
        - Safe handling of small vectors
        - Ensured numerical stability of trigonometric functions
        """
        # Compute the norm of the tangent vector with enhanced stability
        v_norm = jnp.linalg.norm(v)

        # Safe norm to prevent division by zero
        safe_norm = jnp.maximum(v_norm, NumericalConstants.EPSILON)

        # Special handling for small vectors
        is_small = v_norm < 1e-8

        def small_vector_case() -> Array:
            # Use first-order approximation for small vectors while preserving sphere constraint
            result = x + v
            # Normalize to preserve sphere constraint
            result_norm = jnp.linalg.norm(result)
            safe_result_norm = jnp.maximum(result_norm, NumericalConstants.EPSILON)
            return result / safe_result_norm

        def normal_case() -> Array:
            # Normal exponential map calculation
            cos_norm = jnp.cos(v_norm)
            sin_norm = jnp.sin(v_norm)
            return cos_norm * x + sin_norm * v / safe_norm

        return jnp.asarray(lax.cond(is_small, small_vector_case, normal_case))

    def _log_impl(self, x: Array, y: Array) -> Array:
        """JIT-optimized logarithmic map implementation.

        Numerical stability improvements:
        - Inner product clipping
        - Stable handling at antipodal points
        - Batch processing support
        """
        # Project y-x onto the tangent space at x
        diff = y - x
        v = self._proj_impl(x, diff)

        # Compute the norm of the projected vector with stability (batch-aware)
        v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
        safe_norm = jnp.maximum(v_norm, NumericalConstants.EPSILON)

        # Compute the angle between x and y with clipping for stability (batch-aware)
        dot_product = jnp.sum(x * y, axis=-1)  # Use sum for batch processing
        clipped_dot = jnp.clip(dot_product, -1.0 + NumericalConstants.EPSILON, 1.0 - NumericalConstants.EPSILON)
        theta = jnp.arccos(clipped_dot)

        # Special handling near antipodal points (batch-aware)
        is_antipodal = jnp.abs(dot_product + 1.0) < 1e-8

        # Expand theta and is_antipodal for broadcasting
        theta = jnp.expand_dims(theta, axis=-1)
        is_antipodal = jnp.expand_dims(is_antipodal, axis=-1)

        # Use jnp.where for batch-compatible conditional processing
        antipodal_result = jnp.pi * v / safe_norm
        normal_result = theta * v / safe_norm

        return jnp.where(is_antipodal, antipodal_result, normal_result)

    def _inner_impl(self, x: Array, u: Array, v: Array) -> Array:
        """JIT-optimized inner product implementation.

        Simple Euclidean inner product on the sphere,
        with clipping added for numerical stability.
        Batch processing support.
        """
        # Use batch-aware dot product
        dot_product = jnp.sum(u * v, axis=-1)
        # Clipping at extreme values (if necessary)
        return jnp.clip(dot_product, -1e10, 1e10)

    def _dist_impl(self, x: Array, y: Array) -> Array:
        """JIT-optimized distance calculation implementation.

        Numerical stability improvements:
        - Inner product clipping
        - Safe calculation within arccos domain
        """
        # The geodesic distance is the arc length (angle between x and y)
        dot_product = jnp.dot(x, y)

        # Numerical stabilization through clipping - ensure arccos domain [-1, 1]
        clipped_dot = jnp.clip(dot_product, -1.0 + NumericalConstants.EPSILON, 1.0 - NumericalConstants.EPSILON)

        return jnp.arccos(clipped_dot)

    def _get_static_args(self, method_name: str) -> tuple[Any, ...]:
        """Static argument configuration for JIT compilation.

        For Sphere manifold, dimensions may change dynamically,
        so static arguments are not used at present.

        In the future, dimensions could be specified as static arguments
        """
        # No static arguments in current implementation
        # In the future, dimension could be used as static argument
        return ()
