"""Implementation of the sphere manifold S^n with its Riemannian geometry.

This module provides operations for optimization on the unit sphere, a fundamental
manifold in Riemannian geometry with applications in directional statistics,
rotation representations, and constrained optimization.
"""

import jax.numpy as jnp
import jax.random as jr
from jax import Array

from .base import Manifold


class Sphere(Manifold):
    """Sphere manifold S^n embedded in R^(n+1) with the canonical Riemannian metric.

    The n-dimensional sphere S^n consists of all unit vectors in R^(n+1), i.e.,
    all points x ∈ R^(n+1) such that ||x|| = 1.
    """

    def proj(self, x, v):
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
        return v - jnp.sum(x * v) * x

    def exp(self, x, v):
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
        safe_norm = jnp.maximum(v_norm, 1e-10)
        # Follow the great circle
        return jnp.cos(safe_norm) * x + jnp.sin(safe_norm) * v / safe_norm

    def log(self, x, y):
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
        safe_norm = jnp.maximum(v_norm, 1e-10)
        # Compute the angle between x and y (geodesic distance)
        theta = jnp.arccos(jnp.clip(jnp.dot(x, y), -1.0, 1.0))
        # Scale the direction vector by the geodesic distance
        return theta * v / safe_norm

    def retr(self, x, v):
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
        return y / jnp.linalg.norm(y)

    def transp(self, x, y, v):
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
        is_small = log_xy_norm < 1e-10

        # If x and y are close, approximate with projection
        result_small = self.proj(y, v)

        # Otherwise, do parallel transport along the geodesic
        # Using the double reflection method
        if is_small:
            return result_small

        # Normal case: compute parallel transport
        u = log_xy / log_xy_norm
        return v - (jnp.dot(v, u) / (1 + jnp.dot(x, y))) * (u + y)

    def inner(self, x, u, v):
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

    def dist(self, x, y):
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

    def random_point(self, key, *shape):
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
            # Default to generating a single point
            ambient_dim = 3  # Default to S^2 embedded in R^3
            shape = (ambient_dim,)

        # Sample from standard normal distribution
        samples = jr.normal(key, shape)

        # Normalize to get points on the sphere
        return samples / jnp.linalg.norm(samples, axis=-1, keepdims=True)

    def random_tangent(self, key, x, *shape):
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

        return tangent_vectors

    def validate_point(self, x, atol: float = 1e-6) -> Array:
        """Validate that x is a valid point on the sphere."""
        # Check that x is a unit vector
        norm = jnp.linalg.norm(x)
        return jnp.allclose(norm, 1.0, atol=atol)

    def validate_tangent(self, x, v, atol: float = 1e-6) -> Array:
        """Validate that v is in the tangent space at x."""
        if not self.validate_point(x, atol):
            return False
        # Check that v is orthogonal to x
        dot_product = jnp.dot(x, v)
        return jnp.allclose(dot_product, 0.0, atol=atol)

    @property
    def dimension(self) -> int:
        """Dimension of the sphere (n for S^n)."""
        return 2  # Default to S^2 (dimension 2)

    @property
    def ambient_dimension(self) -> int:
        """Ambient space dimension (n+1 for S^n)."""
        return 3  # Default to R^3 for S^2
