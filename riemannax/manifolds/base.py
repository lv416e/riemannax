"""Abstract base classes for Riemannian manifold implementations.

This module defines the core interfaces for Riemannian manifolds, establishing
the contract that concrete manifold implementations must satisfy.
"""

import jax.numpy as jnp


class Manifold:
    """Abstract base class for Riemannian manifolds.

    This class defines the essential operations required for optimization on
    Riemannian manifolds, including tangent space projections and exponential/logarithmic maps.
    """

    def proj(self, x, v):
        """Project a vector from ambient space to the tangent space at point x.

        Args:
            x: Point on the manifold.
            v: Vector in the ambient space to be projected.

        Returns:
            The projection of v onto the tangent space at x.
        """
        raise NotImplementedError("Subclasses must implement projection operation")

    def exp(self, x, v):
        """Apply the exponential map to move from point x along tangent vector v.

        The exponential map takes a point x on the manifold and a tangent vector v at x,
        and returns the point on the manifold reached by following the geodesic in the
        direction of v for a distance of ||v||.

        Args:
            x: Point on the manifold.
            v: Tangent vector at x.

        Returns:
            The point reached by following the geodesic from x in direction v.
        """
        raise NotImplementedError("Subclasses must implement exponential map")

    def log(self, x, y):
        """Apply the logarithmic map to find the tangent vector that maps x to y.

        The logarithmic map is the inverse of the exponential map. It takes two points
        x and y on the manifold and returns the tangent vector v at x such that the
        exponential map of v at x gives y.

        Args:
            x: Starting point on the manifold.
            y: Target point on the manifold.

        Returns:
            The tangent vector v at x such that exp(x, v) = y.
        """
        raise NotImplementedError("Subclasses must implement logarithmic map")

    def retr(self, x, v):
        """Apply retraction to move from point x along tangent vector v.

        Retraction is a cheaper approximation of the exponential map that maintains
        essential properties for optimization algorithms.

        Args:
            x: Point on the manifold.
            v: Tangent vector at x.

        Returns:
            The point reached by the retraction from x in direction v.
        """
        # Default implementation uses exponential map
        return self.exp(x, v)

    def transp(self, x, y, v):
        """Parallel transport vector v from tangent space at x to tangent space at y.

        Parallel transport moves a tangent vector along a geodesic while preserving
        its length and angle with the geodesic.

        Args:
            x: Starting point on the manifold.
            y: Target point on the manifold.
            v: Tangent vector at x to be transported.

        Returns:
            The transported vector in the tangent space at y.
        """
        raise NotImplementedError("Subclasses must implement parallel transport")

    def inner(self, x, u, v):
        """Compute the Riemannian inner product between tangent vectors u and v at point x.

        Args:
            x: Point on the manifold.
            u: First tangent vector at x.
            v: Second tangent vector at x.

        Returns:
            The inner product <u, v>_x in the Riemannian metric.
        """
        raise NotImplementedError("Subclasses must implement Riemannian inner product")

    def dist(self, x, y):
        """Compute the Riemannian distance between points x and y on the manifold.

        Args:
            x: First point on the manifold.
            y: Second point on the manifold.

        Returns:
            The geodesic distance between x and y.
        """
        # Default implementation based on logarithmic map
        v = self.log(x, y)
        return jnp.sqrt(self.inner(x, v, v))

    def random_point(self, key, *shape):
        """Generate random point(s) on the manifold.

        Args:
            key: JAX PRNG key.
            *shape: Shape of the output array of points.

        Returns:
            Random point(s) on the manifold with specified shape.
        """
        raise NotImplementedError("Subclasses must implement random point generation")

    def random_tangent(self, key, x, *shape):
        """Generate random tangent vector(s) at point x.

        Args:
            key: JAX PRNG key.
            x: Point on the manifold.
            *shape: Shape of the output array of tangent vectors.

        Returns:
            Random tangent vector(s) at x with specified shape.
        """
        raise NotImplementedError("Subclasses must implement random tangent generation")
