"""Abstract base classes for Riemannian manifold implementations.

This module defines the core interfaces for Riemannian manifolds, establishing
the contract that concrete manifold implementations must satisfy.
"""

import logging

import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from ..core.type_system import ManifoldPoint, TangentVector

logger = logging.getLogger(__name__)


class ManifoldError(Exception):
    """Base exception for manifold-related errors."""

    pass


class DimensionError(ManifoldError):
    """Exception for dimension mismatches."""

    pass


class Manifold:
    """Abstract base class for Riemannian manifolds.

    This class defines the essential operations required for optimization on
    Riemannian manifolds, including tangent space projections and exponential/logarithmic maps.
    """

    def __init__(self) -> None:
        """Initialize manifold base class."""
        pass

    def proj(self, x: ManifoldPoint, v: Float[Array, "..."]) -> TangentVector:
        """Project a vector from ambient space to the tangent space at point x.

        Args:
            x: Point on the manifold.
            v: Vector in the ambient space to be projected.

        Returns:
            The projection of v onto the tangent space at x.
        """
        raise NotImplementedError("Subclasses must implement projection operation")

    def exp(self, x: ManifoldPoint, v: TangentVector) -> ManifoldPoint:
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

    def log(self, x: ManifoldPoint, y: ManifoldPoint) -> TangentVector:
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

    def retr(self, x: ManifoldPoint, v: TangentVector) -> ManifoldPoint:
        """Apply retraction to move from point x along tangent vector v.

        Retraction is a cheaper approximation of the exponential map that maintains
        essential properties for optimization algorithms.

        Args:
            x: Point on the manifold.
            v: Tangent vector at x.

        Returns:
            The point reached by the retraction from x in direction v.
        """
        raise NotImplementedError("Subclasses must implement retraction")

    def transp(self, x: ManifoldPoint, y: ManifoldPoint, v: TangentVector) -> TangentVector:
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

    def inner(self, x: ManifoldPoint, u: TangentVector, v: TangentVector) -> Array:
        """Compute the Riemannian inner product between tangent vectors u and v at point x.

        Args:
            x: Point on the manifold.
            u: First tangent vector at x.
            v: Second tangent vector at x.

        Returns:
            The inner product <u, v>_x in the Riemannian metric.
        """
        raise NotImplementedError("Subclasses must implement Riemannian inner product")

    def dist(self, x: ManifoldPoint, y: ManifoldPoint) -> Array:
        """Compute the Riemannian distance between points x and y on the manifold.

        Args:
            x: First point on the manifold.
            y: Second point on the manifold.

        Returns:
            The geodesic distance between x and y.
        """
        v = self.log(x, y)
        return jnp.sqrt(self.inner(x, v, v))

    def norm(self, x: ManifoldPoint, v: TangentVector) -> Array:
        """Compute the norm of tangent vector v at point x.

        Args:
            x: Point on the manifold.
            v: Tangent vector at x.

        Returns:
            The norm ||v||_x in the Riemannian metric.
        """
        return jnp.sqrt(self.inner(x, v, v))

    def random_point(self, key: PRNGKeyArray, *shape: int) -> ManifoldPoint:
        """Generate random point(s) on the manifold.

        Args:
            key: JAX PRNG key.
            *shape: Shape of the output array of points.

        Returns:
            Random point(s) on the manifold with specified shape.
        """
        raise NotImplementedError("Subclasses must implement random point generation")

    def random_tangent(self, key: PRNGKeyArray, x: ManifoldPoint, *shape: int) -> TangentVector:
        """Generate random tangent vector(s) at point x.

        Args:
            key: JAX PRNG key.
            x: Point on the manifold.
            *shape: Shape of the output array of tangent vectors.

        Returns:
            Random tangent vector(s) at x with specified shape.
        """
        raise NotImplementedError("Subclasses must implement random tangent generation")

    def curvature_tensor(self, x: ManifoldPoint, u: TangentVector, v: TangentVector, w: TangentVector) -> TangentVector:
        """Compute the Riemann curvature tensor R(u,v)w at point x.

        Args:
            x: Point on the manifold.
            u: First tangent vector at x.
            v: Second tangent vector at x.
            w: Third tangent vector at x.

        Returns:
            The curvature tensor R(u,v)w at x.
        """
        raise NotImplementedError("Curvature tensor computation not implemented")

    def sectional_curvature(self, x: ManifoldPoint, u: TangentVector, v: TangentVector) -> Array:
        """Compute the sectional curvature of the plane spanned by u and v at point x.

        Args:
            x: Point on the manifold.
            u: First tangent vector at x.
            v: Second tangent vector at x.

        Returns:
            The sectional curvature K(u,v) at x.
        """
        raise NotImplementedError("Sectional curvature computation not implemented")

    def injectivity_radius(self, x: ManifoldPoint) -> Array:
        """Compute the injectivity radius at point x.

        Args:
            x: Point on the manifold.

        Returns:
            The injectivity radius at x.
        """
        raise NotImplementedError("Injectivity radius computation not implemented")

    def validate_point(self, x: ManifoldPoint, atol: float = 1e-6) -> bool | Array:
        """Validate that x is a valid point on the manifold.

        Args:
            x: Point to validate.
            atol: Absolute tolerance for validation.

        Returns:
            True if x is on the manifold, False otherwise.
        """
        raise NotImplementedError("Point validation not implemented")

    def validate_tangent(self, x: ManifoldPoint, v: TangentVector, atol: float = 1e-6) -> bool | Array:
        """Validate that v is a valid tangent vector at point x.

        Args:
            x: Point on the manifold.
            v: Vector to validate.
            atol: Absolute tolerance for validation.

        Returns:
            True if v is in the tangent space at x, False otherwise.
        """
        # Default implementation: check if v equals its projection
        proj_v = self.proj(x, v)
        result = jnp.allclose(v, proj_v, atol=atol)
        # Return JAX array directly if in traced context to avoid TracerBoolConversionError
        try:
            return bool(result)
        except TypeError:
            # In JAX traced context, return the array directly
            return result

    @property
    def dimension(self) -> int:
        """Intrinsic dimension of the manifold."""
        raise NotImplementedError("Subclasses must define manifold dimension")

    @property
    def ambient_dimension(self) -> int:
        """Dimension of the ambient space."""
        raise NotImplementedError("Subclasses must define ambient dimension")

    def __repr__(self) -> str:
        """String representation of the manifold."""
        return f"{self.__class__.__name__}()"
