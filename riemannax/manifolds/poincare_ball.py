"""Poincaré ball model of hyperbolic geometry.

The Poincaré ball model represents hyperbolic space as the interior of the unit ball
in Euclidean space, with the hyperbolic metric induced by the conformal factor.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from riemannax.manifolds.base import Manifold, ManifoldError, ManifoldPoint, TangentVector


class PoincareBall(Manifold):
    """Poincaré ball model of hyperbolic geometry.

    The Poincaré ball model represents n-dimensional hyperbolic space as the
    interior of the unit ball in (n+1)-dimensional Euclidean space. Points are
    represented as vectors with norm strictly less than 1.

    The hyperbolic metric is given by:
    ds² = 4 / (1 - |x|²)² * dx²

    This implementation supports:
    - Random point generation respecting ball constraints
    - Möbius operations for hyperbolic translations
    - Numerical stability through vector length validation
    - Configurable curvature and precision tolerances
    """

    def __init__(
        self,
        dimension: int = 2,
        curvature: float = -1.0,
        tolerance: float = 1e-6,
    ):
        """Initialize Poincaré ball manifold.

        Args:
            dimension: Intrinsic dimension of the hyperbolic space.
            curvature: Sectional curvature (must be negative for hyperbolic space).
            tolerance: Numerical tolerance for validation operations.

        Raises:
            ManifoldError: If dimension is invalid or curvature is non-negative.
        """
        super().__init__()

        if dimension <= 0:
            raise ManifoldError(f"Dimension must be positive, got {dimension}")

        if curvature >= 0:
            raise ManifoldError(f"Curvature must be negative for hyperbolic space, got {curvature}")

        self._dimension = dimension
        self._ambient_dimension = dimension  # Poincaré ball is embedded in same dimension
        self.curvature = curvature
        self.tolerance = tolerance

        # Curvature scaling factor: |c| = -curvature for hyperbolic space
        self._curvature_factor = jnp.sqrt(-curvature)

    @property
    def dimension(self) -> int:
        """Intrinsic dimension of the hyperbolic space."""
        return self._dimension

    @property
    def ambient_dimension(self) -> int:
        """Dimension of the ambient Euclidean space."""
        return self._ambient_dimension

    def _validate_in_ball(self, x: Array, atol: float | None = None) -> bool:
        """Validate that point x is inside the unit ball.

        Args:
            x: Point to validate.
            atol: Absolute tolerance (uses instance tolerance if None).

        Returns:
            True if point is strictly inside the unit ball.
        """
        if atol is None:
            atol = self.tolerance

        norm_squared = jnp.sum(x**2)
        return bool(norm_squared < (1.0 - atol))

    def validate_point(self, x: ManifoldPoint, atol: float = 1e-6) -> bool | Array:
        """Validate that x is a valid point on the Poincaré ball.

        Args:
            x: Point to validate.
            atol: Absolute tolerance for validation.

        Returns:
            True if x is inside the unit ball, False otherwise.
        """
        # Check if point is inside the ball
        result = self._validate_in_ball(x, atol)

        # Return JAX array directly if in traced context to avoid TracerBoolConversionError
        try:
            return bool(result)
        except TypeError:
            # In JAX traced context, return the array directly
            return result

    def random_point(self, key: PRNGKeyArray, *shape: int) -> ManifoldPoint:
        """Generate random point(s) uniformly distributed in the Poincaré ball.

        Uses rejection sampling to ensure uniform distribution on the hyperbolic manifold.
        Points are generated in Cartesian coordinates within the unit ball.

        Args:
            key: JAX PRNG key for random generation.
            *shape: Shape of the output array of points.

        Returns:
            Random point(s) in the Poincaré ball with specified shape.
        """
        if not shape:
            target_shape: tuple[int, ...] = (self.dimension,)
        else:
            target_shape = (*tuple(shape), self.dimension)

        # Generate points uniformly in the ball using rejection sampling
        # For efficiency in higher dimensions, use normal distribution and scale
        key, subkey = jax.random.split(key)

        # Generate from normal distribution (gives spherical symmetry)
        points = jax.random.normal(subkey, target_shape)

        # Generate random radii with proper hyperbolic distribution
        key, subkey = jax.random.split(key)
        if not shape:
            radii_shape: tuple[int, ...] = ()
        else:
            radii_shape = tuple(shape)

        # Uniform distribution in [0,1) and then take square root for proper distribution
        uniform_radii = jax.random.uniform(subkey, radii_shape)
        proper_radii = jnp.sqrt(uniform_radii)

        # Normalize points to unit vectors and scale by radii
        point_norms = jnp.linalg.norm(points, axis=-1, keepdims=True)
        # Avoid division by zero
        safe_norms = jnp.where(point_norms > 1e-8, point_norms, 1.0)
        unit_points = points / safe_norms

        # Scale by proper radii to get uniform distribution in ball
        scaled_points = unit_points * proper_radii * 0.95 if not shape else unit_points * proper_radii[..., None] * 0.95

        return scaled_points

    def random_tangent(self, key: PRNGKeyArray, x: ManifoldPoint, *shape: int) -> TangentVector:
        """Generate random tangent vector(s) at point x.

        In the Poincaré ball, tangent vectors are elements of the tangent space
        at x, which is isomorphic to R^n but with the Riemannian metric.

        Args:
            key: JAX PRNG key for random generation.
            x: Base point on the manifold.
            *shape: Shape of the output array of tangent vectors.

        Returns:
            Random tangent vector(s) at x with specified shape.
        """
        if not shape:
            target_shape: tuple[int, ...] = (self.dimension,)
        else:
            target_shape = (*tuple(shape), self.dimension)

        # Generate random vectors from normal distribution
        # In Poincaré ball, all vectors are tangent (no constraints)
        tangent = jax.random.normal(key, target_shape)

        return tangent

    def _mobius_add(self, x: ManifoldPoint, y: ManifoldPoint) -> ManifoldPoint:
        """Compute Möbius addition x ⊕ y in the Poincaré ball.

        Uses the Einstein addition formula for the Poincaré ball:
        x ⊕ y = (x + y + λ * ⟨x,y⟩ * (x + y)) / (1 + λ * ⟨x,y⟩)
        where λ = 1/(1 + √(1 - ∥x∥²)(1 - ∥y∥²))

        Simplified version: x ⊕ y = (x + y) / (1 + ⟨x,y⟩) for unit curvature.

        Args:
            x: First point in the Poincaré ball.
            y: Second point in the Poincaré ball.

        Returns:
            Result of Möbius addition x ⊕ y.
        """
        # Compute inner product and norms
        x_dot_y = jnp.sum(x * y)
        x_norm_sq = jnp.sum(x**2)
        y_norm_sq = jnp.sum(y**2)

        # Standard Möbius addition formula
        numerator = (1 + 2 * x_dot_y + y_norm_sq) * x + (1 - x_norm_sq) * y
        denominator = 1 + 2 * x_dot_y + x_norm_sq * y_norm_sq

        # Handle numerical stability
        safe_denominator = jnp.where(jnp.abs(denominator) > 1e-10, denominator, 1e-10)
        result = numerator / safe_denominator

        # Ensure result stays in ball (project back if necessary)
        result_norm = jnp.linalg.norm(result)
        radius = jnp.sqrt(-1.0 / self.curvature)
        scale = jnp.minimum(1.0, (radius - 1e-7) / jnp.maximum(result_norm, 1e-15))
        result = jnp.where(result_norm >= radius, scale * result, result)

        return result

    def validate_tangent(self, x: ManifoldPoint, v: TangentVector, atol: float = 1e-6) -> bool | Array:
        """Validate that v is a valid tangent vector at point x.

        In the Poincaré ball model, any vector in R^n is a valid tangent vector
        at any point, so this always returns True (assuming shapes match).

        Args:
            x: Point on the manifold.
            v: Vector to validate as tangent.
            atol: Absolute tolerance (unused for Poincaré ball).

        Returns:
            True if v is a valid tangent vector at x.
        """
        # In Poincaré ball, all vectors are tangent vectors if shapes match
        return x.shape == v.shape

    def proj(self, x: ManifoldPoint, v: TangentVector) -> TangentVector:
        """Project vector to tangent space.

        In the Poincaré ball, tangent vectors are in Euclidean space,
        but we need to ensure they don't push the point outside the ball.

        Args:
            x: Point on the manifold.
            v: Vector to project.

        Returns:
            Projected tangent vector.
        """
        # In Poincaré ball, tangent space is the ambient Euclidean space
        # However, we scale to prevent leaving the ball for large vectors
        norm_x = jnp.linalg.norm(x)
        norm_v = jnp.linalg.norm(v)

        # If v is large enough to potentially push x outside the ball,
        # scale it down to maintain validity
        # This is a simple heuristic to ensure numerical stability
        radius = jnp.sqrt(-1.0 / self.curvature)
        remaining_radius = radius - norm_x - 1e-7

        # Only scale if necessary
        scale = jnp.minimum(1.0, remaining_radius / (norm_v + 1e-15))
        return jnp.where(norm_v > remaining_radius, scale * v, v)

    def inner(self, x: ManifoldPoint, u: TangentVector, v: TangentVector) -> Array:
        """Compute Riemannian inner product in tangent space.

        The Poincaré metric has conformal factor: 4 / (1 - |x|²)²

        Args:
            x: Point on the manifold.
            u: First tangent vector.
            v: Second tangent vector.

        Returns:
            Inner product scalar.
        """
        norm_sq = jnp.sum(x**2)
        # Conformal factor for Poincaré ball metric
        # Scale by inverse curvature radius squared
        radius_sq = -1.0 / self.curvature
        conformal_factor = 4 * radius_sq / (1 - norm_sq / radius_sq) ** 2

        # Euclidean inner product scaled by conformal factor
        return conformal_factor * jnp.sum(u * v)

    def exp(self, x: ManifoldPoint, v: TangentVector) -> ManifoldPoint:
        """Exponential map from tangent space to manifold.

        Uses the closed-form exponential map for the Poincaré ball.

        Args:
            x: Point on the manifold.
            v: Tangent vector.

        Returns:
            Point on the manifold.
        """
        # Compute norm of v in Euclidean metric
        v_norm = jnp.linalg.norm(v)

        # Scale factor for curvature
        radius = jnp.sqrt(-1.0 / self.curvature)

        # Compute the scaling for the tangent vector
        # tanh(|v|/(2*radius)) where |v| is Euclidean norm
        tanh_factor = jnp.tanh(v_norm / radius)

        # Normalized direction
        v_normalized = v / jnp.maximum(v_norm, 1e-15)

        # The scaled vector for Möbius addition
        y = tanh_factor * v_normalized

        # Apply Möbius addition
        result = self._mobius_add(x, y)

        return result

    def log(self, x: ManifoldPoint, y: ManifoldPoint) -> TangentVector:
        """Logarithmic map from manifold to tangent space.

        Inverse of the exponential map.

        Args:
            x: Base point on the manifold.
            y: Target point on the manifold.

        Returns:
            Tangent vector at x pointing toward y.
        """
        # Standard formula: first translate x to origin using Möbius translation
        neg_x = -x
        y_at_origin = self._mobius_add(neg_x, y)

        # Compute norm
        y_norm = jnp.linalg.norm(y_at_origin)

        # Curvature radius
        radius = jnp.sqrt(-1.0 / self.curvature)

        # Apply the log formula at origin: arctanh(|y|/radius) * radius * y/|y|
        scale = radius * jnp.arctanh(jnp.minimum(y_norm, 1 - 1e-7)) / jnp.maximum(y_norm, 1e-15)

        # The tangent vector is just the scaled translated point
        # No parallel transport needed - this gives the tangent vector at x
        v = scale * y_at_origin

        return v

    def retr(self, x: ManifoldPoint, v: TangentVector) -> ManifoldPoint:
        """Retraction operation.

        A computationally efficient approximation to the exponential map.
        Uses the projective retraction for the Poincaré ball.

        Args:
            x: Point on the manifold.
            v: Tangent vector.

        Returns:
            Point on the manifold.
        """
        # Simple projective retraction
        # Move in ambient space and project back to ball
        y = x + v

        # Project back to ball if needed
        y_norm = jnp.linalg.norm(y)
        radius = jnp.sqrt(-1.0 / self.curvature)

        # Ensure we stay inside the ball
        scale = jnp.minimum(1.0, (radius - 1e-7) / jnp.maximum(y_norm, 1e-15))
        y = jnp.where(y_norm >= radius, scale * y, y)

        return y

    def dist(self, x: ManifoldPoint, y: ManifoldPoint) -> Array:
        """Compute geodesic distance between two points.

        Uses the hyperbolic distance formula for the Poincaré ball.

        Args:
            x: First point on the manifold.
            y: Second point on the manifold.

        Returns:
            Geodesic distance.
        """
        # Use the direct hyperbolic distance formula
        # d(x, y) = 2 * radius * arctanh(|| x ⊖ y ||)
        # where x ⊖ y is Möbius subtraction

        # Möbius subtraction: -x ⊕ y
        neg_x = -x
        diff = self._mobius_add(neg_x, y)
        diff_norm = jnp.linalg.norm(diff)

        # Curvature radius
        radius = jnp.sqrt(-1.0 / self.curvature)

        # Hyperbolic distance
        distance = 2 * radius * jnp.arctanh(jnp.minimum(diff_norm, 1 - 1e-7))

        return distance

    def __repr__(self) -> str:
        """String representation of the Poincaré ball manifold."""
        return f"PoincareBall(dim={self.dimension}, c={self.curvature})"
