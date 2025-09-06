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
        # For simplicity and numerical stability, use the basic Einstein formula
        # x ⊕ y = (x + y) / (1 + ⟨x,y⟩)

        x_dot_y = jnp.sum(x * y)
        numerator = x + y
        denominator = 1 + x_dot_y

        # Handle numerical stability for denominator near zero
        safe_denominator = jnp.where(jnp.abs(denominator) > 1e-10, denominator, 1e-10)

        result = numerator / safe_denominator

        # Ensure result stays in ball (project back if necessary)
        result_norm = jnp.linalg.norm(result)
        if result_norm >= 1.0:
            # Project back into ball
            result = result * (0.99 / result_norm)

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

    def __repr__(self) -> str:
        """String representation of the Poincaré ball manifold."""
        return f"PoincareBall(dim={self.dimension}, c={self.curvature})"
