"""Hyperbolic-specific data models for manifold operations."""

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import jax.numpy as jnp
from jaxtyping import Array, Float

from riemannax.manifolds.base import ManifoldError


class DataModelError(ManifoldError):
    """Raised when data model validation fails."""

    pass


@dataclass
class HyperbolicPoint:
    """Represents a point in hyperbolic space.

    Supports both Poincaré ball and Lorentz models of hyperbolic geometry.
    Validates manifold constraints and provides geometric operations.
    """

    coordinates: Float[Array, "..."]
    model: Literal["poincare", "lorentz"]
    curvature: float = -1.0
    validate: bool = True
    is_valid: bool = field(init=False)

    def __post_init__(self):
        """Validate point constraints after initialization."""
        if self.validate:
            self._validate_constraints()
            self.is_valid = True
        else:
            # Skip validation entirely for JIT compatibility
            self.is_valid = False

    def _validate_constraints(self) -> None:
        """Validate hyperbolic manifold constraints.

        Raises:
            DataModelError: If constraints are not satisfied.
        """
        if self.model == "poincare":
            self._validate_poincare_constraint()
        elif self.model == "lorentz":
            self._validate_lorentz_constraint()
        else:
            raise DataModelError(f"Unknown hyperbolic model: {self.model}")

    def _validate_poincare_constraint(self) -> None:
        """Validate Poincaré ball constraint (norm < 1)."""
        norm_sq = jnp.sum(self.coordinates**2)
        if norm_sq >= 1.0:
            raise DataModelError(
                f"Point norm {jnp.sqrt(norm_sq):.6f} violates Poincaré ball constraint (must be < 1.0)"
            )

    def _validate_lorentz_constraint(self) -> None:
        """Validate Lorentz model constraint (x₀² - Σxᵢ² = -1/curvature)."""
        if len(self.coordinates) < 2:
            raise DataModelError("Lorentz model requires at least 2 coordinates")

        lorentz_product = self.coordinates[0] ** 2 - jnp.sum(self.coordinates[1:] ** 2)
        expected_value = -1.0 / self.curvature

        if jnp.abs(lorentz_product - expected_value) > 1e-6:
            raise DataModelError(
                f"Point violates Lorentz constraint: x₀² - Σxᵢ² = {lorentz_product:.6f}, expected {expected_value:.6f}"
            )

        # Additional constraint: x₀ > 0 for the upper hyperboloid sheet
        if self.coordinates[0] <= 0:
            raise DataModelError("Lorentz model requires x₀ > 0 (upper hyperboloid sheet)")

    def _check_validity(self) -> bool:
        """Check if point satisfies constraints without raising exceptions."""
        try:
            self._validate_constraints()
            return True
        except DataModelError:
            return False

    def norm(self) -> Array:
        """Calculate Euclidean norm of coordinates.

        Returns:
            Euclidean norm of the coordinate vector.
        """
        return jnp.linalg.norm(self.coordinates)

    def distance_to_origin(self) -> Array:
        """Calculate hyperbolic distance to origin.

        Returns:
            Hyperbolic distance to origin in the given model.
        """
        if self.model == "poincare":
            norm_sq = jnp.sum(self.coordinates**2)
            return jnp.sqrt(-self.curvature) * jnp.arctanh(jnp.sqrt(norm_sq))
        elif self.model == "lorentz":
            # Lorentz distance to origin: arccosh(x₀ / sqrt(-c))
            return jnp.arccosh(self.coordinates[0] / jnp.sqrt(-self.curvature))
        else:
            raise DataModelError(f"Unknown hyperbolic model: {self.model}")


@dataclass
class SE3Transform:
    """Represents an SE(3) transformation (Special Euclidean Group in 3D).

    Encodes rigid body transformations combining rotation and translation.
    Validates orthogonality and determinant constraints for rotation matrices.
    """

    rotation: Array
    translation: Array
    validate: bool = True
    is_valid: bool = field(init=False)

    def __post_init__(self):
        """Validate SE(3) constraints after initialization."""
        if self.validate:
            self._validate_constraints()
            self.is_valid = True
        else:
            # Skip validation entirely for JIT compatibility
            self.is_valid = False

    @classmethod
    def identity(cls) -> "SE3Transform":
        """Create identity transformation.

        Returns:
            Identity SE(3) transformation.
        """
        return cls(rotation=jnp.eye(3), translation=jnp.zeros(3), validate=True)

    def _validate_constraints(self) -> None:
        """Validate SE(3) transformation constraints.

        Raises:
            DataModelError: If rotation matrix is not valid.
        """
        # Check matrix dimensions
        if self.rotation.shape != (3, 3):
            raise DataModelError(f"Rotation matrix must be 3x3, got shape {self.rotation.shape}")

        if self.translation.shape != (3,):
            raise DataModelError(f"Translation vector must be length 3, got shape {self.translation.shape}")

        # Check orthogonality: R^T @ R = I
        identity_check = jnp.dot(self.rotation.T, self.rotation)
        if not jnp.allclose(identity_check, jnp.eye(3), atol=1e-6):
            raise DataModelError("Rotation matrix is not orthogonal (R^T @ R ≠ I)")

        # Check determinant: det(R) = +1 (proper rotation, not reflection)
        det = jnp.linalg.det(self.rotation)
        if not jnp.allclose(det, 1.0, atol=1e-6):
            raise DataModelError(f"Rotation matrix determinant is {det:.6f}, must be +1.0")

    def _check_validity(self) -> bool:
        """Check if transformation satisfies constraints without raising exceptions."""
        try:
            self._validate_constraints()
            return True
        except DataModelError:
            return False

    def homogeneous_matrix(self) -> Array:
        """Convert to homogeneous transformation matrix.

        Returns:
            4x4 homogeneous transformation matrix.
        """
        homogeneous = jnp.zeros((4, 4))
        homogeneous = homogeneous.at[:3, :3].set(self.rotation)
        homogeneous = homogeneous.at[:3, 3].set(self.translation)
        homogeneous = homogeneous.at[3, 3].set(1.0)
        return homogeneous

    def inverse(self) -> "SE3Transform":
        """Compute inverse transformation.

        For SE(3), the inverse is: (R^T, -R^T @ t)

        Returns:
            Inverse SE(3) transformation.
        """
        inv_rotation = self.rotation.T
        inv_translation = -jnp.dot(inv_rotation, self.translation)

        return SE3Transform(
            rotation=inv_rotation,
            translation=inv_translation,
            validate=False,  # Skip validation since we know it's valid
        )

    def compose(self, other: "SE3Transform") -> "SE3Transform":
        """Compose this transformation with another.

        Composition order: self ∘ other (apply other first, then self).

        Args:
            other: The transformation to compose with.

        Returns:
            Composed SE(3) transformation.
        """
        composed_rotation = jnp.dot(self.rotation, other.rotation)
        composed_translation = jnp.dot(self.rotation, other.translation) + self.translation

        return SE3Transform(
            rotation=composed_rotation,
            translation=composed_translation,
            validate=False,  # Skip validation since both inputs are valid
        )


@dataclass
class ManifoldParameters:
    """Configuration parameters for manifold operations.

    Centralizes numerical tolerances, iteration limits, and algorithmic choices
    used across manifold computations.
    """

    tolerance: float = 1e-6
    max_iterations: int = 100
    step_size: float = 0.01
    use_retraction: bool = True
    manifold_type: str = "riemannian"

    def __post_init__(self):
        """Validate parameters after initialization."""
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate parameter constraints.

        Raises:
            DataModelError: If parameters are invalid.
        """
        if self.tolerance <= 0:
            raise DataModelError(f"Tolerance must be positive, got {self.tolerance}")

        if self.max_iterations <= 0:
            raise DataModelError(f"Max iterations must be positive, got {self.max_iterations}")

        if self.step_size <= 0:
            raise DataModelError(f"Step size must be positive, got {self.step_size}")

    def is_valid(self) -> bool:
        """Check if parameters are valid.

        Returns:
            True if all parameters are valid.
        """
        try:
            self._validate_parameters()
            return True
        except DataModelError:
            return False

    def check_convergence(self, error: float, iteration: int) -> bool:
        """Check convergence criteria.

        Args:
            error: Current error measure.
            iteration: Current iteration number.

        Returns:
            True if convergence criteria are met.
        """
        tolerance_met = error < self.tolerance
        max_iterations_reached = iteration >= self.max_iterations

        return tolerance_met or max_iterations_reached

    def summary(self) -> dict[str, Any]:
        """Generate parameter summary.

        Returns:
            Dictionary containing all parameter values.
        """
        return {
            "tolerance": self.tolerance,
            "max_iterations": self.max_iterations,
            "step_size": self.step_size,
            "use_retraction": self.use_retraction,
            "manifold_type": self.manifold_type,
        }

    def copy_with(self, **kwargs: Any) -> "ManifoldParameters":
        """Create a copy with modified parameters.

        Args:
            **kwargs: Parameters to modify.

        Returns:
            New ManifoldParameters instance with modifications.
        """
        new_instance = replace(self, **kwargs)
        new_instance._validate_parameters()
        return new_instance
