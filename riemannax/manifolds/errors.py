"""Enhanced manifold error hierarchy and validation system.

This module provides a comprehensive error handling system for manifold operations,
including specialized exceptions for different types of geometric and numerical errors,
and validation utilities for ensuring mathematical correctness.
"""

import jax.numpy as jnp
from jaxtyping import Array


class ManifoldError(Exception):
    """Base exception for manifold-related errors."""

    pass


class DimensionError(ManifoldError):
    """Exception for dimension mismatches in manifold operations."""

    def __init__(self, message: str, expected: int | tuple | None = None, actual: int | tuple | None = None):
        """Initialize DimensionError with dimension information."""
        super().__init__(message)
        self.expected = expected
        self.actual = actual

    def __str__(self) -> str:
        """Return string representation with dimension information."""
        base_msg = super().__str__()
        if self.expected is not None and self.actual is not None:
            return f"{base_msg} (expected={self.expected}, actual={self.actual})"
        return base_msg


class ConvergenceError(ManifoldError):
    """Exception for algorithms that fail to converge."""

    def __init__(
        self,
        message: str,
        max_iterations: int | None = None,
        final_error: float | None = None,
        tolerance: float | None = None,
    ):
        """Initialize ConvergenceError with algorithm convergence information."""
        super().__init__(message)
        self.max_iterations = max_iterations
        self.final_error = final_error
        self.tolerance = tolerance


class NumericalStabilityError(ManifoldError):
    """Exception for numerical stability issues in manifold computations."""

    def __init__(
        self,
        message: str,
        condition_number: float | None = None,
        matrix_norm: float | None = None,
        recommended_action: str | None = None,
    ):
        """Initialize NumericalStabilityError with numerical diagnostics."""
        super().__init__(message)
        self.condition_number = condition_number
        self.matrix_norm = matrix_norm
        self.recommended_action = recommended_action


class InvalidPointError(ManifoldError):
    """Exception for points that do not lie on the manifold."""

    def __init__(
        self,
        message: str,
        point: Array | None = None,
        violated_constraint: str | None = None,
        constraint_value: float | None = None,
    ):
        """Initialize InvalidPointError with constraint violation information."""
        super().__init__(message)
        self.point = point
        self.violated_constraint = violated_constraint
        self.constraint_value = constraint_value


class InvalidTangentVectorError(ManifoldError):
    """Exception for tangent vectors that do not lie in the tangent space."""

    def __init__(
        self,
        message: str,
        tangent_vector: Array | None = None,
        base_point: Array | None = None,
        orthogonality_error: float | None = None,
    ):
        """Initialize InvalidTangentVectorError with tangent space violation information."""
        super().__init__(message)
        self.tangent_vector = tangent_vector
        self.base_point = base_point
        self.orthogonality_error = orthogonality_error


class ManifoldConstraintError(ManifoldError):
    """Exception for constraint violations in manifold operations."""

    def __init__(
        self,
        message: str,
        constraint_name: str | None = None,
        violation_magnitude: float | None = None,
        tolerance: float | None = None,
    ):
        """Initialize ManifoldConstraintError with constraint violation details."""
        super().__init__(message)
        self.constraint_name = constraint_name
        self.violation_magnitude = violation_magnitude
        self.tolerance = tolerance


class GeometricError(ManifoldError):
    """Exception for geometric operation failures on manifolds."""

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        manifold_type: str | None = None,
        step_size: float | None = None,
        norm_tangent: float | None = None,
    ):
        """Initialize GeometricError with geometric operation context."""
        super().__init__(message)
        self.operation = operation
        self.manifold_type = manifold_type
        self.step_size = step_size
        self.norm_tangent = norm_tangent


def validate_manifold_point(point: Array, manifold_type: str, tolerance: float = 1e-8) -> None:
    """Validate that a point lies on the specified manifold.

    Args:
        point: Point to validate
        manifold_type: Type of manifold ('sphere', 'spd', 'grassmann', etc.)
        tolerance: Numerical tolerance for validation

    Raises:
        InvalidPointError: If point does not satisfy manifold constraints
    """
    if manifold_type.lower() == "sphere":
        norm_sq = jnp.sum(point**2)
        norm_error = abs(norm_sq - 1.0)
        if norm_error > tolerance:
            raise InvalidPointError(
                "Point does not have unit norm for sphere manifold",
                point=point,
                violated_constraint="unit_norm",
                constraint_value=float(norm_error),
            )
    elif manifold_type.lower() == "spd":
        # Check positive definiteness
        validate_positive_definite(point, tolerance=tolerance)
    else:
        # For other manifolds, basic array validation
        if not isinstance(point, jnp.ndarray):
            raise InvalidPointError(f"Point must be a JAX array for {manifold_type} manifold", point=point)


def validate_tangent_vector(tangent: Array, base_point: Array, manifold_type: str, tolerance: float = 1e-8) -> None:
    """Validate that a tangent vector lies in the tangent space at base_point.

    Args:
        tangent: Tangent vector to validate
        base_point: Base point on the manifold
        manifold_type: Type of manifold
        tolerance: Numerical tolerance for validation

    Raises:
        InvalidTangentVectorError: If tangent vector is not in tangent space
    """
    if manifold_type.lower() == "sphere":
        # On sphere, tangent vectors must be orthogonal to the point
        dot_product = jnp.dot(tangent, base_point)
        if abs(dot_product) > tolerance:
            raise InvalidTangentVectorError(
                "Tangent vector not orthogonal to base point on sphere",
                tangent_vector=tangent,
                base_point=base_point,
                orthogonality_error=float(abs(dot_product)),
            )
    elif manifold_type.lower() == "spd":
        # For SPD manifolds, tangent vectors should be symmetric
        if tangent.ndim == 2:
            symmetry_error = jnp.max(jnp.abs(tangent - tangent.T))
            if symmetry_error > tolerance:
                raise InvalidTangentVectorError(
                    "Tangent vector not symmetric for SPD manifold",
                    tangent_vector=tangent,
                    base_point=base_point,
                    orthogonality_error=float(symmetry_error),
                )


def validate_positive_definite(matrix: Array, tolerance: float = 1e-8) -> None:
    """Validate that a matrix is positive definite.

    Args:
        matrix: Matrix to validate
        tolerance: Tolerance for eigenvalue positivity

    Raises:
        NumericalStabilityError: If matrix is not positive definite
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise NumericalStabilityError("Matrix must be square for positive definiteness check")

    # Check symmetry first
    symmetry_error = jnp.max(jnp.abs(matrix - matrix.T))
    if symmetry_error > tolerance:
        raise NumericalStabilityError(f"Matrix is not symmetric (error: {symmetry_error})")

    # Check eigenvalues
    eigenvals = jnp.linalg.eigvals(matrix)
    min_eigenval = jnp.min(jnp.real(eigenvals))  # Take real part to handle numerical precision

    if min_eigenval <= tolerance:
        max_eigenval = jnp.max(jnp.real(eigenvals))
        safe_min_eigenval = jnp.maximum(min_eigenval, tolerance)
        condition_number = max_eigenval / safe_min_eigenval
        raise NumericalStabilityError(
            "Matrix is not positive definite",
            condition_number=float(jnp.real(condition_number)),
            matrix_norm=float(jnp.linalg.norm(matrix)),
            recommended_action="Add regularization or check input data",
        )


def check_numerical_stability(
    matrix: Array, operation: str, max_condition: float = 1e12, tolerance: float = 1e-12
) -> None:
    """Check numerical stability of a matrix for a given operation.

    Args:
        matrix: Matrix to check
        operation: Name of operation for error reporting
        max_condition: Maximum acceptable condition number
        tolerance: Tolerance for stability checks

    Raises:
        NumericalStabilityError: If matrix is numerically unstable
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return  # Only check square matrices

    # Compute condition number with proper handling of numerical issues
    try:
        condition_number = jnp.linalg.cond(matrix)
        # Handle infinite or NaN condition numbers
        if jnp.isinf(condition_number) or jnp.isnan(condition_number):
            condition_number = 1e16  # Very large number to indicate ill-conditioning
        condition_number = float(jnp.real(condition_number))
    except Exception:
        # Fallback: assume ill-conditioned if computation fails
        condition_number = 1e16

    if condition_number > max_condition:
        raise NumericalStabilityError(
            f"Matrix is ill-conditioned for operation '{operation}'",
            condition_number=condition_number,
            matrix_norm=float(jnp.linalg.norm(matrix)),
            recommended_action=f"Consider regularization or alternative algorithm for {operation}",
        )


def validate_dimensions_match(arrays: list[Array], operation: str, axis: int | None = None) -> None:
    """Validate that arrays have compatible dimensions for an operation.

    Args:
        arrays: List of arrays to check
        operation: Name of operation for error reporting
        axis: Specific axis to check (None for all dimensions)

    Raises:
        DimensionError: If dimensions don't match
    """
    if len(arrays) < 2:
        return

    reference_shape = arrays[0].shape

    for i, array in enumerate(arrays[1:], 1):
        if axis is None:
            if array.shape != reference_shape:
                raise DimensionError(
                    f"Shape mismatch in {operation} at array {i}", expected=reference_shape, actual=array.shape
                )
        else:
            if array.shape[axis] != reference_shape[axis]:
                raise DimensionError(
                    f"Dimension mismatch in {operation} at array {i}, axis {axis}",
                    expected=reference_shape[axis],
                    actual=array.shape[axis],
                )
