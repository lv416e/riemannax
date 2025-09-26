"""Validation functions for RiemannAX high-level APIs."""

import dataclasses
from typing import Any

import jax.numpy as jnp
from jaxtyping import Array


@dataclasses.dataclass
class ValidationResult:
    """Result of a validation operation.

    Attributes:
        is_valid: Whether the validation passed.
        violations: List of validation violation messages.
        suggestions: List of suggested corrections.
    """

    is_valid: bool
    violations: list[str] = dataclasses.field(default_factory=list)
    suggestions: list[str] = dataclasses.field(default_factory=list)


def validate_sphere_constraint(x: Array, atol: float = 1e-6) -> ValidationResult:
    """Validate that x satisfies sphere manifold constraints (unit norm).

    Args:
        x: Array to validate.
        atol: Absolute tolerance for validation.

    Returns:
        ValidationResult indicating whether x is on the unit sphere.
    """
    violations = []
    suggestions = []

    # Check if the norm is approximately 1
    norm = jnp.linalg.norm(x)
    is_unit = jnp.allclose(norm, 1.0, atol=atol)

    if not is_unit:
        violations.append(f"Vector must have unit norm, got norm={float(norm):.6f}")
        suggestions.append("Normalize the vector using x / ||x||")

    return ValidationResult(is_valid=bool(is_unit), violations=violations, suggestions=suggestions)


def validate_orthogonal_constraint(X: Array, atol: float = 1e-6) -> ValidationResult:
    """Validate that X satisfies orthogonal manifold constraints (X^T X = I).

    Args:
        X: Matrix to validate.
        atol: Absolute tolerance for validation.

    Returns:
        ValidationResult indicating whether X is orthogonal.
    """
    violations = []
    suggestions = []

    # Check if matrix is square (for Stiefel, can be rectangular)
    m, n = X.shape

    XTX = X.T @ X
    I = jnp.eye(n)
    is_orthogonal = jnp.allclose(XTX, I, atol=atol)

    if not is_orthogonal:
        message = (
            "Matrix must be orthogonal (X^T X = I)" if m == n else "Matrix columns must be orthonormal (X^T X = I)"
        )
        violations.append(message)
        suggestions.append("Use QR decomposition: X, _ = jax.scipy.linalg.qr(X)")

    return ValidationResult(is_valid=len(violations) == 0, violations=violations, suggestions=suggestions)


def validate_spd_constraint(X: Array, atol: float = 1e-6) -> ValidationResult:
    """Validate that X satisfies SPD manifold constraints (symmetric positive definite).

    Args:
        X: Matrix to validate.
        atol: Absolute tolerance for validation.

    Returns:
        ValidationResult indicating whether X is SPD.
    """
    violations: list[str] = []
    suggestions: list[str] = []

    # Check if matrix is square
    m, n = X.shape
    if m != n:
        violations.append(f"Matrix must be square, got shape {X.shape}")
        return ValidationResult(is_valid=False, violations=violations, suggestions=suggestions)

    # Check symmetry
    is_symmetric = jnp.allclose(X, X.T, atol=atol)
    if not is_symmetric:
        violations.append("Matrix must be symmetric (X = X^T)")
        suggestions.append("Symmetrize using X = (X + X^T) / 2")
    else:
        # Check positive definiteness via eigenvalues only if symmetric
        try:
            # Use eigvalsh for symmetric matrices: it's more stable and guarantees real eigenvalues.
            eigenvals = jnp.linalg.eigvalsh(X)
            min_eigenval = jnp.min(eigenvals)
            is_positive_definite = min_eigenval > atol

            if not is_positive_definite:
                violations.append(f"Matrix must be positive definite, minimum eigenvalue={float(min_eigenval):.6f}")
                suggestions.append("Add regularization: X + ε*I where ε > 0")
        except (ValueError, RuntimeError):
            violations.append("Could not compute eigenvalues for positive definiteness check")
            suggestions.append("Ensure matrix is numerically stable")

    return ValidationResult(is_valid=len(violations) == 0, violations=violations, suggestions=suggestions)


def validate_parameter_type(value: Any, expected_type: type, parameter_name: str) -> ValidationResult:
    """Validate that a parameter has the expected type.

    Args:
        value: Parameter value to validate.
        expected_type: Expected type.
        parameter_name: Name of the parameter for error messages.

    Returns:
        ValidationResult indicating type validity.
    """
    violations = []
    suggestions = []

    # Explicitly reject booleans for numeric types, as `isinstance(True, int)` is `True`.
    is_valid = False if isinstance(value, bool) and expected_type in (int, float) else isinstance(value, expected_type)

    if not is_valid:
        violations.append(
            f"Parameter '{parameter_name}' must be of type {expected_type.__name__}, got {type(value).__name__}"
        )
        suggestions.append(f"Convert to {expected_type.__name__}: {expected_type.__name__}(value)")

    return ValidationResult(is_valid=len(violations) == 0, violations=violations, suggestions=suggestions)


def validate_learning_rate(learning_rate: float) -> ValidationResult:
    """Validate learning rate parameter.

    Args:
        learning_rate: Learning rate value to validate.

    Returns:
        ValidationResult indicating learning rate validity.
    """
    violations = []
    suggestions = []

    # Use tuple form for isinstance to avoid TypeError with union syntax
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
        violations.append(f"Learning rate must be numeric, got {type(learning_rate).__name__}")
        suggestions.append("Use a float value like 0.01")
        return ValidationResult(is_valid=False, violations=violations, suggestions=suggestions)

    if learning_rate <= 0:
        violations.append(f"Learning rate must be positive, got {learning_rate}")
        suggestions.append("Use a small positive value like 0.01 or 0.1")

    if learning_rate > 1.0:
        violations.append(f"Learning rate {learning_rate} is unusually large")
        suggestions.append("Consider using a smaller value like 0.01-0.1 for stability")

    return ValidationResult(is_valid=len(violations) == 0, violations=violations, suggestions=suggestions)
