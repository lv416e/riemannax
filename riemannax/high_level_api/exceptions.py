"""Exception classes for high-level API error handling.

This module defines specific exception types for clear error reporting
and validation failures in the high-level API.

Requirements Coverage:
- R8.1: Specific exception types with detailed error messages
- R8.2: Constraint violation detection
"""

from typing import Any


class RiemannianOptimizationError(Exception):
    """Base exception class for all RiemannAX high-level API errors."""

    pass


class InvalidManifoldError(RiemannianOptimizationError):
    """Raised when an invalid manifold specification is provided.

    This error is raised when:
    - An unsupported manifold string is provided
    - A non-string manifold specification is given
    - A None manifold is specified

    Requirements Coverage: R1.4, R8.1
    """

    def __init__(self, manifold_spec: Any, available_manifolds: list[str] | None = None):
        """Initialize InvalidManifoldError with helpful suggestions.

        Args:
            manifold_spec: The invalid manifold specification provided
            available_manifolds: List of valid manifold names for suggestions
        """
        if available_manifolds is None:
            available_manifolds = ["sphere", "grassmann", "stiefel", "spd", "so"]

        if manifold_spec is None:
            message = (
                "Manifold specification cannot be None. "
                f"Please specify one of the available manifolds: {', '.join(available_manifolds)}"
            )
        elif not isinstance(manifold_spec, str):
            message = (
                f"Manifold specification must be a string, got {type(manifold_spec).__name__}: {manifold_spec}. "
                f"Available manifolds: {', '.join(available_manifolds)}"
            )
        else:
            message = (
                f"Invalid manifold specification: '{manifold_spec}'. "
                f"Available manifolds: {', '.join(available_manifolds)}. "
                f"Did you mean one of these: {', '.join(available_manifolds)}?"
            )

        super().__init__(message)
        self.manifold_spec = manifold_spec
        self.available_manifolds = available_manifolds


class ParameterValidationError(RiemannianOptimizationError):
    """Raised when parameter validation fails.

    This error is raised when:
    - Learning rate is not positive
    - Maximum iterations is not positive
    - Tolerance is not positive
    - Other parameter constraints are violated

    Requirements Coverage: R8.1, R8.2
    """

    def __init__(self, parameter_name: str, parameter_value: Any, constraint: str):
        """Initialize ParameterValidationError with detailed information.

        Args:
            parameter_name: Name of the invalid parameter
            parameter_value: Value that failed validation
            constraint: Description of the constraint that was violated
        """
        message = (
            f"Parameter '{parameter_name}' with value {parameter_value} violates constraint: {constraint}. "
            f"Please provide a valid value that satisfies the requirement."
        )

        super().__init__(message)
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value
        self.constraint = constraint


class ConstraintViolationError(RiemannianOptimizationError):
    """Raised when manifold constraints are violated during optimization.

    This error is raised when:
    - Points are not on the specified manifold
    - Tangent vectors are not in the tangent space
    - Optimization steps violate manifold constraints

    Requirements Coverage: R8.2
    """

    def __init__(self, manifold_name: str, constraint_description: str, violation_details: str = ""):
        """Initialize ConstraintViolationError with diagnostic information.

        Args:
            manifold_name: Name of the manifold whose constraints were violated
            constraint_description: Description of the constraint that was violated
            violation_details: Additional details about the violation
        """
        message = f"Manifold constraint violation on {manifold_name}: {constraint_description}"
        if violation_details:
            message += f". Details: {violation_details}"

        super().__init__(message)
        self.manifold_name = manifold_name
        self.constraint_description = constraint_description
        self.violation_details = violation_details
