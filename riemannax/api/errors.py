"""Custom exception classes for RiemannAX high-level APIs."""

from typing import Any


class RiemannAXAPIError(Exception):
    """Base exception class for RiemannAX high-level API errors."""

    pass


class ManifoldDetectionError(RiemannAXAPIError):
    """Exception raised when automatic manifold detection fails."""

    def __init__(self, message: str, alternatives: list[str] | None = None):
        """Initialize ManifoldDetectionError.

        Args:
            message: Error description.
            alternatives: List of suggested manifold types.
        """
        super().__init__(message)
        self.alternatives = alternatives or []


class ConstraintViolationError(RiemannAXAPIError):
    """Exception raised when manifold constraints are violated."""

    def __init__(
        self,
        message: str,
        constraint_type: str | None = None,
        suggestions: list[str] | None = None,
    ):
        """Initialize ConstraintViolationError.

        Args:
            message: Error description.
            constraint_type: Type of constraint that was violated.
            suggestions: List of suggested corrections.
        """
        super().__init__(message)
        self.constraint_type = constraint_type
        self.suggestions = suggestions or []


class ParameterValidationError(RiemannAXAPIError):
    """Exception raised when parameter validation fails."""

    def __init__(
        self,
        message: str,
        parameter_name: str | None = None,
        expected_type: type | None = None,
        received_value: Any = None,
    ):
        """Initialize ParameterValidationError.

        Args:
            message: Error description.
            parameter_name: Name of the invalid parameter.
            expected_type: Expected parameter type.
            received_value: The actual received value.
        """
        super().__init__(message)
        self.parameter_name = parameter_name
        self.expected_type = expected_type
        self.received_value = received_value
