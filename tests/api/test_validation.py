"""Tests for error handling and validation framework."""

import pytest
import jax.numpy as jnp

from riemannax.api.errors import (
    ManifoldDetectionError,
    ConstraintViolationError,
    ParameterValidationError,
)
from riemannax.api.validation import (
    validate_sphere_constraint,
    validate_orthogonal_constraint,
    validate_spd_constraint,
    validate_parameter_type,
    validate_learning_rate,
    ValidationResult,
)


class TestValidationFunctions:
    """Test validation functions for manifold constraints."""

    def test_validate_sphere_constraint_valid(self):
        """Test sphere constraint validation with valid unit vectors."""
        # Valid unit vector
        x = jnp.array([1.0, 0.0])
        result = validate_sphere_constraint(x)
        assert result.is_valid == True
        assert len(result.violations) == 0

        # Another valid unit vector
        x = jnp.array([0.6, 0.8])
        result = validate_sphere_constraint(x)
        assert result.is_valid == True
        assert len(result.violations) == 0

    def test_validate_sphere_constraint_invalid(self):
        """Test sphere constraint validation with invalid vectors."""
        # Non-unit vector
        x = jnp.array([2.0, 0.0])
        result = validate_sphere_constraint(x)
        assert result.is_valid == False
        assert len(result.violations) > 0
        assert "unit norm" in result.violations[0].lower()

        # Zero vector
        x = jnp.array([0.0, 0.0])
        result = validate_sphere_constraint(x)
        assert result.is_valid == False
        assert len(result.violations) > 0

    def test_validate_orthogonal_constraint_valid(self):
        """Test orthogonal constraint validation with valid orthogonal matrices."""
        # Identity matrix
        X = jnp.eye(2)
        result = validate_orthogonal_constraint(X)
        assert result.is_valid == True
        assert len(result.violations) == 0

        # Rotation matrix
        theta = jnp.pi / 4
        X = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                       [jnp.sin(theta), jnp.cos(theta)]])
        result = validate_orthogonal_constraint(X)
        assert result.is_valid == True
        assert len(result.violations) == 0

    def test_validate_orthogonal_constraint_invalid(self):
        """Test orthogonal constraint validation with invalid matrices."""
        # Non-square matrix
        X = jnp.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0]])
        result = validate_orthogonal_constraint(X)
        assert result.is_valid == False
        assert len(result.violations) > 0

        # Non-orthogonal square matrix
        X = jnp.array([[2.0, 0.0],
                       [0.0, 1.0]])
        result = validate_orthogonal_constraint(X)
        assert result.is_valid == False
        assert len(result.violations) > 0
        assert "orthogonal" in result.violations[0].lower()

    def test_validate_spd_constraint_valid(self):
        """Test SPD constraint validation with valid SPD matrices."""
        # Simple SPD matrix
        X = jnp.array([[2.0, 1.0],
                       [1.0, 2.0]])
        result = validate_spd_constraint(X)
        assert result.is_valid == True
        assert len(result.violations) == 0

        # Identity matrix
        X = jnp.eye(3)
        result = validate_spd_constraint(X)
        assert result.is_valid == True
        assert len(result.violations) == 0

    def test_validate_spd_constraint_invalid(self):
        """Test SPD constraint validation with invalid matrices."""
        # Non-symmetric matrix
        X = jnp.array([[1.0, 2.0],
                       [0.0, 1.0]])
        result = validate_spd_constraint(X)
        assert result.is_valid == False
        assert len(result.violations) > 0
        assert "symmetric" in result.violations[0].lower()

        # Negative definite matrix
        X = jnp.array([[-1.0, 0.0],
                       [0.0, -1.0]])
        result = validate_spd_constraint(X)
        assert result.is_valid == False
        assert len(result.violations) > 0
        assert "positive definite" in result.violations[0].lower()

    def test_validate_parameter_type(self):
        """Test parameter type validation."""
        # Valid float
        result = validate_parameter_type(0.5, float, "learning_rate")
        assert result.is_valid == True

        # Valid int
        result = validate_parameter_type(100, int, "max_iterations")
        assert result.is_valid == True

        # Invalid type
        result = validate_parameter_type("invalid", float, "learning_rate")
        assert result.is_valid == False
        assert "learning_rate" in result.violations[0]
        assert "float" in result.violations[0]

    def test_validate_learning_rate(self):
        """Test learning rate validation."""
        # Valid learning rate
        result = validate_learning_rate(0.01)
        assert result.is_valid == True

        # Invalid negative learning rate
        result = validate_learning_rate(-0.01)
        assert result.is_valid == False
        assert "positive" in result.violations[0].lower()

        # Invalid zero learning rate
        result = validate_learning_rate(0.0)
        assert result.is_valid == False
        assert "positive" in result.violations[0].lower()


class TestValidationResult:
    """Test ValidationResult class functionality."""

    def test_validation_result_creation(self):
        """Test ValidationResult creation and basic functionality."""
        violations = ["Test violation message"]
        suggestions = ["Test suggestion"]

        result = ValidationResult(
            is_valid=False,
            violations=violations,
            suggestions=suggestions
        )

        assert result.is_valid == False
        assert result.violations == violations
        assert result.suggestions == suggestions

    def test_validation_result_default_values(self):
        """Test ValidationResult with default values."""
        result = ValidationResult(is_valid=True)
        assert result.is_valid == True
        assert result.violations == []
        assert result.suggestions == []


class TestCustomExceptions:
    """Test custom exception classes."""

    def test_manifold_detection_error(self):
        """Test ManifoldDetectionError exception."""
        with pytest.raises(ManifoldDetectionError) as exc_info:
            raise ManifoldDetectionError("Could not detect manifold type")

        assert "Could not detect manifold type" in str(exc_info.value)

    def test_constraint_violation_error(self):
        """Test ConstraintViolationError exception."""
        with pytest.raises(ConstraintViolationError) as exc_info:
            raise ConstraintViolationError("Constraint violated",
                                         constraint_type="sphere",
                                         suggestions=["Normalize the vector"])

        assert "Constraint violated" in str(exc_info.value)
        assert exc_info.value.constraint_type == "sphere"
        assert "Normalize the vector" in exc_info.value.suggestions

    def test_parameter_validation_error(self):
        """Test ParameterValidationError exception."""
        with pytest.raises(ParameterValidationError) as exc_info:
            raise ParameterValidationError("Invalid parameter",
                                         parameter_name="learning_rate",
                                         expected_type=float,
                                         received_value="invalid")

        assert "Invalid parameter" in str(exc_info.value)
        assert exc_info.value.parameter_name == "learning_rate"
        assert exc_info.value.expected_type == float
        assert exc_info.value.received_value == "invalid"
