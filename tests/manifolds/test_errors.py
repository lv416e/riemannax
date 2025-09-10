"""Tests for enhanced manifold error hierarchy and type system.

This module tests the comprehensive error handling system for manifold operations,
including specialized exceptions for different types of geometric and numerical errors.
"""

import pytest
import jax.numpy as jnp
import numpy as np

from riemannax.manifolds.errors import (
    ManifoldError,
    DimensionError,
    ConvergenceError,
    NumericalStabilityError,
    InvalidPointError,
    InvalidTangentVectorError,
    ManifoldConstraintError,
    GeometricError,
    validate_manifold_point,
    validate_tangent_vector,
    validate_positive_definite,
    check_numerical_stability,
    validate_dimensions_match,
)
from riemannax.core.type_system import (
    ManifoldPoint,
    TangentVector,
    RiemannianMetric,
    validate_jaxtyping_annotation,
    ensure_array_dtype,
)


class TestManifoldErrorHierarchy:
    """Test comprehensive manifold error hierarchy."""

    def test_manifold_error_is_base_exception(self):
        """Test that ManifoldError is the base exception."""
        error = ManifoldError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_dimension_error_inheritance(self):
        """Test DimensionError inherits from ManifoldError."""
        error = DimensionError("Dimension mismatch", expected=3, actual=2)
        assert isinstance(error, ManifoldError)
        assert error.expected == 3
        assert error.actual == 2
        assert "expected=3" in str(error)
        assert "actual=2" in str(error)

    def test_convergence_error_with_iterations(self):
        """Test ConvergenceError with iteration information."""
        error = ConvergenceError("Algorithm failed to converge", max_iterations=100, final_error=1e-3, tolerance=1e-6)
        assert isinstance(error, ManifoldError)
        assert error.max_iterations == 100
        assert error.final_error == 1e-3
        assert error.tolerance == 1e-6

    def test_numerical_stability_error_with_condition_number(self):
        """Test NumericalStabilityError with numerical diagnostics."""
        error = NumericalStabilityError(
            "Matrix is ill-conditioned", condition_number=1e12, matrix_norm=1e8, recommended_action="Use regularization"
        )
        assert isinstance(error, ManifoldError)
        assert error.condition_number == 1e12
        assert error.matrix_norm == 1e8
        assert "regularization" in error.recommended_action

    def test_invalid_point_error_with_constraint_info(self):
        """Test InvalidPointError with constraint violation details."""
        point = jnp.array([[1.0, 0.5], [0.5, 2.0]])
        error = InvalidPointError(
            "Point not on manifold", point=point, violated_constraint="positive_definite", constraint_value=-0.1
        )
        assert isinstance(error, ManifoldError)
        assert jnp.allclose(error.point, point)
        assert error.violated_constraint == "positive_definite"
        assert error.constraint_value == -0.1

    def test_invalid_tangent_vector_error(self):
        """Test InvalidTangentVectorError with geometric information."""
        tangent = jnp.array([1.0, 2.0, 3.0])
        base_point = jnp.array([0.0, 0.0, 1.0])
        error = InvalidTangentVectorError(
            "Tangent vector not in tangent space",
            tangent_vector=tangent,
            base_point=base_point,
            orthogonality_error=0.5,
        )
        assert isinstance(error, ManifoldError)
        assert jnp.allclose(error.tangent_vector, tangent)
        assert jnp.allclose(error.base_point, base_point)
        assert error.orthogonality_error == 0.5

    def test_manifold_constraint_error_with_details(self):
        """Test ManifoldConstraintError with constraint details."""
        error = ManifoldConstraintError(
            "Orthogonality constraint violated",
            constraint_name="orthogonality",
            violation_magnitude=1e-10,
            tolerance=1e-12,
        )
        assert isinstance(error, ManifoldError)
        assert error.constraint_name == "orthogonality"
        assert error.violation_magnitude == 1e-10
        assert error.tolerance == 1e-12

    def test_geometric_error_with_operation_context(self):
        """Test GeometricError with operation context."""
        error = GeometricError(
            "Exponential map failed", operation="exp_map", manifold_type="Grassmann", step_size=0.1, norm_tangent=10.0
        )
        assert isinstance(error, ManifoldError)
        assert error.operation == "exp_map"
        assert error.manifold_type == "Grassmann"
        assert error.step_size == 0.1
        assert error.norm_tangent == 10.0


class TestValidationFunctions:
    """Test manifold validation utility functions."""

    def test_validate_manifold_point_valid_sphere_point(self):
        """Test validation of valid point on unit sphere."""
        point = jnp.array([0.6, 0.8, 0.0])
        # Should not raise any exception
        validate_manifold_point(point, manifold_type="sphere", tolerance=1e-10)

    def test_validate_manifold_point_invalid_sphere_point(self):
        """Test validation of invalid point on unit sphere."""
        point = jnp.array([1.0, 1.0, 0.0])  # Not unit norm
        with pytest.raises(InvalidPointError) as exc_info:
            validate_manifold_point(point, manifold_type="sphere", tolerance=1e-10)

        error = exc_info.value
        assert "unit norm" in str(error).lower()
        assert jnp.allclose(error.point, point)

    def test_validate_tangent_vector_valid(self):
        """Test validation of valid tangent vector."""
        base_point = jnp.array([1.0, 0.0, 0.0])
        tangent = jnp.array([0.0, 1.0, 0.0])
        # Should not raise exception for orthogonal tangent on sphere
        validate_tangent_vector(tangent, base_point, manifold_type="sphere")

    def test_validate_tangent_vector_invalid(self):
        """Test validation of invalid tangent vector."""
        base_point = jnp.array([1.0, 0.0, 0.0])
        tangent = jnp.array([1.0, 0.0, 0.0])  # Not orthogonal
        with pytest.raises(InvalidTangentVectorError) as exc_info:
            validate_tangent_vector(tangent, base_point, manifold_type="sphere")

        error = exc_info.value
        assert error.orthogonality_error > 0.5

    def test_validate_positive_definite_valid(self):
        """Test validation of positive definite matrix."""
        matrix = jnp.array([[2.0, 1.0], [1.0, 2.0]])
        # Should not raise exception
        validate_positive_definite(matrix, tolerance=1e-10)

    def test_validate_positive_definite_invalid(self):
        """Test validation of non-positive definite matrix."""
        matrix = jnp.array([[1.0, 2.0], [2.0, 1.0]])  # Indefinite
        with pytest.raises(NumericalStabilityError) as exc_info:
            validate_positive_definite(matrix, tolerance=1e-10)

        error = exc_info.value
        assert "positive definite" in str(error)
        assert error.condition_number is not None

    def test_check_numerical_stability_stable(self):
        """Test numerical stability check for well-conditioned matrix."""
        matrix = jnp.eye(3)
        # Should not raise exception
        check_numerical_stability(matrix, operation="matrix_operation")

    def test_check_numerical_stability_unstable(self):
        """Test numerical stability check for ill-conditioned matrix."""
        matrix = jnp.array([[1.0, 1.0], [1.0, 1.0]])  # Singular matrix
        with pytest.raises(NumericalStabilityError) as exc_info:
            check_numerical_stability(matrix, operation="matrix_operation", max_condition=1e8)

        error = exc_info.value
        assert error.condition_number > 1e8
        assert "ill-conditioned" in str(error)

    def test_validate_dimensions_match_valid(self):
        """Test dimension matching validation for valid arrays."""
        x = jnp.ones((3, 4))
        y = jnp.ones((3, 4))
        # Should not raise exception
        validate_dimensions_match([x, y], operation="addition")

    def test_validate_dimensions_match_invalid(self):
        """Test dimension matching validation for mismatched arrays."""
        x = jnp.ones((3, 4))
        y = jnp.ones((2, 4))
        with pytest.raises(DimensionError) as exc_info:
            validate_dimensions_match([x, y], operation="addition")

        error = exc_info.value
        assert error.expected == (3, 4)
        assert error.actual == (2, 4)


class TestEnhancedTypeSystem:
    """Test enhanced type system with jaxtyping."""

    def test_manifold_point_type_annotation(self):
        """Test ManifoldPoint type annotation validation."""
        point = jnp.array([1.0, 2.0, 3.0])
        assert validate_jaxtyping_annotation(point, ManifoldPoint)

    def test_tangent_vector_type_annotation(self):
        """Test TangentVector type annotation validation."""
        vector = jnp.array([0.1, 0.2, 0.3])
        assert validate_jaxtyping_annotation(vector, TangentVector)

    def test_riemannian_metric_type_annotation(self):
        """Test RiemannianMetric type annotation validation."""
        metric = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        assert validate_jaxtyping_annotation(metric, RiemannianMetric)

    def test_invalid_type_annotation(self):
        """Test validation failure for mismatched types."""
        scalar = 5.0  # Not an array
        assert not validate_jaxtyping_annotation(scalar, ManifoldPoint)

    def test_ensure_array_dtype_float32(self):
        """Test dtype enforcement for float32."""
        arr = jnp.array([1, 2, 3])  # int array
        result = ensure_array_dtype(arr, jnp.float32)
        assert result.dtype == jnp.float32
        assert jnp.allclose(result, jnp.array([1.0, 2.0, 3.0]))

    def test_ensure_array_dtype_complex64(self):
        """Test dtype enforcement for complex64."""
        arr = jnp.array([1.0, 2.0, 3.0])
        result = ensure_array_dtype(arr, jnp.complex64)
        assert result.dtype == jnp.complex64
        assert jnp.allclose(result, jnp.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j]))


class TestIntegrationWithExistingCode:
    """Test integration with existing manifold implementations."""

    def test_error_propagation_in_manifold_operations(self):
        """Test that errors propagate correctly in manifold operations."""
        # This test will verify that the new error system integrates
        # properly with existing manifold code

        invalid_point = jnp.array([2.0, 3.0, 4.0])  # Not unit norm

        # Should raise InvalidPointError when validating sphere point
        with pytest.raises(InvalidPointError):
            validate_manifold_point(invalid_point, manifold_type="sphere")

    def test_type_checking_integration(self):
        """Test that type checking works with manifold functions."""
        # Test that our enhanced type system works with real manifold operations
        point = jnp.array([1.0, 0.0, 0.0])
        tangent = jnp.array([0.0, 1.0, 0.0])

        # These should pass type validation
        assert validate_jaxtyping_annotation(point, ManifoldPoint)
        assert validate_jaxtyping_annotation(tangent, TangentVector)


if __name__ == "__main__":
    pytest.main([__file__])
