"""Tests for RiemannianEstimator base class.

This module tests the base estimator framework that provides scikit-learn
compatible interfaces for Riemannian optimization.

Requirements Coverage:
- R1.1: String manifold specification constructor
- R1.3: get_params()/set_params() methods
- R1.4: Error handling for invalid manifold specifications
- R8.1: Specific exception types with detailed error messages
- R8.2: Constraint violation detection
"""

import pytest
from unittest.mock import Mock, patch
import jax.numpy as jnp
from jax import random

# Test imports - these will fail initially (RED phase)
from riemannax.high_level_api.base import RiemannianEstimator
from riemannax.high_level_api.exceptions import (
    InvalidManifoldError,
    ParameterValidationError
)


class TestRiemannianEstimatorInterface:
    """Test scikit-learn compatible interface."""

    def test_inherits_from_sklearn_base_estimator(self):
        """Test that RiemannianEstimator inherits from sklearn BaseEstimator."""
        from sklearn.base import BaseEstimator

        estimator = RiemannianEstimator(manifold="sphere")
        assert isinstance(estimator, BaseEstimator)

    def test_implements_get_params_method(self):
        """Test that get_params() method exists and works correctly."""
        estimator = RiemannianEstimator(manifold="sphere", lr=0.01)

        params = estimator.get_params()
        assert isinstance(params, dict)
        assert "manifold" in params
        assert "lr" in params
        assert params["manifold"] == "sphere"
        assert params["lr"] == 0.01

    def test_implements_set_params_method(self):
        """Test that set_params() method exists and works correctly."""
        estimator = RiemannianEstimator(manifold="sphere", lr=0.01)

        # Test setting parameters
        estimator.set_params(lr=0.02, manifold="grassmann")

        params = estimator.get_params()
        assert params["lr"] == 0.02
        assert params["manifold"] == "grassmann"

    def test_set_params_returns_self(self):
        """Test that set_params() returns self for method chaining."""
        estimator = RiemannianEstimator(manifold="sphere")
        result = estimator.set_params(lr=0.01)
        assert result is estimator

    def test_get_params_deep_parameter(self):
        """Test get_params() with deep parameter."""
        estimator = RiemannianEstimator(manifold="sphere", lr=0.01)

        # Test deep=True (default)
        params_deep = estimator.get_params(deep=True)
        params_shallow = estimator.get_params(deep=False)

        # For simple estimator without sub-estimators, should be same
        assert params_deep == params_shallow


class TestStringManifoldSpecification:
    """Test string manifold specification functionality (R1.1)."""

    def test_accepts_valid_sphere_manifold(self):
        """Test that sphere manifold string is accepted."""
        estimator = RiemannianEstimator(manifold="sphere")
        assert estimator.manifold == "sphere"

    def test_accepts_valid_grassmann_manifold(self):
        """Test that grassmann manifold string is accepted."""
        estimator = RiemannianEstimator(manifold="grassmann")
        assert estimator.manifold == "grassmann"

    def test_accepts_valid_stiefel_manifold(self):
        """Test that stiefel manifold string is accepted."""
        estimator = RiemannianEstimator(manifold="stiefel")
        assert estimator.manifold == "stiefel"

    def test_accepts_valid_spd_manifold(self):
        """Test that spd manifold string is accepted."""
        estimator = RiemannianEstimator(manifold="spd")
        assert estimator.manifold == "spd"

    def test_accepts_valid_so_manifold(self):
        """Test that so manifold string is accepted."""
        estimator = RiemannianEstimator(manifold="so")
        assert estimator.manifold == "so"

    def test_stores_manifold_parameters_correctly(self):
        """Test that manifold parameters are stored correctly."""
        estimator = RiemannianEstimator(
            manifold="sphere",
            lr=0.01,
            max_iter=100,
            tol=1e-6
        )

        assert estimator.manifold == "sphere"
        assert estimator.lr == 0.01
        assert estimator.max_iter == 100
        assert estimator.tol == 1e-6


class TestInvalidManifoldHandling:
    """Test error handling for invalid manifold specifications (R1.4, R8.1)."""

    def test_raises_error_for_invalid_manifold_string(self):
        """Test that invalid manifold raises InvalidManifoldError."""
        with pytest.raises(InvalidManifoldError) as exc_info:
            RiemannianEstimator(manifold="invalid_manifold")

        error_msg = str(exc_info.value)
        assert "invalid_manifold" in error_msg
        assert "Available manifolds:" in error_msg

    def test_raises_error_for_none_manifold(self):
        """Test that None manifold raises appropriate error."""
        with pytest.raises(InvalidManifoldError) as exc_info:
            RiemannianEstimator(manifold=None)

        error_msg = str(exc_info.value)
        assert "None" in error_msg or "null" in error_msg.lower()

    def test_raises_error_for_numeric_manifold(self):
        """Test that numeric manifold raises appropriate error."""
        with pytest.raises(InvalidManifoldError) as exc_info:
            RiemannianEstimator(manifold=123)

        error_msg = str(exc_info.value)
        assert "string" in error_msg.lower()

    def test_error_message_suggests_available_manifolds(self):
        """Test that error message includes available manifold options."""
        with pytest.raises(InvalidManifoldError) as exc_info:
            RiemannianEstimator(manifold="typo_sphere")

        error_msg = str(exc_info.value)
        assert "sphere" in error_msg
        assert "grassmann" in error_msg
        assert "stiefel" in error_msg
        assert "spd" in error_msg
        assert "so" in error_msg

    def test_error_message_is_descriptive(self):
        """Test that error messages are descriptive and helpful."""
        with pytest.raises(InvalidManifoldError) as exc_info:
            RiemannianEstimator(manifold="unknown")

        error_msg = str(exc_info.value)
        assert len(error_msg) > 50  # Should be descriptive
        assert "unknown" in error_msg  # Should mention the invalid input


class TestParameterValidation:
    """Test parameter validation and error reporting (R8.1, R8.2)."""

    def test_validates_learning_rate_parameter(self):
        """Test that learning rate is validated."""
        # Valid learning rate should work
        estimator = RiemannianEstimator(manifold="sphere", lr=0.01)
        assert estimator.lr == 0.01

        # Invalid learning rate should raise error
        with pytest.raises(ParameterValidationError) as exc_info:
            RiemannianEstimator(manifold="sphere", lr=-0.01)

        assert "learning rate" in str(exc_info.value).lower()
        assert "positive" in str(exc_info.value).lower()

    def test_validates_max_iter_parameter(self):
        """Test that max_iter is validated."""
        # Valid max_iter should work
        estimator = RiemannianEstimator(manifold="sphere", max_iter=100)
        assert estimator.max_iter == 100

        # Invalid max_iter should raise error
        with pytest.raises(ParameterValidationError) as exc_info:
            RiemannianEstimator(manifold="sphere", max_iter=0)

        assert "max_iter" in str(exc_info.value)
        assert "positive" in str(exc_info.value).lower()

    def test_validates_tolerance_parameter(self):
        """Test that tolerance is validated."""
        # Valid tolerance should work
        estimator = RiemannianEstimator(manifold="sphere", tol=1e-6)
        assert estimator.tol == 1e-6

        # Invalid tolerance should raise error
        with pytest.raises(ParameterValidationError) as exc_info:
            RiemannianEstimator(manifold="sphere", tol=-1e-6)

        assert "tolerance" in str(exc_info.value).lower()
        assert "positive" in str(exc_info.value).lower()

    def test_parameter_validation_in_set_params(self):
        """Test that set_params also validates parameters."""
        estimator = RiemannianEstimator(manifold="sphere", lr=0.01)

        # Valid parameter update should work
        estimator.set_params(lr=0.02)
        assert estimator.lr == 0.02

        # Invalid parameter update should raise error
        with pytest.raises(ParameterValidationError):
            estimator.set_params(lr=-0.01)

    def test_manifold_validation_in_set_params(self):
        """Test that set_params validates manifold changes."""
        estimator = RiemannianEstimator(manifold="sphere")

        # Valid manifold change should work
        estimator.set_params(manifold="grassmann")
        assert estimator.manifold == "grassmann"

        # Invalid manifold change should raise error
        with pytest.raises(InvalidManifoldError):
            estimator.set_params(manifold="invalid")


class TestConsistentErrorReporting:
    """Test consistent parameter validation and error reporting patterns."""

    def test_error_message_format_consistency(self):
        """Test that all error messages follow consistent format."""
        test_cases = [
            ({"manifold": "invalid"}, InvalidManifoldError),
            ({"manifold": "sphere", "lr": -0.01}, ParameterValidationError),
            ({"manifold": "sphere", "max_iter": 0}, ParameterValidationError),
            ({"manifold": "sphere", "tol": -1e-6}, ParameterValidationError),
        ]

        for params, expected_error in test_cases:
            with pytest.raises(expected_error) as exc_info:
                RiemannianEstimator(**params)

            error_msg = str(exc_info.value)
            # All error messages should be reasonably long and descriptive
            assert len(error_msg) > 20
            # Should not be generic Python error
            assert "object" not in error_msg.lower()

    def test_error_types_are_specific(self):
        """Test that specific error types are raised for different validation failures."""
        # Manifold errors should raise InvalidManifoldError
        with pytest.raises(InvalidManifoldError):
            RiemannianEstimator(manifold="invalid")

        # Parameter errors should raise ParameterValidationError
        with pytest.raises(ParameterValidationError):
            RiemannianEstimator(manifold="sphere", lr=-0.01)

    def test_all_parameters_are_stored_as_attributes(self):
        """Test that all constructor parameters are stored as attributes."""
        params = {
            "manifold": "sphere",
            "lr": 0.01,
            "max_iter": 100,
            "tol": 1e-6,
        }

        estimator = RiemannianEstimator(**params)

        for param_name, param_value in params.items():
            assert hasattr(estimator, param_name)
            assert getattr(estimator, param_name) == param_value


class TestDefaultParameters:
    """Test default parameter values."""

    def test_has_sensible_default_lr(self):
        """Test that learning rate has a sensible default."""
        estimator = RiemannianEstimator(manifold="sphere")
        assert hasattr(estimator, "lr")
        assert estimator.lr > 0
        assert estimator.lr <= 1.0  # Should be reasonable

    def test_has_sensible_default_max_iter(self):
        """Test that max_iter has a sensible default."""
        estimator = RiemannianEstimator(manifold="sphere")
        assert hasattr(estimator, "max_iter")
        assert estimator.max_iter > 0
        assert estimator.max_iter <= 10000  # Should be reasonable

    def test_has_sensible_default_tol(self):
        """Test that tolerance has a sensible default."""
        estimator = RiemannianEstimator(manifold="sphere")
        assert hasattr(estimator, "tol")
        assert estimator.tol > 0
        assert estimator.tol <= 1e-3  # Should be reasonable

    def test_can_override_all_defaults(self):
        """Test that all default parameters can be overridden."""
        custom_params = {
            "manifold": "grassmann",
            "lr": 0.05,
            "max_iter": 500,
            "tol": 1e-8,
        }

        estimator = RiemannianEstimator(**custom_params)

        for param_name, param_value in custom_params.items():
            assert getattr(estimator, param_name) == param_value
