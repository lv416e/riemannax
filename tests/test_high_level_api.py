"""Tests for high-level API estimators."""

import jax.numpy as jnp
import pytest
from sklearn.base import BaseEstimator

from riemannax.api.estimators import RiemannianSGD, RiemannianAdam


class TestBaseEstimatorFramework:
    """Tests for Task 1.1 - Base Estimator Framework."""

    def test_riemannian_sgd_inherits_from_base_estimator(self):
        """Test that RiemannianSGD inherits from sklearn BaseEstimator."""
        estimator = RiemannianSGD(manifold="sphere", learning_rate=0.01)
        assert isinstance(estimator, BaseEstimator)

    def test_riemannian_sgd_init_with_string_manifold(self):
        """Test RiemannianSGD can be initialized with string manifold specification."""
        estimator = RiemannianSGD(manifold="sphere", learning_rate=0.01)
        assert estimator.manifold == "sphere"
        assert estimator.learning_rate == 0.01

    def test_riemannian_sgd_get_params(self):
        """Test get_params method returns correct parameters."""
        estimator = RiemannianSGD(manifold="stiefel", learning_rate=0.05, max_iterations=200)
        params = estimator.get_params()

        expected_params = {
            "manifold": "stiefel",
            "learning_rate": 0.05,
            "max_iterations": 200,
            "tolerance": 1e-6,  # default value
            "random_state": None,  # default value
        }
        assert params == expected_params

    def test_riemannian_sgd_set_params(self):
        """Test set_params method updates parameters correctly."""
        estimator = RiemannianSGD(manifold="sphere")
        new_estimator = estimator.set_params(learning_rate=0.001, max_iterations=50)

        assert new_estimator is estimator  # should return self
        assert estimator.learning_rate == 0.001
        assert estimator.max_iterations == 50
        assert estimator.manifold == "sphere"  # unchanged

    def test_riemannian_sgd_invalid_manifold_raises_error(self):
        """Test that invalid manifold specification raises descriptive error."""
        with pytest.raises(ValueError, match="Unsupported manifold type"):
            RiemannianSGD(manifold="invalid_manifold")

    def test_riemannian_sgd_fit_returns_self(self):
        """Test that fit method returns self for method chaining."""
        estimator = RiemannianSGD(manifold="sphere", learning_rate=0.01)

        def sphere_objective(x):
            return jnp.sum(x**2)

        x0 = jnp.array([1.0, 0.0, 0.0])  # point on unit sphere
        result = estimator.fit(sphere_objective, x0)

        assert result is estimator

    def test_riemannian_sgd_fit_stores_result(self):
        """Test that fit method stores optimization result."""
        estimator = RiemannianSGD(manifold="sphere", learning_rate=0.01, max_iterations=10)

        def sphere_objective(x):
            return jnp.sum((x - jnp.array([0.0, 0.0, 1.0]))**2)

        x0 = jnp.array([1.0, 0.0, 0.0])  # point on unit sphere
        estimator.fit(sphere_objective, x0)

        # Check that result is stored
        assert hasattr(estimator, 'optimization_result_')
        assert hasattr(estimator.optimization_result_, 'x')
        assert hasattr(estimator.optimization_result_, 'fun')
        assert hasattr(estimator.optimization_result_, 'success')
        assert hasattr(estimator.optimization_result_, 'niter')

    def test_riemannian_adam_basic_functionality(self):
        """Test RiemannianAdam basic functionality."""
        estimator = RiemannianAdam(manifold="grassmann", learning_rate=0.001)
        assert estimator.manifold == "grassmann"
        assert estimator.learning_rate == 0.001
        assert isinstance(estimator, BaseEstimator)

    def test_all_supported_manifolds_work(self):
        """Test that all supported manifold types can be instantiated."""
        supported_manifolds = ["sphere", "stiefel", "grassmann", "so", "spd"]

        for manifold_type in supported_manifolds:
            estimator = RiemannianSGD(manifold=manifold_type)
            assert estimator.manifold == manifold_type

    def test_estimator_unfitted_state(self):
        """Test estimator behavior before fitting."""
        estimator = RiemannianSGD(manifold="sphere")

        # Should not have result before fitting
        assert not hasattr(estimator, 'optimization_result_')

        # Should be able to check if fitted
        assert not estimator._is_fitted()

    def test_estimator_fitted_state(self):
        """Test estimator behavior after fitting."""
        estimator = RiemannianSGD(manifold="sphere", max_iterations=5)

        def objective(x):
            return jnp.sum(x**2)

        x0 = jnp.array([1.0, 0.0, 0.0])
        estimator.fit(objective, x0)

        # Should have result after fitting
        assert hasattr(estimator, 'optimization_result_')
        assert estimator._is_fitted()

    def test_default_parameters(self):
        """Test that default parameters are set correctly."""
        estimator = RiemannianSGD(manifold="sphere")

        assert estimator.learning_rate == 0.1  # default
        assert estimator.max_iterations == 100  # default
        assert estimator.tolerance == 1e-6  # default
        assert estimator.random_state is None  # default

    def test_parameter_validation(self):
        """Test parameter validation in constructor."""
        # Test negative learning rate
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            RiemannianSGD(manifold="sphere", learning_rate=-0.1)

        # Test non-positive max_iterations
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            RiemannianSGD(manifold="sphere", max_iterations=0)

        # Test negative tolerance
        with pytest.raises(ValueError, match="tolerance must be non-negative"):
            RiemannianSGD(manifold="sphere", tolerance=-1e-6)
