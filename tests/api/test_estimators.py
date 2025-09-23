"""Tests for scikit-learn compatible estimator framework."""

import pytest
import jax.numpy as jnp
import jax

from riemannax.api.estimators import (
    RiemannianEstimator,
    RiemannianSGD,
    RiemannianAdam,
)
from riemannax.api.results import OptimizationResult, ConvergenceStatus
from riemannax.api.detection import ManifoldType
from riemannax.api.errors import ParameterValidationError, ManifoldDetectionError


class TestRiemannianEstimator:
    """Test base RiemannianEstimator class functionality."""

    def test_estimator_initialization(self):
        """Test basic estimator initialization."""
        estimator = RiemannianSGD(manifold="sphere", learning_rate=0.01)

        assert estimator.manifold == "sphere"
        assert estimator.learning_rate == 0.01
        assert estimator.max_iterations == 100  # default
        assert estimator.tolerance == 1e-6  # default

    def test_get_params(self):
        """Test get_params method for hyperparameter access."""
        estimator = RiemannianSGD(
            manifold="stiefel",
            learning_rate=0.05,
            max_iterations=200,
            tolerance=1e-8
        )

        params = estimator.get_params()

        assert params["manifold"] == "stiefel"
        assert params["learning_rate"] == 0.05
        assert params["max_iterations"] == 200
        assert params["tolerance"] == 1e-8

    def test_get_params_deep(self):
        """Test get_params with deep=True."""
        estimator = RiemannianSGD(manifold="sphere", learning_rate=0.01)

        params_shallow = estimator.get_params(deep=False)
        params_deep = estimator.get_params(deep=True)

        # For this simple case, should be the same
        assert params_shallow == params_deep

    def test_set_params(self):
        """Test set_params method for hyperparameter modification."""
        estimator = RiemannianSGD(manifold="sphere", learning_rate=0.01)

        # Modify parameters
        estimator.set_params(learning_rate=0.1, max_iterations=50)

        assert estimator.learning_rate == 0.1
        assert estimator.max_iterations == 50
        assert estimator.manifold == "sphere"  # unchanged

    def test_set_params_invalid(self):
        """Test set_params with invalid parameters."""
        estimator = RiemannianSGD(manifold="sphere", learning_rate=0.01)

        with pytest.raises(ParameterValidationError):
            estimator.set_params(invalid_param="value")

    def test_estimator_not_fitted_initially(self):
        """Test that estimator is not fitted initially."""
        estimator = RiemannianSGD(manifold="sphere", learning_rate=0.01)

        assert hasattr(estimator, "_is_fitted")
        assert estimator._is_fitted == False

    def test_parameter_validation(self):
        """Test parameter validation during initialization."""
        # Valid parameters
        estimator = RiemannianSGD(manifold="sphere", learning_rate=0.01)
        assert estimator is not None

        # Invalid learning rate
        with pytest.raises(ParameterValidationError):
            RiemannianSGD(manifold="sphere", learning_rate=-0.01)

        # Invalid manifold
        with pytest.raises(ParameterValidationError):
            RiemannianSGD(manifold="invalid_manifold", learning_rate=0.01)

        # Invalid tolerance
        with pytest.raises(ParameterValidationError):
            RiemannianSGD(manifold="sphere", learning_rate=0.01, tolerance=-0.01)

        # Invalid random_state
        with pytest.raises(ParameterValidationError):
            RiemannianSGD(manifold="sphere", learning_rate=0.01, random_state="invalid")

    def test_so_manifold_det_constraint(self):
        """Test that SO(n) manifold maintains det=+1 constraint during optimization."""
        # Start with a valid SO(2) rotation matrix (det = +1)
        theta = jnp.pi / 6
        x0 = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                        [jnp.sin(theta), jnp.cos(theta)]])

        def objective_func(X):
            # Simple objective to minimize trace
            return jnp.trace(X)

        estimator = RiemannianSGD(
            manifold="so",
            learning_rate=0.01,
            max_iterations=5
        )

        result = estimator.fit(objective_func, x0)

        # Check that final result has det = +1 (within tolerance)
        final_det = float(jnp.linalg.det(result.optimized_params))
        assert jnp.allclose(final_det, 1.0, atol=1e-6), f"Expected det=+1, got det={final_det}"

        # Check that result is orthogonal
        X = result.optimized_params
        XTX = X.T @ X
        I = jnp.eye(X.shape[0])
        assert jnp.allclose(XTX, I, atol=1e-6), "Result should be orthogonal"

    def test_so_wrapper_det_projection(self):
        """Test that the SO wrapper correctly projects det=-1 to det=+1."""
        from riemannax.api.estimators import _SOManifoldWrapper

        # Create SO(2) wrapper
        so_manifold = _SOManifoldWrapper(n=2)

        # Test the determinant projection method directly
        reflection = jnp.array([[1.0, 0.0],
                               [0.0, -1.0]])  # det = -1

        projected = so_manifold._ensure_det_positive(reflection)
        final_det = float(jnp.linalg.det(projected))

        assert jnp.allclose(final_det, 1.0, atol=1e-10), f"Expected det=+1 after projection, got {final_det}"

        # Test that positive determinant matrices are unchanged
        rotation = jnp.array([[0.0, -1.0],
                             [1.0, 0.0]])  # det = +1

        unchanged = so_manifold._ensure_det_positive(rotation)
        assert jnp.allclose(unchanged, rotation, atol=1e-10), "Matrices with det=+1 should be unchanged"


class TestRiemannianSGD:
    """Test RiemannianSGD estimator functionality."""

    def test_fit_sphere_manifold(self):
        """Test fitting with sphere manifold."""
        # Define a simple objective function (distance from target)
        target = jnp.array([0.6, 0.8])
        def objective_func(x):
            return jnp.sum((x - target) ** 2)

        # Initial point on sphere
        x0 = jnp.array([1.0, 0.0])

        estimator = RiemannianSGD(
            manifold="sphere",
            learning_rate=0.1,
            max_iterations=10
        )

        result = estimator.fit(objective_func, x0)

        # Check result structure
        assert isinstance(result, OptimizationResult)
        assert result.convergence_status in [ConvergenceStatus.CONVERGED, ConvergenceStatus.MAX_ITERATIONS]
        assert result.iteration_count <= 10
        assert result.optimized_params.shape == x0.shape

        # Check estimator is fitted
        assert estimator._is_fitted

    def test_fit_auto_manifold_detection(self):
        """Test fitting with automatic manifold detection."""
        def objective_func(x):
            return jnp.sum(x ** 2)

        # Unit vector should be detected as sphere
        x0 = jnp.array([1.0, 0.0])

        estimator = RiemannianSGD(
            manifold="auto",
            learning_rate=0.1,
            max_iterations=5
        )

        result = estimator.fit(objective_func, x0)

        assert isinstance(result, OptimizationResult)
        assert estimator._detected_manifold_type == ManifoldType.SPHERE

    def test_fit_spd_manifold(self):
        """Test fitting with SPD manifold."""
        def objective_func(X):
            return jnp.trace(X)  # Simple trace minimization

        # SPD matrix
        x0 = jnp.array([[2.0, 1.0],
                        [1.0, 2.0]])

        estimator = RiemannianSGD(
            manifold="spd",
            learning_rate=0.01,
            max_iterations=5
        )

        result = estimator.fit(objective_func, x0)

        assert isinstance(result, OptimizationResult)
        assert result.optimized_params.shape == x0.shape

    def test_fit_invalid_initial_point(self):
        """Test fitting with invalid initial point for manifold."""
        def objective_func(x):
            return jnp.sum(x ** 2)

        # Non-unit vector for sphere manifold
        x0 = jnp.array([2.0, 0.0])

        estimator = RiemannianSGD(
            manifold="sphere",
            learning_rate=0.1,
            max_iterations=5
        )

        with pytest.raises(ManifoldDetectionError):
            estimator.fit(objective_func, x0)

    def test_fit_auto_detection_failure(self):
        """Test fitting when auto detection fails."""
        def objective_func(x):
            return jnp.sum(x ** 2)

        # Scalar - cannot detect manifold
        x0 = jnp.array(5.0)

        estimator = RiemannianSGD(
            manifold="auto",
            learning_rate=0.1,
            max_iterations=5
        )

        with pytest.raises(ManifoldDetectionError):
            estimator.fit(objective_func, x0)

    def test_predict_not_implemented(self):
        """Test that predict method raises NotImplementedError."""
        estimator = RiemannianSGD(manifold="sphere", learning_rate=0.01)

        with pytest.raises(NotImplementedError):
            estimator.predict(jnp.array([1.0, 0.0]))

    def test_score_method(self):
        """Test score method returns final objective value."""
        def objective_func(x):
            return jnp.sum(x ** 2)

        x0 = jnp.array([1.0, 0.0])

        estimator = RiemannianSGD(
            manifold="sphere",
            learning_rate=0.1,
            max_iterations=5
        )

        # Fit first
        estimator.fit(objective_func, x0)

        # Score should return negative objective value (higher is better)
        score = estimator.score(objective_func, x0)
        assert isinstance(score, float)
        assert score <= 0  # Negative because we minimize


class TestRiemannianAdam:
    """Test RiemannianAdam estimator functionality."""

    def test_adam_initialization(self):
        """Test Adam-specific parameter initialization."""
        estimator = RiemannianAdam(
            manifold="sphere",
            learning_rate=0.01,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8
        )

        params = estimator.get_params()
        assert params["beta1"] == 0.9
        assert params["beta2"] == 0.999
        assert params["eps"] == 1e-8

    def test_adam_parameter_validation(self):
        """Test Adam parameter validation."""
        # Valid parameters
        estimator = RiemannianAdam(manifold="sphere", learning_rate=0.01)
        assert estimator is not None

        # Invalid beta1
        with pytest.raises(ParameterValidationError):
            RiemannianAdam(manifold="sphere", learning_rate=0.01, beta1=1.5)

        # Invalid beta2
        with pytest.raises(ParameterValidationError):
            RiemannianAdam(manifold="sphere", learning_rate=0.01, beta2=-0.1)

    def test_adam_fit(self):
        """Test Adam optimizer fitting."""
        def objective_func(x):
            return jnp.sum((x - jnp.array([0.6, 0.8])) ** 2)

        x0 = jnp.array([1.0, 0.0])

        estimator = RiemannianAdam(
            manifold="sphere",
            learning_rate=0.1,
            max_iterations=10
        )

        result = estimator.fit(objective_func, x0)

        assert isinstance(result, OptimizationResult)
        assert estimator._is_fitted


class TestScikitLearnCompatibility:
    """Test scikit-learn pipeline compatibility."""

    def test_estimator_interface_compliance(self):
        """Test that estimators follow scikit-learn interface."""
        estimator = RiemannianSGD(manifold="sphere", learning_rate=0.01)

        # Should have required methods
        assert hasattr(estimator, "fit")
        assert hasattr(estimator, "get_params")
        assert hasattr(estimator, "set_params")

        # Methods should be callable
        assert callable(estimator.fit)
        assert callable(estimator.get_params)
        assert callable(estimator.set_params)

    def test_parameter_grid_compatibility(self):
        """Test compatibility with parameter grid search."""
        estimator = RiemannianSGD(manifold="sphere", learning_rate=0.01)

        # Should be able to set different parameter combinations
        param_combinations = [
            {"learning_rate": 0.01, "max_iterations": 100},
            {"learning_rate": 0.1, "max_iterations": 50},
            {"learning_rate": 0.001, "max_iterations": 200},
        ]

        for params in param_combinations:
            estimator.set_params(**params)
            retrieved_params = estimator.get_params()
            for key, value in params.items():
                assert retrieved_params[key] == value

    def test_clone_compatibility(self):
        """Test that estimators can be cloned (sklearn.base.clone)."""
        estimator = RiemannianSGD(
            manifold="sphere",
            learning_rate=0.01,
            max_iterations=100
        )

        # Manual clone simulation (sklearn.base.clone logic)
        params = estimator.get_params(deep=True)
        cloned_estimator = RiemannianSGD(**params)

        assert cloned_estimator.learning_rate == estimator.learning_rate
        assert cloned_estimator.manifold == estimator.manifold
        assert cloned_estimator.max_iterations == estimator.max_iterations
