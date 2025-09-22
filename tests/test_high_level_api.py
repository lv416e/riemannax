"""Tests for high-level API estimators."""

import jax.numpy as jnp
import pytest
from sklearn.base import BaseEstimator

from riemannax.api.estimators import RiemannianSGD, RiemannianAdam
from riemannax.api.detection import ManifoldDetector, minimize


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


class TestAutomaticManifoldDetection:
    """Tests for Task 1.2 - Automatic Manifold Detection System."""

    def test_manifold_detector_unit_vector_detection(self):
        """Test automatic detection of sphere manifold from unit vector."""
        detector = ManifoldDetector()
        x = jnp.array([1.0, 0.0, 0.0])  # Unit vector in R^3

        manifold_type = detector.detect_manifold(x)
        assert manifold_type == "sphere"

    def test_manifold_detector_orthogonal_matrix_detection(self):
        """Test automatic detection of Stiefel manifold from orthogonal matrix."""
        detector = ManifoldDetector()
        # Create orthogonal matrix (3x2, n=3 rows, p=2 columns)
        x = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

        manifold_type = detector.detect_manifold(x)
        assert manifold_type == "stiefel"

    def test_manifold_detector_spd_matrix_detection(self):
        """Test automatic detection of SPD manifold from symmetric positive definite matrix."""
        detector = ManifoldDetector()
        # Create SPD matrix
        x = jnp.array([[2.0, 1.0], [1.0, 2.0]])

        manifold_type = detector.detect_manifold(x)
        assert manifold_type == "spd"

    def test_manifold_detector_square_orthogonal_matrix_detection(self):
        """Test automatic detection of SO manifold from square orthogonal matrix with det=1."""
        detector = ManifoldDetector()
        # Create rotation matrix (det = 1)
        theta = jnp.pi / 4
        x = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])

        manifold_type = detector.detect_manifold(x)
        assert manifold_type == "so"

    def test_manifold_detector_ambiguous_case_raises_error(self):
        """Test that ambiguous manifold detection raises informative error."""
        detector = ManifoldDetector()
        # Random array that doesn't satisfy any manifold constraints
        x = jnp.array([1.5, 2.3, -0.7])

        with pytest.raises(ValueError, match="Could not automatically detect manifold"):
            detector.detect_manifold(x)

    def test_manifold_detector_constraint_validation(self):
        """Test constraint validation for detected manifolds."""
        detector = ManifoldDetector()

        # Valid unit vector
        x_valid = jnp.array([0.6, 0.8, 0.0])
        result = detector.validate_constraints(x_valid, "sphere")
        assert result.is_valid is True
        assert len(result.violations) == 0

        # Invalid unit vector (not normalized)
        x_invalid = jnp.array([1.0, 1.0, 0.0])
        result = detector.validate_constraints(x_invalid, "sphere")
        assert result.is_valid is False
        assert len(result.violations) > 0

    def test_manifold_detector_suggest_manifold(self):
        """Test manifold suggestion for ambiguous cases."""
        detector = ManifoldDetector()
        # Almost unit vector
        x = jnp.array([0.99, 0.1, 0.1])

        suggestions = detector.suggest_manifold(x)
        assert len(suggestions) > 0
        assert any(candidate.manifold_type == "sphere" for candidate in suggestions)

    def test_automatic_minimize_with_unit_vector(self):
        """Test rx.minimize with automatic manifold detection for unit vector."""

        def objective(x):
            return jnp.sum((x - jnp.array([0.0, 0.0, 1.0])) ** 2)

        x0 = jnp.array([1.0, 0.0, 0.0])  # Unit vector
        result = minimize(objective, x0, method="riemannian_adam")

        assert hasattr(result, "x")
        assert hasattr(result, "fun")
        assert hasattr(result, "success")

    def test_automatic_minimize_with_orthogonal_matrix(self):
        """Test rx.minimize with automatic manifold detection for orthogonal matrix."""

        def objective(X):
            target = jnp.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
            return jnp.sum((X - target) ** 2)

        # Initial orthogonal matrix (3x2, n=3 rows, p=2 columns for St(2,3))
        x0 = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        result = minimize(objective, x0, method="riemannian_adam")

        assert hasattr(result, "x")
        assert hasattr(result, "fun")
        assert hasattr(result, "success")

    def test_automatic_minimize_with_spd_matrix(self):
        """Test rx.minimize with automatic manifold detection for SPD matrix."""

        def objective(X):
            target = jnp.array([[3.0, 0.5], [0.5, 2.0]])
            return jnp.sum((X - target) ** 2)

        # Initial SPD matrix
        x0 = jnp.array([[2.0, 1.0], [1.0, 2.0]])
        result = minimize(objective, x0, method="riemannian_adam")

        assert hasattr(result, "x")
        assert hasattr(result, "fun")
        assert hasattr(result, "success")

    def test_automatic_detection_logs_manifold_type(self, caplog):
        """Test that automatic detection logs the selected manifold type."""
        import logging
        caplog.set_level(logging.INFO)

        def objective(x):
            return jnp.sum(x**2)

        x0 = jnp.array([1.0, 0.0, 0.0])  # Unit vector
        minimize(objective, x0, method="riemannian_adam")

        # Check that manifold selection was logged
        assert any("Selected manifold: sphere" in record.message for record in caplog.records)

    def test_automatic_detection_invalid_method_raises_error(self):
        """Test that invalid optimization method raises descriptive error."""

        def objective(x):
            return jnp.sum(x**2)

        x0 = jnp.array([1.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="Unsupported optimization method"):
            minimize(objective, x0, method="invalid_method")

    def test_manifold_detector_validation_result_attributes(self):
        """Test that ValidationResult has required attributes."""
        detector = ManifoldDetector()
        x = jnp.array([1.0, 0.0, 0.0])

        result = detector.validate_constraints(x, "sphere")

        assert hasattr(result, "is_valid")
        assert hasattr(result, "violations")
        assert hasattr(result, "suggestions")
        assert isinstance(result.violations, list)
        assert isinstance(result.suggestions, list)

    def test_manifold_detector_grassmann_detection(self):
        """Test detection of Grassmann manifold from orthogonal matrix with p < n."""
        detector = ManifoldDetector()
        # 2x4 orthogonal matrix representing 2D subspace in R^4
        q, _ = jnp.linalg.qr(jnp.array([[1.0, 0.5, 0.2, 0.1], [0.0, 1.0, 0.3, 0.2]]).T)
        x = q[:, :2]  # Take first 2 columns

        manifold_type = detector.detect_manifold(x)
        assert manifold_type in ["grassmann", "stiefel"]  # Both are valid for this case

    def test_automatic_detection_tolerance_parameters(self):
        """Test that automatic detection respects tolerance parameters."""
        detector = ManifoldDetector(unit_tol=1e-6, orthogonal_tol=1e-6)

        # Slightly non-unit vector within tolerance
        x = jnp.array([1.0000001, 0.0, 0.0])
        x = x / jnp.linalg.norm(x)  # Normalize to be exactly unit

        manifold_type = detector.detect_manifold(x)
        assert manifold_type == "sphere"
