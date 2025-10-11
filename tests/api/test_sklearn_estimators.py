"""Tests for scikit-learn estimator compatibility."""

import numbers

import jax
import jax.numpy as jnp
import pytest

try:
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.pipeline import Pipeline

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from riemannax.api.sklearn_estimators import (
    RiemannianManifoldEstimator,
    RiemannianOptimizer,
    RiemannianPCA,
)
from riemannax.manifolds import Sphere, Stiefel

pytestmark = pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")


class TestRiemannianManifoldEstimator:
    """Test suite for RiemannianManifoldEstimator base class."""

    def test_estimator_inherits_from_base_estimator(self):
        """Test that RiemannianManifoldEstimator inherits from BaseEstimator."""
        # Arrange & Act
        manifold = Sphere(n=3)
        estimator = RiemannianManifoldEstimator(manifold=manifold)

        # Assert
        assert isinstance(estimator, BaseEstimator)

    def test_estimator_has_get_params(self):
        """Test that estimator implements get_params() for GridSearchCV."""
        # Arrange
        manifold = Sphere(n=3)
        estimator = RiemannianManifoldEstimator(manifold=manifold)

        # Act
        params = estimator.get_params(deep=True)

        # Assert
        assert params is not None
        assert "manifold" in params

    def test_estimator_has_set_params(self):
        """Test that estimator implements set_params() for GridSearchCV."""
        # Arrange
        manifold1 = Sphere(n=3)
        manifold2 = Sphere(n=5)
        estimator = RiemannianManifoldEstimator(manifold=manifold1)

        # Act
        estimator.set_params(manifold=manifold2)

        # Assert
        params = estimator.get_params()
        assert params["manifold"].dimension == 5


class TestRiemannianPCA:
    """Test suite for RiemannianPCA transformer."""

    def test_pca_inherits_from_transformer_mixin(self):
        """Test that RiemannianPCA inherits from TransformerMixin."""
        # Arrange & Act
        manifold = Stiefel(n=10, p=5)
        pca = RiemannianPCA(manifold=manifold, n_components=3)

        # Assert
        assert isinstance(pca, TransformerMixin)
        assert isinstance(pca, BaseEstimator)

    def test_pca_fit_method(self):
        """Test that RiemannianPCA implements fit()."""
        # Arrange
        key = jax.random.PRNGKey(0)
        manifold = Stiefel(n=10, p=5)
        pca = RiemannianPCA(manifold=manifold, n_components=3)

        # Generate sample data on manifold
        X = jnp.stack([manifold.random_point(jax.random.fold_in(key, i)) for i in range(20)])

        # Act
        pca_fitted = pca.fit(X)

        # Assert
        assert pca_fitted is pca  # fit should return self
        assert hasattr(pca, "components_")
        assert pca.components_.shape == (3, 10, 5)

    def test_pca_transform_method(self):
        """Test that RiemannianPCA implements transform()."""
        # Arrange
        key = jax.random.PRNGKey(0)
        manifold = Stiefel(n=10, p=5)
        pca = RiemannianPCA(manifold=manifold, n_components=3)

        X = jnp.stack([manifold.random_point(jax.random.fold_in(key, i)) for i in range(20)])
        pca.fit(X)

        # Act
        X_transformed = pca.transform(X)

        # Assert
        assert X_transformed.shape == (20, 3)

    def test_pca_fit_transform_method(self):
        """Test that RiemannianPCA supports fit_transform()."""
        # Arrange
        key = jax.random.PRNGKey(0)
        manifold = Stiefel(n=10, p=5)
        pca = RiemannianPCA(manifold=manifold, n_components=3)

        X = jnp.stack([manifold.random_point(jax.random.fold_in(key, i)) for i in range(20)])

        # Act
        X_transformed = pca.fit_transform(X)

        # Assert
        assert X_transformed.shape == (20, 3)


class TestRiemannianOptimizer:
    """Test suite for RiemannianOptimizer as scikit-learn estimator."""

    def test_optimizer_fit_method(self):
        """Test that RiemannianOptimizer implements fit()."""
        # Arrange
        manifold = Sphere(n=3)
        optimizer = RiemannianOptimizer(
            manifold=manifold, learning_rate=0.01, max_iter=10
        )

        # Simple objective: minimize distance from target point
        target = jnp.array([1.0, 0.0, 0.0, 0.0])

        def objective(x):
            return manifold.dist(x, target) ** 2

        # Initial point
        X = jnp.array([[0.0, 1.0, 0.0, 0.0]])

        # Act
        optimizer_fitted = optimizer.fit(X, objective)

        # Assert
        assert optimizer_fitted is optimizer
        assert hasattr(optimizer, "result_")

    def test_optimizer_score_method(self):
        """Test that RiemannianOptimizer implements score()."""
        # Arrange
        manifold = Sphere(n=3)
        optimizer = RiemannianOptimizer(
            manifold=manifold, learning_rate=0.01, max_iter=10
        )

        target = jnp.array([1.0, 0.0, 0.0, 0.0])

        def objective(x):
            return manifold.dist(x, target) ** 2

        X = jnp.array([[0.0, 1.0, 0.0, 0.0]])
        optimizer.fit(X, objective)

        # Act
        score = optimizer.score(X, objective)

        # Assert
        assert isinstance(score, (int, float))
        assert score <= 0  # Negative loss (scikit-learn convention)

    def test_optimizer_adam_method(self):
        """Test that RiemannianOptimizer works with Adam method."""
        # Arrange
        manifold = Sphere(n=3)
        optimizer = RiemannianOptimizer(
            manifold=manifold,
            learning_rate=0.1,
            max_iter=20,
            method="adam",
            b1=0.9,
            b2=0.999,
        )

        target = jnp.array([1.0, 0.0, 0.0, 0.0])

        def objective(x):
            return manifold.dist(x, target) ** 2

        X = jnp.array([[0.0, 1.0, 0.0, 0.0]])

        # Act
        optimizer.fit(X, objective)

        # Assert
        assert optimizer.result_ is not None
        assert manifold.validate_point(optimizer.result_)
        # Adam should converge to a point closer to target
        final_dist = float(manifold.dist(optimizer.result_, target))
        initial_dist = float(manifold.dist(X[0], target))
        assert final_dist < initial_dist

    def test_optimizer_method_validation(self):
        """Test that RiemannianOptimizer validates method parameter."""
        # Arrange
        manifold = Sphere(n=3)

        # Act & Assert - should raise ValueError for invalid method
        try:
            RiemannianOptimizer(
                manifold=manifold,
                method="invalid_method"
            )
            raise AssertionError("Expected ValueError for invalid method")
        except ValueError as e:
            assert "Unsupported method" in str(e)


class TestPipelineIntegration:
    """Test suite for pipeline integration with scikit-learn."""

    def test_riemannian_pca_in_pipeline(self):
        """Test that RiemannianPCA works in scikit-learn Pipeline."""
        # Arrange
        key = jax.random.PRNGKey(0)
        manifold = Stiefel(n=10, p=5)

        pipeline = Pipeline(
            [
                ("pca", RiemannianPCA(manifold=manifold, n_components=3)),
            ]
        )

        X = jnp.stack([manifold.random_point(jax.random.fold_in(key, i)) for i in range(20)])

        # Act
        X_transformed = pipeline.fit_transform(X)

        # Assert
        assert X_transformed.shape == (20, 3)

    def test_pipeline_get_params(self):
        """Test that pipeline parameters can be accessed with double underscore notation."""
        # Arrange
        manifold = Stiefel(n=10, p=5)
        pipeline = Pipeline(
            [
                ("pca", RiemannianPCA(manifold=manifold, n_components=3)),
            ]
        )

        # Act
        params = pipeline.get_params(deep=True)

        # Assert
        assert "pca" in params
        assert "pca__manifold" in params
        assert "pca__n_components" in params


class TestGridSearchIntegration:
    """Test suite for GridSearchCV integration."""

    def test_grid_search_with_riemannian_pca(self):
        """Test that RiemannianPCA works with GridSearchCV."""
        # Arrange
        key = jax.random.PRNGKey(0)
        manifold = Stiefel(n=10, p=5)

        X = jnp.stack([manifold.random_point(jax.random.fold_in(key, i)) for i in range(20)])

        # Create a simple scoring function
        def score_fn(estimator, X, y=None):
            X_transformed = estimator.transform(X)
            # Simple score: negative variance (higher is better)
            return -float(jnp.var(X_transformed))

        pca = RiemannianPCA(manifold=manifold, n_components=2)

        param_grid = {"n_components": [2, 3]}

        # Act
        grid_search = GridSearchCV(
            pca, param_grid, cv=3, scoring=score_fn
        )
        grid_search.fit(X)

        # Assert
        assert hasattr(grid_search, "best_params_")
        assert "n_components" in grid_search.best_params_


class TestCrossValidation:
    """Test suite for cross-validation compatibility."""

    def test_cross_val_score_with_riemannian_optimizer(self):
        """Test that RiemannianOptimizer works with custom scoring."""
        # Arrange
        manifold = Sphere(n=3)

        # Generate sample data
        key = jax.random.PRNGKey(0)
        X = jnp.stack([manifold.random_point(jax.random.fold_in(key, i)) for i in range(10)])

        target = jnp.array([1.0, 0.0, 0.0, 0.0])

        def objective(x):
            return manifold.dist(x, target) ** 2

        # Custom scorer that works with the optimizer
        def custom_scorer(estimator, X_test, y=None):
            # For each point in X_test, optimize and return negative loss
            scores = []
            for x in X_test:
                estimator.fit(x.reshape(1, -1), objective)
                score = estimator.score(x.reshape(1, -1), objective)
                scores.append(score)
            return float(jnp.mean(jnp.array(scores)))

        optimizer = RiemannianOptimizer(
            manifold=manifold, learning_rate=0.01, max_iter=5
        )

        # Act - Use custom scoring instead of y parameter
        scores = cross_val_score(optimizer, X, cv=3, scoring=custom_scorer)

        # Assert
        assert len(scores) == 3
        assert all(isinstance(s, numbers.Real) for s in scores)
