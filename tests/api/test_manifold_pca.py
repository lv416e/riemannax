"""Tests for Manifold PCA (Principal Geodesic Analysis) problem template."""

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from riemannax.api.problems import ManifoldPCA
from riemannax.manifolds import Sphere


class TestManifoldPCAInterface:
    """Test ManifoldPCA sklearn-compatible interface."""

    def test_initialization_with_valid_parameters(self):
        """Test initialization with valid n_components parameter."""
        pca = ManifoldPCA(n_components=2, max_iter=100, tolerance=1e-6)
        assert pca.n_components == 2
        assert pca.max_iter == 100
        assert pca.tolerance == 1e-6

    def test_initialization_validates_n_components(self):
        """Test n_components validation during initialization."""
        with pytest.raises(ValueError, match="n_components must be positive"):
            ManifoldPCA(n_components=0)

        with pytest.raises(ValueError, match="n_components must be positive"):
            ManifoldPCA(n_components=-1)

    def test_get_params_returns_parameters(self):
        """Test get_params returns all estimator parameters."""
        pca = ManifoldPCA(n_components=3, max_iter=200, tolerance=1e-5)
        params = pca.get_params(deep=True)

        assert params["n_components"] == 3
        assert params["max_iter"] == 200
        assert params["tolerance"] == 1e-5

    def test_set_params_updates_parameters(self):
        """Test set_params updates parameters and validates them."""
        pca = ManifoldPCA(n_components=2)
        pca.set_params(n_components=4, max_iter=300)

        assert pca.n_components == 4
        assert pca.max_iter == 300

    def test_set_params_validates_invalid_n_components(self):
        """Test set_params rejects invalid n_components."""
        pca = ManifoldPCA(n_components=2)

        with pytest.raises(ValueError, match="n_components must be positive"):
            pca.set_params(n_components=-1)


class TestManifoldPCAFit:
    """Test ManifoldPCA fit method."""

    def test_fit_with_sphere_data(self):
        """Test fitting with data on sphere manifold."""
        # Create synthetic data on unit sphere
        key = jax.random.PRNGKey(42)
        n_samples = 50
        dim = 3
        n_components = 2

        # Generate random points on sphere
        X_raw = jax.random.normal(key, (n_samples, dim))
        X = X_raw / jnp.linalg.norm(X_raw, axis=1, keepdims=True)

        # Create manifold and fit PCA (Sphere(n=dim-1) for ambient dimension dim)
        manifold = Sphere(n=dim - 1)
        pca = ManifoldPCA(manifold=manifold, n_components=n_components, max_iter=50)
        result = pca.fit(X)

        assert result is pca  # sklearn convention: fit returns self
        assert pca.mean_ is not None
        assert pca.components_ is not None
        assert pca.explained_variance_ is not None
        assert pca.components_.shape == (n_components, dim)
        assert pca.explained_variance_.shape == (n_components,)

    def test_fit_validates_manifold_parameter(self):
        """Test fit requires manifold parameter."""
        pca = ManifoldPCA(n_components=2)
        X = jnp.ones((10, 3))

        with pytest.raises(ValueError, match="manifold must be provided"):
            pca.fit(X)

    def test_fit_requires_2d_data(self):
        """Test fit rejects non-2D input."""
        X = jnp.array([1.0, 2.0, 3.0])  # 1D array
        manifold = Sphere(n=2)

        pca = ManifoldPCA(manifold=manifold, n_components=1)

        with pytest.raises(ValueError, match="Expected 2D data matrix"):
            pca.fit(X)

    def test_fit_rejects_n_components_exceeding_dimension(self):
        """Test fit rejects n_components larger than ambient dimension."""
        X = jnp.ones((20, 3))
        manifold = Sphere(n=2)

        pca = ManifoldPCA(manifold=manifold, n_components=5)  # > 3

        with pytest.raises(ValueError, match="n_components.*cannot exceed"):
            pca.fit(X)

    def test_fit_warns_with_insufficient_samples(self):
        """Test fit warns when n_samples < n_components."""
        X = jnp.ones((2, 5))
        X = X / jnp.linalg.norm(X, axis=1, keepdims=True)  # Normalize
        manifold = Sphere(n=4)

        pca = ManifoldPCA(manifold=manifold, n_components=3)

        with pytest.warns(UserWarning, match="n_components.*exceeds n_samples"):
            pca.fit(X)

    def test_fit_validates_data_on_manifold(self):
        """Test fit validates that data lies on manifold."""
        # Create points NOT on unit sphere
        X = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Not normalized
        manifold = Sphere(n=2)

        pca = ManifoldPCA(manifold=manifold, n_components=1)

        with pytest.raises(ValueError, match="Data points.*must have unit norm|Data points must lie on the manifold"):
            pca.fit(X)


class TestManifoldPCATransform:
    """Test ManifoldPCA transform method."""

    def test_transform_projects_to_lower_dimension(self):
        """Test transform returns projected coordinates."""
        key = jax.random.PRNGKey(42)
        n_samples = 50
        dim = 4
        n_components = 2

        # Generate data on sphere
        X_raw = jax.random.normal(key, (n_samples, dim))
        X = X_raw / jnp.linalg.norm(X_raw, axis=1, keepdims=True)

        # Fit and transform
        manifold = Sphere(n=dim - 1)
        pca = ManifoldPCA(manifold=manifold, n_components=n_components, max_iter=100)
        pca.fit(X)
        X_transformed = pca.transform(X)

        assert X_transformed.shape == (n_samples, n_components)
        # Transformed coordinates should have lower dimension
        assert X_transformed.shape[1] < X.shape[1]

    def test_transform_before_fit_raises_error(self):
        """Test transform raises error when called before fit."""
        X = jnp.ones((10, 3))
        manifold = Sphere(n=2)

        pca = ManifoldPCA(manifold=manifold, n_components=2)

        with pytest.raises(ValueError, match="not fitted"):
            pca.transform(X)

    def test_transform_validates_consistent_dimensions(self):
        """Test transform validates input dimensions match fitted data."""
        # Fit on one dimension
        X_fit = jnp.ones((20, 4))
        X_fit = X_fit / jnp.linalg.norm(X_fit, axis=1, keepdims=True)

        manifold = Sphere(n=3)
        pca = ManifoldPCA(manifold=manifold, n_components=2, max_iter=10)
        pca.fit(X_fit)

        # Try to transform different dimension
        X_transform = jnp.ones((10, 5))  # Different ambient dim

        with pytest.raises(ValueError, match="Dimension mismatch"):
            pca.transform(X_transform)


class TestManifoldPCAInverseTransform:
    """Test ManifoldPCA inverse_transform method."""

    def test_inverse_transform_reconstructs_manifold_points(self):
        """Test inverse_transform returns points on manifold."""
        key = jax.random.PRNGKey(42)
        n_samples = 30
        dim = 4
        n_components = 2

        # Generate data
        X_raw = jax.random.normal(key, (n_samples, dim))
        X = X_raw / jnp.linalg.norm(X_raw, axis=1, keepdims=True)

        # Fit, transform, and inverse transform
        manifold = Sphere(n=dim - 1)
        pca = ManifoldPCA(manifold=manifold, n_components=n_components, max_iter=100)
        pca.fit(X)
        X_transformed = pca.transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)

        assert X_reconstructed.shape == X.shape
        # Check reconstructed points lie on sphere (unit norm)
        # Projection step explicitly normalizes, so tolerance should be tight
        norms = jnp.linalg.norm(X_reconstructed, axis=1)
        assert jnp.allclose(norms, 1.0, atol=1e-6)  # Projection ensures unit norm

    def test_inverse_transform_before_fit_raises_error(self):
        """Test inverse_transform requires fitting first."""
        X_transformed = jnp.array([[1.0, 2.0]])
        manifold = Sphere(n=1)

        pca = ManifoldPCA(manifold=manifold, n_components=2)

        with pytest.raises(ValueError, match="not fitted"):
            pca.inverse_transform(X_transformed)

    def test_inverse_transform_validates_n_components(self):
        """Test inverse_transform validates input shape."""
        X_fit = jnp.ones((20, 4))
        X_fit = X_fit / jnp.linalg.norm(X_fit, axis=1, keepdims=True)

        manifold = Sphere(n=3)
        pca = ManifoldPCA(manifold=manifold, n_components=2, max_iter=10)
        pca.fit(X_fit)

        # Try to inverse transform with wrong n_components
        X_wrong = jnp.ones((10, 3))  # Should be (10, 2)

        with pytest.raises(ValueError, match="must have.*components"):
            pca.inverse_transform(X_wrong)


class TestManifoldPCAFitTransform:
    """Test ManifoldPCA fit_transform method."""

    def test_fit_transform_combines_fit_and_transform(self):
        """Test fit_transform is equivalent to fit().transform()."""
        key = jax.random.PRNGKey(42)
        n_samples = 40
        dim = 5
        n_components = 3

        # Generate data
        X_raw = jax.random.normal(key, (n_samples, dim))
        X = X_raw / jnp.linalg.norm(X_raw, axis=1, keepdims=True)

        # Method 1: fit_transform
        manifold = Sphere(n=dim - 1)
        pca1 = ManifoldPCA(manifold=manifold, n_components=n_components, max_iter=50)
        X_transformed_1 = pca1.fit_transform(X)

        # Method 2: fit then transform
        pca2 = ManifoldPCA(manifold=manifold, n_components=n_components, max_iter=50)
        pca2.fit(X)
        X_transformed_2 = pca2.transform(X)

        # Results should be identical (algorithm is deterministic; random_state is unused)
        assert jnp.allclose(X_transformed_1, X_transformed_2, atol=1e-6)


class TestManifoldPCAExplainedVariance:
    """Test ManifoldPCA explained variance properties."""

    def test_explained_variance_is_positive_and_sorted(self):
        """Test explained variance is positive and in descending order."""
        key = jax.random.PRNGKey(42)
        n_samples = 60
        dim = 4
        n_components = 3

        # Generate data
        X_raw = jax.random.normal(key, (n_samples, dim))
        X = X_raw / jnp.linalg.norm(X_raw, axis=1, keepdims=True)

        manifold = Sphere(n=dim - 1)
        pca = ManifoldPCA(manifold=manifold, n_components=n_components, max_iter=50)
        pca.fit(X)

        # Check explained variance properties
        assert jnp.all(pca.explained_variance_ >= 0)  # Non-negative
        # Should be in descending order
        assert jnp.all(pca.explained_variance_[:-1] >= pca.explained_variance_[1:])

    def test_explained_variance_ratio_sums_to_at_most_one(self):
        """Test explained variance ratio sums to at most 1."""
        key = jax.random.PRNGKey(42)
        n_samples = 50
        dim = 5
        n_components = 3

        X_raw = jax.random.normal(key, (n_samples, dim))
        X = X_raw / jnp.linalg.norm(X_raw, axis=1, keepdims=True)

        manifold = Sphere(n=dim - 1)
        pca = ManifoldPCA(manifold=manifold, n_components=n_components, max_iter=50)
        pca.fit(X)

        ratio = pca.explained_variance_ratio_
        assert jnp.sum(ratio) <= 1.0 + 1e-6  # Allow small numerical error


class TestManifoldPCAConvergence:
    """Test ManifoldPCA convergence properties."""

    def test_reconstruction_error_decreases_with_more_components(self):
        """Test that more components lead to better reconstruction."""
        key = jax.random.PRNGKey(42)
        n_samples = 50
        dim = 5

        # Generate data
        X_raw = jax.random.normal(key, (n_samples, dim))
        X = X_raw / jnp.linalg.norm(X_raw, axis=1, keepdims=True)

        manifold = Sphere(n=dim - 1)

        # Fit with 2 components
        pca2 = ManifoldPCA(manifold=manifold, n_components=2, max_iter=100)
        pca2.fit(X)
        X_proj2 = pca2.transform(X)
        X_recon2 = pca2.inverse_transform(X_proj2)
        # Compute distances using vectorized manifold.dist
        dists2 = jax.vmap(manifold.dist)(X, X_recon2)
        error2 = float(jnp.mean(dists2 ** 2))

        # Fit with 3 components
        pca3 = ManifoldPCA(manifold=manifold, n_components=3, max_iter=100)
        pca3.fit(X)
        X_proj3 = pca3.transform(X)
        X_recon3 = pca3.inverse_transform(X_proj3)
        dists3 = jax.vmap(manifold.dist)(X, X_recon3)
        error3 = float(jnp.mean(dists3 ** 2))

        # More components should give lower error
        assert error3 <= error2 + 1e-5  # Allow small numerical tolerance

    def test_handles_concentrated_data_gracefully(self):
        """Test algorithm handles data concentrated in small region."""
        key = jax.random.PRNGKey(42)
        n_samples = 40
        dim = 4

        # Generate data concentrated near north pole
        X_raw = jax.random.normal(key, (n_samples, dim)) * 0.1  # Small variance
        X_raw = X_raw.at[:, 0].add(1.0)  # Shift toward north pole
        X = X_raw / jnp.linalg.norm(X_raw, axis=1, keepdims=True)

        manifold = Sphere(n=dim - 1)
        pca = ManifoldPCA(manifold=manifold, n_components=2, max_iter=100)
        pca.fit(X)

        X_transformed = pca.transform(X)

        # Should complete without errors
        assert X_transformed.shape == (n_samples, 2)
        assert not jnp.any(jnp.isnan(X_transformed))
        assert not jnp.any(jnp.isinf(X_transformed))
