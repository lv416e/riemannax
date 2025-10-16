"""Tests for Robust Covariance Estimation problem template."""

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from riemannax.api.problems import RobustCovarianceEstimation
from riemannax.manifolds import SymmetricPositiveDefinite


class TestRobustCovarianceInterface:
    """Test RobustCovarianceEstimation sklearn-compatible interface."""

    def test_initialization_with_valid_parameters(self):
        """Test initialization with valid parameters."""
        rce = RobustCovarianceEstimation(max_iter=100, tolerance=1e-6)
        assert rce.max_iter == 100
        assert rce.tolerance == 1e-6

    def test_initialization_with_custom_metric(self):
        """Test initialization with custom metric type."""
        rce = RobustCovarianceEstimation(metric="affine_invariant")
        assert rce.metric == "affine_invariant"

        rce = RobustCovarianceEstimation(metric="log_euclidean")
        assert rce.metric == "log_euclidean"

    def test_get_params_returns_parameters(self):
        """Test get_params returns all estimator parameters."""
        rce = RobustCovarianceEstimation(max_iter=200, tolerance=1e-5)
        params = rce.get_params(deep=True)

        assert params["max_iter"] == 200
        assert params["tolerance"] == 1e-5

    def test_set_params_updates_parameters(self):
        """Test set_params updates parameters."""
        rce = RobustCovarianceEstimation()
        rce.set_params(max_iter=300, tolerance=1e-4)

        assert rce.max_iter == 300
        assert rce.tolerance == 1e-4


class TestRobustCovarianceFit:
    """Test RobustCovarianceEstimation fit method."""

    def test_fit_with_spd_matrices(self):
        """Test fitting with SPD matrices."""
        # Create synthetic SPD matrices
        key = jax.random.PRNGKey(42)
        n_samples = 20
        matrix_dim = 3

        # Generate random SPD matrices
        manifold = SymmetricPositiveDefinite(n=matrix_dim)
        X = manifold.random_point(key, n_samples)

        # Fit robust covariance
        rce = RobustCovarianceEstimation(max_iter=50)
        result = rce.fit(X)

        assert result is rce  # sklearn convention: fit returns self
        assert rce.geometric_median_ is not None
        assert rce.geometric_median_.shape == (matrix_dim, matrix_dim)

    def test_fit_infers_manifold_from_data(self):
        """Test fit correctly infers manifold from input data shape."""
        # Create valid SPD matrices
        manifold = SymmetricPositiveDefinite(n=3)
        X = manifold.random_point(jax.random.PRNGKey(42), 10)
        rce = RobustCovarianceEstimation()

        # Should work without explicit manifold - infers from data
        result = rce.fit(X)
        assert result is rce

    def test_fit_rejects_invalid_flattened_input(self):
        """Test fit rejects 2D input with non-perfect-square dimension."""
        X = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2D array with ambient_dim=3 (not a perfect square)
        rce = RobustCovarianceEstimation()

        with pytest.raises(ValueError, match="ambient_dim must be a perfect square"):
            rce.fit(X)

    def test_fit_validates_square_matrices(self):
        """Test fit rejects non-square matrices."""
        X = jnp.ones((10, 3, 4))  # Non-square matrices
        rce = RobustCovarianceEstimation()

        with pytest.raises(ValueError, match="Expected square matrices"):
            rce.fit(X)

    def test_fit_validates_spd_matrices(self):
        """Test fit validates that matrices are SPD."""
        # Create non-SPD matrices (not positive definite)
        X = jnp.array([
            [[-1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, -1.0]]
        ])
        rce = RobustCovarianceEstimation()

        with pytest.raises(ValueError, match="All matrices must be SPD"):
            rce.fit(X)

    def test_fit_validates_symmetric_matrices(self):
        """Test fit validates that matrices are symmetric."""
        # Create non-symmetric matrices
        X = jnp.array([
            [[1.0, 2.0], [0.0, 1.0]],  # Not symmetric
            [[1.0, 0.0], [0.0, 1.0]]
        ])
        rce = RobustCovarianceEstimation()

        with pytest.raises(ValueError, match="All matrices must be symmetric"):
            rce.fit(X)


class TestRobustCovarianceTransform:
    """Test RobustCovarianceEstimation transform method."""

    def test_transform_returns_tangent_vectors(self):
        """Test transform maps SPD matrices to tangent space at geometric median."""
        key = jax.random.PRNGKey(42)
        n_samples = 15
        matrix_dim = 3

        # Generate data
        manifold = SymmetricPositiveDefinite(n=matrix_dim)
        X = manifold.random_point(key, n_samples)

        # Fit and transform
        rce = RobustCovarianceEstimation(max_iter=50)
        rce.fit(X)
        X_transformed = rce.transform(X)

        assert X_transformed.shape == (n_samples, matrix_dim, matrix_dim)
        # Transformed data should be tangent vectors (symmetric matrices)
        for i in range(n_samples):
            assert jnp.allclose(X_transformed[i], X_transformed[i].T, atol=1e-5)

    def test_transform_before_fit_raises_error(self):
        """Test transform raises error when called before fit."""
        X = jnp.eye(3)[None, :, :]
        rce = RobustCovarianceEstimation()

        with pytest.raises(ValueError, match="not fitted"):
            rce.transform(X)

    def test_transform_validates_consistent_dimensions(self):
        """Test transform validates input dimensions match fitted data."""
        # Fit on one dimension
        manifold = SymmetricPositiveDefinite(n=3)
        X_fit = manifold.random_point(jax.random.PRNGKey(42), 20)

        rce = RobustCovarianceEstimation(max_iter=10)
        rce.fit(X_fit)

        # Try to transform different dimension
        X_transform = jnp.ones((10, 4, 4))  # Different matrix size

        with pytest.raises(ValueError, match="Dimension mismatch"):
            rce.transform(X_transform)


class TestRobustCovarianceInverseTransform:
    """Test RobustCovarianceEstimation inverse_transform method."""

    def test_inverse_transform_maps_tangent_to_manifold(self):
        """Test inverse_transform returns SPD matrices from tangent vectors."""
        key = jax.random.PRNGKey(42)
        n_samples = 15
        matrix_dim = 3

        # Generate data
        manifold = SymmetricPositiveDefinite(n=matrix_dim)
        X = manifold.random_point(key, n_samples)

        # Fit, transform, and inverse transform
        rce = RobustCovarianceEstimation(max_iter=50)
        rce.fit(X)
        X_transformed = rce.transform(X)
        X_reconstructed = rce.inverse_transform(X_transformed)

        assert X_reconstructed.shape == X.shape
        # Reconstructed matrices should be SPD
        for i in range(n_samples):
            eigenvalues = jnp.linalg.eigvalsh(X_reconstructed[i])
            assert jnp.all(eigenvalues > 1e-9), f"Matrix {i} not positive definite"

    def test_inverse_transform_before_fit_raises_error(self):
        """Test inverse_transform requires fitting first."""
        X_tangent = jnp.zeros((5, 3, 3))
        rce = RobustCovarianceEstimation()

        with pytest.raises(ValueError, match="not fitted"):
            rce.inverse_transform(X_tangent)


class TestRobustCovarianceFitTransform:
    """Test RobustCovarianceEstimation fit_transform method."""

    def test_fit_transform_combines_fit_and_transform(self):
        """Test fit_transform is equivalent to fit().transform()."""
        key = jax.random.PRNGKey(42)
        n_samples = 20
        matrix_dim = 3

        # Generate data
        manifold = SymmetricPositiveDefinite(n=matrix_dim)
        X = manifold.random_point(key, n_samples)

        # Method 1: fit_transform
        rce1 = RobustCovarianceEstimation(max_iter=50, random_state=42)
        X_transformed_1 = rce1.fit_transform(X)

        # Method 2: fit then transform
        rce2 = RobustCovarianceEstimation(max_iter=50, random_state=42)
        rce2.fit(X)
        X_transformed_2 = rce2.transform(X)

        # Results should be identical (same random state)
        assert jnp.allclose(X_transformed_1, X_transformed_2, atol=1e-6)


class TestRobustCovarianceGeometricMedian:
    """Test RobustCovarianceEstimation geometric median computation."""

    def test_geometric_median_is_spd(self):
        """Test computed geometric median is SPD."""
        key = jax.random.PRNGKey(42)
        n_samples = 25
        matrix_dim = 3

        # Generate data
        manifold = SymmetricPositiveDefinite(n=matrix_dim)
        X = manifold.random_point(key, n_samples)

        rce = RobustCovarianceEstimation(max_iter=100)
        rce.fit(X)

        # Check geometric median is SPD
        median = rce.geometric_median_
        assert jnp.allclose(median, median.T, atol=1e-5)  # Symmetric
        eigenvalues = jnp.linalg.eigvalsh(median)
        assert jnp.all(eigenvalues > 1e-9)  # Positive definite

    def test_geometric_median_robust_to_outliers(self):
        """Test geometric median is more robust than mean."""
        key = jax.random.PRNGKey(42)
        n_samples = 20
        matrix_dim = 2

        manifold = SymmetricPositiveDefinite(n=matrix_dim)

        # Create clean data concentrated around identity
        # Use exponential map to move matrices closer to identity while preserving SPD
        X_clean = manifold.random_point(key, n_samples)
        identity = jnp.eye(matrix_dim)
        X_clean = jax.vmap(lambda x: manifold.exp(identity, 0.1 * manifold.log(identity, x)))(X_clean)

        # Add outliers (matrices far from others)
        outliers = manifold.random_point(jax.random.fold_in(key, 1), 3) * 10.0
        X_with_outliers = jnp.concatenate([X_clean, outliers], axis=0)

        # Compute geometric median with and without outliers
        rce_clean = RobustCovarianceEstimation(max_iter=100)
        rce_clean.fit(X_clean)

        rce_outliers = RobustCovarianceEstimation(max_iter=100)
        rce_outliers.fit(X_with_outliers)

        # Geometric median should be relatively stable
        median_clean = rce_clean.geometric_median_
        median_outliers = rce_outliers.geometric_median_

        # Distance between medians should be smaller than distance to outliers
        dist_medians = manifold.dist(median_clean, median_outliers)
        dist_to_outlier = manifold.dist(median_clean, outliers[0])

        assert dist_medians < dist_to_outlier * 0.5  # Median is robust


class TestRobustCovarianceConvergence:
    """Test RobustCovarianceEstimation convergence properties."""

    def test_weiszfeld_algorithm_converges(self):
        """Test Weiszfeld algorithm converges for geometric median."""
        key = jax.random.PRNGKey(42)
        n_samples = 30
        matrix_dim = 3

        # Generate data
        manifold = SymmetricPositiveDefinite(n=matrix_dim)
        X = manifold.random_point(key, n_samples)

        rce = RobustCovarianceEstimation(max_iter=200, tolerance=1e-8)
        rce.fit(X)

        # Check that we have a valid median
        assert rce.geometric_median_ is not None
        assert not jnp.any(jnp.isnan(rce.geometric_median_))
        assert not jnp.any(jnp.isinf(rce.geometric_median_))

    def test_handles_identical_matrices(self):
        """Test algorithm handles edge case of identical matrices."""
        matrix_dim = 3
        matrix = jnp.eye(matrix_dim) * 2.0
        X = jnp.tile(matrix[None, :, :], (10, 1, 1))

        rce = RobustCovarianceEstimation(max_iter=10)
        rce.fit(X)

        # Geometric median should be the common matrix
        assert jnp.allclose(rce.geometric_median_, matrix, atol=1e-5)

    def test_metric_comparison_affine_vs_log_euclidean(self):
        """Test both affine-invariant and log-Euclidean metrics work."""
        key = jax.random.PRNGKey(42)
        n_samples = 15
        matrix_dim = 2

        manifold = SymmetricPositiveDefinite(n=matrix_dim)
        X = manifold.random_point(key, n_samples)

        # Affine-invariant metric
        rce_affine = RobustCovarianceEstimation(metric="affine_invariant", max_iter=50)
        rce_affine.fit(X)

        # Log-Euclidean metric
        rce_log_euc = RobustCovarianceEstimation(metric="log_euclidean", max_iter=50)
        rce_log_euc.fit(X)

        # Both should produce valid SPD medians
        for median in [rce_affine.geometric_median_, rce_log_euc.geometric_median_]:
            assert jnp.allclose(median, median.T, atol=1e-5)
            eigenvalues = jnp.linalg.eigvalsh(median)
            assert jnp.all(eigenvalues > 1e-9)  # Positive definite
