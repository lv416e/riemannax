"""Tests for Product manifold basic structure and functionality.

This module tests the ProductManifold implementation, which enables composition
of multiple manifolds into a single product manifold M = M₁ × M₂ × ... × Mₖ.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from typing import Tuple

from riemannax.manifolds.product import ProductManifold
from riemannax.manifolds.sphere import Sphere
from riemannax.manifolds.spd import SymmetricPositiveDefinite


def is_point_on_manifold(manifold, point):
    """Helper function to check if a point is on a manifold.

    Different manifolds have different validation methods, so we need
    to handle each type appropriately.
    """
    if isinstance(manifold, Sphere):
        # For sphere manifolds, check unit norm
        return jnp.allclose(jnp.linalg.norm(point), 1.0, atol=1e-10)
    elif isinstance(manifold, SymmetricPositiveDefinite):
        # SPD manifolds have their own validation method
        return manifold._is_in_manifold(point)
    else:
        # For other manifolds, we could add more specific checks
        # For now, assume point is valid (this is a test limitation)
        return True


class TestProductManifoldBasicStructure:
    """Test suite for ProductManifold basic structure implementation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.key = jr.PRNGKey(42)

        # Create component manifolds
        self.sphere3 = Sphere(n=3)  # S³ (4D ambient)
        self.sphere2 = Sphere(n=2)  # S² (3D ambient)
        self.spd3 = SymmetricPositiveDefinite(n=3)  # 3×3 SPD matrices

        # Create product manifolds for testing
        self.two_spheres = ProductManifold(manifolds=(self.sphere3, self.sphere2))
        self.sphere_spd = ProductManifold(manifolds=(self.sphere2, self.spd3))

    def test_product_manifold_class_exists(self):
        """Test that ProductManifold class exists and can be instantiated."""
        # This will fail initially - we haven't created the class yet
        assert hasattr(ProductManifold, '__init__')
        assert callable(ProductManifold)

    def test_product_manifold_init_accepts_tuple_of_manifolds(self):
        """Test that ProductManifold.__init__ accepts Tuple[BaseManifold, ...]."""
        # Should accept tuple of manifolds
        product = ProductManifold(manifolds=(self.sphere3, self.sphere2))
        assert hasattr(product, 'manifolds')
        assert len(product.manifolds) == 2
        assert product.manifolds[0] is self.sphere3
        assert product.manifolds[1] is self.sphere2

    def test_product_manifold_init_validation(self):
        """Test that ProductManifold.__init__ validates input."""
        # Should reject empty tuple
        with pytest.raises((ValueError, TypeError)):
            ProductManifold(manifolds=())

        # Should reject non-manifold objects
        with pytest.raises((ValueError, TypeError)):
            ProductManifold(manifolds=(self.sphere3, "not_a_manifold"))

    def test_split_point_method_exists(self):
        """Test that _split_point method exists."""
        assert hasattr(self.two_spheres, '_split_point')
        assert callable(getattr(self.two_spheres, '_split_point'))

    def test_combine_points_method_exists(self):
        """Test that _combine_points method exists."""
        assert hasattr(self.two_spheres, '_combine_points')
        assert callable(getattr(self.two_spheres, '_combine_points'))

    def test_split_point_basic_functionality(self):
        """Test basic _split_point functionality."""
        key1, key2 = jr.split(self.key, 2)

        # Create component points
        sphere3_point = self.sphere3.random_point(key1)  # Shape: (4,)
        sphere2_point = self.sphere2.random_point(key2)  # Shape: (3,)

        # Combine into product point
        product_point = self.two_spheres._combine_points((sphere3_point, sphere2_point))

        # Split should return the original components
        components = self.two_spheres._split_point(product_point)

        assert len(components) == 2
        assert jnp.allclose(components[0], sphere3_point, atol=1e-12)
        assert jnp.allclose(components[1], sphere2_point, atol=1e-12)

    def test_combine_points_basic_functionality(self):
        """Test basic _combine_points functionality."""
        key1, key2 = jr.split(self.key, 2)

        # Create component points
        sphere3_point = self.sphere3.random_point(key1)  # Shape: (4,)
        sphere2_point = self.sphere2.random_point(key2)  # Shape: (3,)

        # Combine into product point
        product_point = self.two_spheres._combine_points((sphere3_point, sphere2_point))

        # Should be concatenated array
        expected_shape = (7,)  # 4 + 3
        assert product_point.shape == expected_shape

        # First 4 elements should be sphere3_point
        assert jnp.allclose(product_point[:4], sphere3_point, atol=1e-12)
        # Next 3 elements should be sphere2_point
        assert jnp.allclose(product_point[4:], sphere2_point, atol=1e-12)

    def test_dimension_property_sum_of_components(self):
        """Test that dimension property returns sum of component dimensions."""
        # sphere3.dimension = 3, sphere2.dimension = 2
        expected_dim = 3 + 2  # 5
        assert self.two_spheres.dimension == expected_dim

        # sphere2.dimension = 2, spd3.dimension = 6 (3×3 SPD has 6 DoF)
        expected_dim_mixed = 2 + 6  # 8
        assert self.sphere_spd.dimension == expected_dim_mixed

    def test_ambient_dimension_property_sum_of_components(self):
        """Test that ambient_dimension property returns sum of component ambient dimensions."""
        # sphere3.ambient_dimension = 4, sphere2.ambient_dimension = 3
        expected_ambient_dim = 4 + 3  # 7
        assert self.two_spheres.ambient_dimension == expected_ambient_dim

        # sphere2.ambient_dimension = 3, spd3.ambient_dimension = 9 (3×3 matrices)
        expected_ambient_dim_mixed = 3 + 9  # 12
        assert self.sphere_spd.ambient_dimension == expected_ambient_dim_mixed

    def test_split_combine_inverse_operations(self):
        """Test that split and combine are inverse operations."""
        key1, key2 = jr.split(self.key, 2)

        # Create random component points
        components = (
            self.sphere3.random_point(key1),
            self.sphere2.random_point(key2)
        )

        # combine -> split should return original
        product_point = self.two_spheres._combine_points(components)
        recovered_components = self.two_spheres._split_point(product_point)

        assert len(recovered_components) == len(components)
        for orig, recovered in zip(components, recovered_components):
            assert jnp.allclose(orig, recovered, atol=1e-12)

    def test_manifold_inheritance(self):
        """Test that ProductManifold inherits from base Manifold class."""
        from riemannax.manifolds.base import Manifold
        assert isinstance(self.two_spheres, Manifold)
        assert isinstance(self.sphere_spd, Manifold)

    def test_different_manifold_combinations(self):
        """Test ProductManifold with different manifold type combinations."""
        # Single manifold (edge case)
        single_product = ProductManifold(manifolds=(self.sphere3,))
        assert single_product.dimension == self.sphere3.dimension
        assert single_product.ambient_dimension == self.sphere3.ambient_dimension

        # Three manifolds
        three_manifolds = ProductManifold(manifolds=(self.sphere3, self.sphere2, self.spd3))
        expected_dim = 3 + 2 + 6  # 11
        expected_ambient_dim = 4 + 3 + 9  # 16
        assert three_manifolds.dimension == expected_dim
        assert three_manifolds.ambient_dimension == expected_ambient_dim


class TestProductManifoldPointManipulation:
    """Test suite for product manifold point manipulation methods."""

    def setup_method(self):
        """Setup test fixtures."""
        self.key = jr.PRNGKey(123)
        self.sphere2 = Sphere(n=2)
        self.sphere3 = Sphere(n=3)
        self.product = ProductManifold(manifolds=(self.sphere2, self.sphere3))

    def test_split_point_preserves_shapes(self):
        """Test that _split_point preserves component shapes correctly."""
        key1, key2 = jr.split(self.key, 2)

        point1 = self.sphere2.random_point(key1)  # (3,)
        point2 = self.sphere3.random_point(key2)  # (4,)

        product_point = self.product._combine_points((point1, point2))
        components = self.product._split_point(product_point)

        assert components[0].shape == point1.shape
        assert components[1].shape == point2.shape

    def test_combine_points_handles_different_shapes(self):
        """Test that _combine_points handles components with different shapes."""
        key1, key2 = jr.split(self.key, 2)

        # Different shaped components
        point1 = self.sphere2.random_point(key1)  # (3,)
        point2 = self.sphere3.random_point(key2)  # (4,)

        product_point = self.product._combine_points((point1, point2))

        # Total shape should be concatenation
        expected_size = point1.size + point2.size
        assert product_point.size == expected_size

    def test_split_combine_with_matrices(self):
        """Test split/combine with matrix-valued manifolds like SPD."""
        spd = SymmetricPositiveDefinite(n=2)
        sphere = Sphere(n=2)
        product = ProductManifold(manifolds=(spd, sphere))

        key1, key2 = jr.split(self.key, 2)

        spd_point = spd.random_point(key1)  # (2, 2)
        sphere_point = sphere.random_point(key2)  # (3,)

        # Combine and split
        product_point = product._combine_points((spd_point, sphere_point))
        recovered = product._split_point(product_point)

        assert len(recovered) == 2
        assert jnp.allclose(recovered[0].reshape(spd_point.shape), spd_point, atol=1e-12)
        assert jnp.allclose(recovered[1], sphere_point, atol=1e-12)


class TestProductManifoldGeometricOperations:
    """Test suite for ProductManifold geometric operations implementation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.key = jr.PRNGKey(456)
        self.sphere2 = Sphere(n=2)
        self.sphere3 = Sphere(n=3)
        self.product = ProductManifold(manifolds=(self.sphere2, self.sphere3))

    def test_exp_method_exists(self):
        """Test that exp method exists on ProductManifold."""
        assert hasattr(self.product, 'exp')
        assert callable(getattr(self.product, 'exp'))

    def test_log_method_exists(self):
        """Test that log method exists on ProductManifold."""
        assert hasattr(self.product, 'log')
        assert callable(getattr(self.product, 'log'))

    def test_inner_method_exists(self):
        """Test that inner method exists on ProductManifold."""
        assert hasattr(self.product, 'inner')
        assert callable(getattr(self.product, 'inner'))

    def test_proj_method_exists(self):
        """Test that proj method exists on ProductManifold."""
        assert hasattr(self.product, 'proj')
        assert callable(getattr(self.product, 'proj'))

    def test_dist_method_exists(self):
        """Test that dist method exists on ProductManifold."""
        assert hasattr(self.product, 'dist')
        assert callable(getattr(self.product, 'dist'))

    def test_exp_component_wise_functionality(self):
        """Test that exp applies component-wise exponential maps."""
        key1, key2, key3 = jr.split(self.key, 3)

        # Create product point and tangent vector
        point1 = self.sphere2.random_point(key1)
        point2 = self.sphere3.random_point(key2)
        product_point = self.product._combine_points((point1, point2))

        tangent1 = self.sphere2.random_tangent(key3, point1)
        tangent2 = self.sphere3.random_tangent(key3, point2)
        product_tangent = self.product._combine_points((tangent1, tangent2))

        # Apply product exponential map
        exp_result = self.product.exp(product_point, product_tangent)

        # Split result and compare with component exponential maps
        exp_components = self.product._split_point(exp_result)

        expected_comp1 = self.sphere2.exp(point1, tangent1)
        expected_comp2 = self.sphere3.exp(point2, tangent2)

        assert jnp.allclose(exp_components[0], expected_comp1, atol=1e-12)
        assert jnp.allclose(exp_components[1], expected_comp2, atol=1e-12)

    def test_log_component_wise_functionality(self):
        """Test that log applies component-wise logarithmic maps."""
        key1, key2 = jr.split(self.key, 2)

        # Create two product points
        point1_comp1 = self.sphere2.random_point(key1)
        point1_comp2 = self.sphere3.random_point(key1)
        product_point1 = self.product._combine_points((point1_comp1, point1_comp2))

        point2_comp1 = self.sphere2.random_point(key2)
        point2_comp2 = self.sphere3.random_point(key2)
        product_point2 = self.product._combine_points((point2_comp1, point2_comp2))

        # Apply product logarithmic map
        log_result = self.product.log(product_point1, product_point2)

        # Split result and compare with component logarithmic maps
        log_components = self.product._split_point(log_result)

        expected_comp1 = self.sphere2.log(point1_comp1, point2_comp1)
        expected_comp2 = self.sphere3.log(point1_comp2, point2_comp2)

        assert jnp.allclose(log_components[0], expected_comp1, atol=1e-12)
        assert jnp.allclose(log_components[1], expected_comp2, atol=1e-12)

    def test_inner_sum_of_component_inners(self):
        """Test that inner product is sum of component inner products."""
        key1, key2, key3 = jr.split(self.key, 3)

        # Create product point and two tangent vectors
        point1 = self.sphere2.random_point(key1)
        point2 = self.sphere3.random_point(key2)
        product_point = self.product._combine_points((point1, point2))

        tangent1_comp1 = self.sphere2.random_tangent(key3, point1)
        tangent1_comp2 = self.sphere3.random_tangent(key3, point2)
        product_tangent1 = self.product._combine_points((tangent1_comp1, tangent1_comp2))

        tangent2_comp1 = self.sphere2.random_tangent(key3, point1)
        tangent2_comp2 = self.sphere3.random_tangent(key3, point2)
        product_tangent2 = self.product._combine_points((tangent2_comp1, tangent2_comp2))

        # Compute product inner product
        product_inner = self.product.inner(product_point, product_tangent1, product_tangent2)

        # Compute sum of component inner products
        inner_comp1 = self.sphere2.inner(point1, tangent1_comp1, tangent2_comp1)
        inner_comp2 = self.sphere3.inner(point2, tangent1_comp2, tangent2_comp2)
        expected_inner = inner_comp1 + inner_comp2

        assert jnp.allclose(product_inner, expected_inner, atol=1e-12)

    def test_proj_component_wise_functionality(self):
        """Test that projection applies component-wise projections."""
        key1, key2, key3 = jr.split(self.key, 3)

        # Create product point and ambient vector
        point1 = self.sphere2.random_point(key1)
        point2 = self.sphere3.random_point(key2)
        product_point = self.product._combine_points((point1, point2))

        # Create ambient vector (not necessarily in tangent space)
        ambient1 = jr.normal(key3, point1.shape)
        ambient2 = jr.normal(key3, point2.shape)
        product_ambient = self.product._combine_points((ambient1, ambient2))

        # Apply product projection
        proj_result = self.product.proj(product_point, product_ambient)

        # Split result and compare with component projections
        proj_components = self.product._split_point(proj_result)

        expected_comp1 = self.sphere2.proj(point1, ambient1)
        expected_comp2 = self.sphere3.proj(point2, ambient2)

        assert jnp.allclose(proj_components[0], expected_comp1, atol=1e-12)
        assert jnp.allclose(proj_components[1], expected_comp2, atol=1e-12)

    def test_dist_euclidean_composition(self):
        """Test that distance follows Euclidean composition of component distances."""
        key1, key2 = jr.split(self.key, 2)

        # Create two product points
        point1_comp1 = self.sphere2.random_point(key1)
        point1_comp2 = self.sphere3.random_point(key1)
        product_point1 = self.product._combine_points((point1_comp1, point1_comp2))

        point2_comp1 = self.sphere2.random_point(key2)
        point2_comp2 = self.sphere3.random_point(key2)
        product_point2 = self.product._combine_points((point2_comp1, point2_comp2))

        # Compute product distance
        product_distance = self.product.dist(product_point1, product_point2)

        # Compute Euclidean composition of component distances
        dist_comp1 = self.sphere2.dist(point1_comp1, point2_comp1)
        dist_comp2 = self.sphere3.dist(point1_comp2, point2_comp2)
        expected_distance = jnp.sqrt(dist_comp1**2 + dist_comp2**2)

        assert jnp.allclose(product_distance, expected_distance, atol=1e-12)

    def test_exp_log_inverse_property(self):
        """Test that exp and log are inverse operations on product manifold."""
        key1, key2, key3 = jr.split(self.key, 3)

        # Create product point and tangent vector
        point1 = self.sphere2.random_point(key1)
        point2 = self.sphere3.random_point(key2)
        product_point = self.product._combine_points((point1, point2))

        tangent1 = self.sphere2.random_tangent(key3, point1)
        tangent2 = self.sphere3.random_tangent(key3, point2)
        product_tangent = self.product._combine_points((tangent1, tangent2))

        # Test exp(log_x(exp_x(v))) = exp_x(v)
        exp_result = self.product.exp(product_point, product_tangent)
        log_result = self.product.log(product_point, exp_result)

        # Use realistic tolerance for floating point operations on manifolds
        assert jnp.allclose(log_result, product_tangent, atol=1e-6)

    def test_inner_product_properties(self):
        """Test mathematical properties of the inner product."""
        key1, key2, key3, key4 = jr.split(self.key, 4)

        # Create product point and tangent vectors
        point1 = self.sphere2.random_point(key1)
        point2 = self.sphere3.random_point(key2)
        product_point = self.product._combine_points((point1, point2))

        tangent1_comp1 = self.sphere2.random_tangent(key3, point1)
        tangent1_comp2 = self.sphere3.random_tangent(key3, point2)
        u = self.product._combine_points((tangent1_comp1, tangent1_comp2))

        tangent2_comp1 = self.sphere2.random_tangent(key4, point1)
        tangent2_comp2 = self.sphere3.random_tangent(key4, point2)
        v = self.product._combine_points((tangent2_comp1, tangent2_comp2))

        # Test symmetry: <u, v> = <v, u>
        inner_uv = self.product.inner(product_point, u, v)
        inner_vu = self.product.inner(product_point, v, u)
        assert jnp.allclose(inner_uv, inner_vu, atol=1e-12)

        # Test positive definiteness: <u, u> >= 0
        inner_uu = self.product.inner(product_point, u, u)
        assert inner_uu >= 0

        # Test linearity in first argument
        a, b = 2.5, -1.3
        linear_combo = a * u + b * v
        inner_linear = self.product.inner(product_point, linear_combo, v)
        expected_linear = a * inner_uv + b * self.product.inner(product_point, v, v)
        assert jnp.allclose(inner_linear, expected_linear, atol=1e-10)

    def test_distance_properties(self):
        """Test mathematical properties of the distance function."""
        key1, key2, key3 = jr.split(self.key, 3)

        # Create three product points
        point1_comp1 = self.sphere2.random_point(key1)
        point1_comp2 = self.sphere3.random_point(key1)
        p = self.product._combine_points((point1_comp1, point1_comp2))

        point2_comp1 = self.sphere2.random_point(key2)
        point2_comp2 = self.sphere3.random_point(key2)
        q = self.product._combine_points((point2_comp1, point2_comp2))

        point3_comp1 = self.sphere2.random_point(key3)
        point3_comp2 = self.sphere3.random_point(key3)
        r = self.product._combine_points((point3_comp1, point3_comp2))

        # Test non-negativity: d(p, q) >= 0
        dist_pq = self.product.dist(p, q)
        assert dist_pq >= 0

        # Test identity of indiscernibles: d(p, p) = 0
        dist_pp = self.product.dist(p, p)
        assert jnp.allclose(dist_pp, 0.0, atol=1e-12)

        # Test symmetry: d(p, q) = d(q, p)
        dist_qp = self.product.dist(q, p)
        assert jnp.allclose(dist_pq, dist_qp, atol=1e-12)

        # Test triangle inequality: d(p, r) <= d(p, q) + d(q, r)
        dist_pr = self.product.dist(p, r)
        dist_qr = self.product.dist(q, r)
        assert dist_pr <= dist_pq + dist_qr + 1e-10  # Small tolerance for numerical errors


class TestProductManifoldRandomSampling:
    """Test suite for ProductManifold random sampling and batch processing functionality."""

    def setup_method(self):
        """Setup test fixtures for random sampling tests."""
        self.key = jr.PRNGKey(12345)

        # Create component manifolds with different dimensions
        self.sphere2 = Sphere(n=2)  # S² (3D ambient, 2D intrinsic)
        self.sphere3 = Sphere(n=3)  # S³ (4D ambient, 3D intrinsic)
        self.spd2 = SymmetricPositiveDefinite(n=2)  # 2×2 SPD (4D ambient, 3D intrinsic)
        self.spd3 = SymmetricPositiveDefinite(n=3)  # 3×3 SPD (9D ambient, 6D intrinsic)

        # Create various product manifolds
        self.two_spheres = ProductManifold(manifolds=(self.sphere2, self.sphere3))
        self.sphere_spd = ProductManifold(manifolds=(self.sphere2, self.spd2))
        self.mixed_dims = ProductManifold(manifolds=(self.sphere3, self.spd2, self.sphere2))

    def test_random_point_method_exists(self):
        """Test that random_point method exists on ProductManifold."""
        # This will fail initially - method doesn't exist yet
        assert hasattr(self.two_spheres, 'random_point')
        assert callable(self.two_spheres.random_point)

    def test_random_point_single_sample(self):
        """Test random_point method generates single valid samples."""
        # Test single sample generation
        point = self.two_spheres.random_point(self.key)

        # Point should be a flattened array with correct total ambient dimension
        expected_shape = (self.sphere2.ambient_dimension + self.sphere3.ambient_dimension,)
        assert point.shape == expected_shape

        # Split point and verify each component is valid on its manifold
        components = self.two_spheres._split_point(point)
        assert len(components) == 2

        # Verify component 1 is on Sphere(2) - should have unit norm
        assert jnp.allclose(jnp.linalg.norm(components[0]), 1.0, atol=1e-10)

        # Verify component 2 is on Sphere(3) - should have unit norm
        assert jnp.allclose(jnp.linalg.norm(components[1]), 1.0, atol=1e-10)

    def test_random_point_independent_component_sampling(self):
        """Test that components are sampled independently."""
        # Generate multiple samples to test statistical independence
        n_samples = 100
        keys = jr.split(self.key, n_samples)

        points = []
        for key in keys:
            point = self.sphere_spd.random_point(key)
            points.append(point)

        points = jnp.stack(points)

        # Split all points into components
        sphere_components = []
        spd_components = []

        for point in points:
            components = self.sphere_spd._split_point(point)
            sphere_components.append(components[0])
            spd_components.append(components[1])

        sphere_components = jnp.stack(sphere_components)
        spd_components = jnp.stack(spd_components)

        # Test statistical independence via correlation analysis
        # For independent samples, correlation should be near zero
        sphere_flat = sphere_components.reshape(n_samples, -1)
        spd_flat = spd_components.reshape(n_samples, -1)

        # Compute cross-correlation matrix
        sphere_centered = sphere_flat - jnp.mean(sphere_flat, axis=0)
        spd_centered = spd_flat - jnp.mean(spd_flat, axis=0)

        cross_corr = jnp.abs(jnp.corrcoef(
            jnp.concatenate([sphere_centered, spd_centered], axis=1)
        )[:sphere_centered.shape[1], sphere_centered.shape[1]:])

        # Cross-correlation should be small for independent sampling
        # Note: This is a statistical test and can be somewhat flaky
        # We test that most correlations are reasonably small, but allow some higher ones
        reasonable_correlations = jnp.mean(cross_corr < 0.7)
        assert reasonable_correlations > 0.8, f"Most component correlations should be small, got {reasonable_correlations:.2f} < 0.8"

    def test_random_point_batch_generation(self):
        """Test random_point method with batch_size parameter."""
        batch_size = 5

        # This should generate a batch of points
        batch_points = self.two_spheres.random_point(self.key, batch_size)

        # Should return (batch_size, ambient_dimension) array
        expected_shape = (batch_size, self.two_spheres.ambient_dimension)
        assert batch_points.shape == expected_shape

        # Each point in batch should be valid
        for i in range(batch_size):
            point = batch_points[i]
            components = self.two_spheres._split_point(point)
            assert is_point_on_manifold(self.sphere2, components[0])
            assert is_point_on_manifold(self.sphere3, components[1])

    def test_random_point_different_manifold_combinations(self):
        """Test random_point works with different manifold dimension combinations."""
        test_cases = [
            (self.two_spheres, "Two spheres of different dimensions"),
            (self.sphere_spd, "Sphere and SPD manifold"),
            (self.mixed_dims, "Three manifolds: Sphere(3) x SPD(2) x Sphere(2)")
        ]

        for product_manifold, description in test_cases:
            # Generate sample
            point = product_manifold.random_point(self.key)

            # Verify correct total dimension
            assert point.shape == (product_manifold.ambient_dimension,)

            # Verify components are valid
            components = product_manifold._split_point(point)
            assert len(components) == len(product_manifold.manifolds)

            # Check each component is on its respective manifold
            for component, manifold in zip(components, product_manifold.manifolds):
                assert is_point_on_manifold(manifold, component), f"Failed for {description}"

    def test_random_point_vmap_compatibility(self):
        """Test that random_point is compatible with jax.vmap for batch processing."""
        # Create multiple keys for vmap
        batch_size = 10
        keys = jr.split(self.key, batch_size)

        # Apply vmap to random_point
        vmap_random_point = jax.vmap(self.two_spheres.random_point)
        batch_points = vmap_random_point(keys)

        # Verify shape
        expected_shape = (batch_size, self.two_spheres.ambient_dimension)
        assert batch_points.shape == expected_shape

        # Verify each point is valid
        for i in range(batch_size):
            point = batch_points[i]
            components = self.two_spheres._split_point(point)
            assert is_point_on_manifold(self.sphere2, components[0])
            assert is_point_on_manifold(self.sphere3, components[1])

    def test_random_point_reproducibility(self):
        """Test that random_point is deterministic given the same key."""
        # Generate two points with same key
        point1 = self.sphere_spd.random_point(self.key)
        point2 = self.sphere_spd.random_point(self.key)

        # Should be identical
        assert jnp.allclose(point1, point2, atol=1e-12)

        # Generate with different keys - should be different
        key2 = jr.PRNGKey(54321)
        point3 = self.sphere_spd.random_point(key2)

        # Should be different
        assert not jnp.allclose(point1, point3, atol=1e-6)

    def test_random_tangent_method_exists(self):
        """Test that random_tangent method exists for ProductManifold."""
        # This will fail initially - method doesn't exist yet
        assert hasattr(self.two_spheres, 'random_tangent')
        assert callable(self.two_spheres.random_tangent)

    def test_random_tangent_single_sample(self):
        """Test random_tangent method generates valid tangent vectors."""
        # Generate base point
        base_point = self.two_spheres.random_point(self.key)

        # Generate tangent vector
        key_tangent = jr.fold_in(self.key, 1)
        tangent = self.two_spheres.random_tangent(key_tangent, base_point)

        # Should have same shape as base point
        assert tangent.shape == base_point.shape

        # Verify tangent vector is in tangent space (orthogonal to base point for spheres)
        components_base = self.two_spheres._split_point(base_point)
        components_tangent = self.two_spheres._split_point(tangent)

        # For sphere components, tangent should be orthogonal to base
        sphere2_inner = jnp.dot(components_base[0], components_tangent[0])
        sphere3_inner = jnp.dot(components_base[1], components_tangent[1])

        assert jnp.allclose(sphere2_inner, 0.0, atol=1e-6)
        assert jnp.allclose(sphere3_inner, 0.0, atol=1e-6)

    def test_random_tangent_batch_generation(self):
        """Test random_tangent with batch processing."""
        batch_size = 7

        # Generate base point and batch of tangent vectors
        base_point = self.sphere_spd.random_point(self.key)
        key_tangent = jr.fold_in(self.key, 1)

        batch_tangents = self.sphere_spd.random_tangent(key_tangent, base_point, batch_size)

        # Verify shape
        expected_shape = (batch_size, self.sphere_spd.ambient_dimension)
        assert batch_tangents.shape == expected_shape

        # Each tangent should be valid
        for i in range(batch_size):
            tangent = batch_tangents[i]

            # Verify is in tangent space using projection property
            projected = self.sphere_spd.proj(base_point, tangent)
            assert jnp.allclose(tangent, projected, atol=1e-10)

    def test_property_based_manifold_properties(self):
        """Property-based test verifying fundamental manifold properties."""
        n_trials = 20

        for trial in range(n_trials):
            key_trial = jr.fold_in(self.key, trial)

            # Generate random points
            point = self.mixed_dims.random_point(key_trial)

            # Test 1: Point should satisfy manifold constraints
            components = self.mixed_dims._split_point(point)
            for component, manifold in zip(components, self.mixed_dims.manifolds):
                assert is_point_on_manifold(manifold, component)

            # Test 2: Projection should be idempotent
            key_ambient = jr.fold_in(key_trial, 1)
            ambient_vec = jr.normal(key_ambient, point.shape)
            projected = self.mixed_dims.proj(point, ambient_vec)
            double_projected = self.mixed_dims.proj(point, projected)
            assert jnp.allclose(projected, double_projected, atol=1e-6)

            # Test 3: Exponential of zero tangent should return original point
            zero_tangent = jnp.zeros_like(point)
            exp_zero = self.mixed_dims.exp(point, zero_tangent)
            assert jnp.allclose(point, exp_zero, atol=1e-10)

    def test_statistical_properties_of_random_sampling(self):
        """Test statistical properties of the random sampling distribution."""
        # Generate large sample to test statistical properties
        n_samples = 1000
        keys = jr.split(self.key, n_samples)

        # Generate samples
        samples = []
        for key in keys:
            sample = self.two_spheres.random_point(key)
            samples.append(sample)
        samples = jnp.stack(samples)

        # Test 1: Mean should be near origin for spheres (due to symmetry)
        # This is a weak test but verifies distribution isn't concentrated
        mean_sample = jnp.mean(samples, axis=0)
        assert jnp.linalg.norm(mean_sample) < 0.2  # Should be small due to symmetry

        # Test 2: Standard deviation should be reasonable (not zero, not too large)
        std_sample = jnp.std(samples, axis=0)
        assert jnp.all(std_sample > 0.1)  # Not degenerate
        assert jnp.all(std_sample < 2.0)  # Not too spread out

        # Test 3: Samples should cover the manifold reasonably
        # Check that we get diverse samples by measuring pairwise distances
        n_check = min(50, n_samples)
        check_samples = samples[:n_check]

        min_distances = []
        for i in range(n_check):
            distances = []
            for j in range(n_check):
                if i != j:
                    dist = self.two_spheres.dist(check_samples[i], check_samples[j])
                    distances.append(dist)
            min_distances.append(jnp.min(jnp.array(distances)))

        min_distances = jnp.array(min_distances)
        # Minimum distance to nearest neighbor shouldn't be too small (good coverage)
        assert jnp.mean(min_distances) > 0.01
