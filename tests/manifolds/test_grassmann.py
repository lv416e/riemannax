"""Tests for Grassmann manifold implementation."""

import jax
import jax.numpy as jnp
import pytest

from riemannax.manifolds import DimensionError, Grassmann


class TestGrassmann:
    """Test suite for Grassmann manifold."""

    @pytest.fixture
    def manifold(self):
        """Create a Grassmann(5, 3) manifold for testing."""
        return Grassmann(5, 3)

    @pytest.fixture
    def point(self, manifold):
        """Generate a random point on the manifold."""
        key = jax.random.key(42)
        return manifold.random_point(key)

    @pytest.fixture
    def tangent(self, manifold, point):
        """Generate a random tangent vector."""
        key = jax.random.key(123)
        return manifold.random_tangent(key, point)

    def test_initialization(self):
        """Test manifold initialization."""
        manifold = Grassmann(5, 3)
        assert manifold.n == 5
        assert manifold.p == 3
        assert manifold.dimension == 3 * (5 - 3)
        assert manifold.ambient_dimension == 5 * 3

    def test_initialization_errors(self):
        """Test initialization error cases."""
        with pytest.raises(DimensionError):
            Grassmann(3, 5)  # p > n
        with pytest.raises(DimensionError):
            Grassmann(0, 3)  # n <= 0
        with pytest.raises(DimensionError):
            Grassmann(5, 0)  # p <= 0

    def test_random_point_properties(self, manifold):
        """Test that random points satisfy manifold constraints."""
        key = jax.random.key(42)
        x = manifold.random_point(key)

        # Check shape
        assert x.shape == (manifold.n, manifold.p)

        # Check orthonormality: X^T X = I
        should_be_identity = x.T @ x
        identity = jnp.eye(manifold.p)
        assert jnp.allclose(should_be_identity, identity, atol=1e-6)

        # Test validation
        assert manifold.validate_point(x)

    def test_random_tangent_properties(self, manifold, point):
        """Test that random tangent vectors are in tangent space."""
        key = jax.random.key(123)
        v = manifold.random_tangent(key, point)

        # Check shape
        assert v.shape == (manifold.n, manifold.p)

        # Check tangent space condition: X^T V = 0
        should_be_zero = point.T @ v
        assert jnp.allclose(should_be_zero, 0.0, atol=1e-6)

        # Test validation
        assert manifold.validate_tangent(point, v)

    def test_projection(self, manifold, point):
        """Test tangent space projection."""
        key = jax.random.key(456)
        v_ambient = jax.random.normal(key, (manifold.n, manifold.p))

        # Project to tangent space
        v_tangent = manifold.proj(point, v_ambient)

        # Check that projection is in tangent space
        assert manifold.validate_tangent(point, v_tangent)

        # Check that projecting twice gives same result
        v_double_proj = manifold.proj(point, v_tangent)
        assert jnp.allclose(v_tangent, v_double_proj, atol=1e-6)

    def test_exponential_map(self, manifold, point, tangent):
        """Test exponential map properties."""
        # Exponential map should give point on manifold
        y = manifold.exp(point, tangent)
        assert manifold.validate_point(y)

        # exp(x, 0) = x
        zero_tangent = jnp.zeros_like(tangent)
        y_zero = manifold.exp(point, zero_tangent)
        assert jnp.allclose(y_zero, point, atol=1e-6)

    def test_retraction(self, manifold, point, tangent):
        """Test retraction properties."""
        # Retraction should give point on manifold
        y = manifold.retr(point, tangent)
        assert manifold.validate_point(y)

        # retr(x, 0) = x
        zero_tangent = jnp.zeros_like(tangent)
        y_zero = manifold.retr(point, zero_tangent)
        assert jnp.allclose(y_zero, point, atol=1e-6)

    def test_logarithmic_map_inverse(self, manifold, point, tangent):
        """Test that log-exp provides reasonable approximation for small tangent vectors."""
        # Scale down tangent vector to ensure we're in injectivity radius
        small_tangent = 0.001 * tangent

        # exp followed by log should give approximate recovery
        y = manifold.exp(point, small_tangent)
        recovered_tangent = manifold.log(point, y)

        # With JIT-compiled exponential map implementation, expect good precision
        # (slightly relaxed tolerance to account for JIT compilation numerical differences)
        tangent_norm = manifold.norm(point, small_tangent)
        if tangent_norm > 1e-10:
            relative_error = manifold.norm(point, small_tangent - recovered_tangent) / tangent_norm
            assert relative_error < 1e-4, f"Relative error {relative_error:.2e} exceeds 1e-4 threshold"

    def test_distance_properties(self, manifold, point):
        """Test distance function properties."""
        # Distance to self is zero
        dist_self = manifold.dist(point, point)
        assert jnp.allclose(dist_self, 0.0, atol=1e-6)

        # Distance is symmetric
        key = jax.random.key(789)
        other_point = manifold.random_point(key)
        dist_xy = manifold.dist(point, other_point)
        dist_yx = manifold.dist(other_point, point)
        assert jnp.allclose(dist_xy, dist_yx, atol=1e-6)

        # Distance is non-negative
        assert dist_xy >= 0

    def test_inner_product_properties(self, manifold, point):
        """Test Riemannian inner product properties."""
        key = jax.random.key(321)
        u = manifold.random_tangent(key, point)
        v = manifold.random_tangent(jax.random.split(key)[1], point)

        # Symmetry
        uv = manifold.inner(point, u, v)
        vu = manifold.inner(point, v, u)
        assert jnp.allclose(uv, vu, atol=1e-6)

        # Positive definiteness
        uu = manifold.inner(point, u, u)
        assert uu >= 0

        # Linearity in first argument
        a = 2.0
        au_v = manifold.inner(point, a * u, v)
        a_uv = a * manifold.inner(point, u, v)
        assert jnp.allclose(au_v, a_uv, atol=1e-6)

    def test_parallel_transport(self, manifold, point, tangent):
        """Test parallel transport properties."""
        # Transport to same point is identity
        transported_same = manifold.transp(point, point, tangent)
        assert jnp.allclose(transported_same, tangent, atol=1e-6)

        # Transported vector is in target tangent space
        key = jax.random.key(654)
        target_point = manifold.random_point(key)
        transported = manifold.transp(point, target_point, tangent)
        assert manifold.validate_tangent(target_point, transported)

    def test_batched_operations(self, manifold):
        """Test batched operations."""
        key = jax.random.key(999)

        # Batch of random points
        batch_points = manifold.random_point(key, 3)
        assert batch_points.shape == (3, manifold.n, manifold.p)

        # All points should be valid
        for i in range(3):
            assert manifold.validate_point(batch_points[i])

    def test_numerical_stability(self, manifold):
        """Test numerical stability with edge cases."""
        # Very small manifold
        small_manifold = Grassmann(3, 1)
        key = jax.random.key(111)
        x = small_manifold.random_point(key)
        v = small_manifold.random_tangent(key, x)

        # Should not crash or produce NaNs
        y = small_manifold.exp(x, v * 1e-10)
        assert jnp.all(jnp.isfinite(y))
        assert small_manifold.validate_point(y)

    def test_representation(self, manifold):
        """Test string representation."""
        assert repr(manifold) == "Grassmann(5, 3)"
