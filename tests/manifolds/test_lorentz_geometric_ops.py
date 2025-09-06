"""Tests for Lorentz manifold geometric operations.

Tests cover the core geometric operations: exp, log, inner, dist, retr, proj.
"""

import jax
import jax.numpy as jnp
import pytest

from riemannax.manifolds import Lorentz


class TestLorentzExponentialMap:
    """Test exponential map operation."""

    def test_exp_zero_vector(self):
        """Test exponential of zero vector returns original point."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(42)
        
        x = manifold.random_point(key)
        zero_v = jnp.zeros_like(x)
        
        result = manifold.exp(x, zero_v)
        
        # Should return original point
        assert jnp.allclose(result, x, atol=1e-10)
        assert manifold.validate_point(result)

    def test_exp_validates_result(self):
        """Test exponential map always returns valid points."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(42)
        
        for i in range(10):
            subkey1, subkey2, key = jax.random.split(key, 3)
            x = manifold.random_point(subkey1)
            v = manifold.random_tangent(subkey2, x)
            
            result = manifold.exp(x, v)
            
            # Result should be valid point on hyperboloid
            assert manifold.validate_point(result)

    def test_exp_small_vectors(self):
        """Test exponential map with small tangent vectors."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(123)
        
        x = manifold.random_point(key)
        
        # Small tangent vector
        small_v = 1e-8 * manifold.random_tangent(key, x)
        
        result = manifold.exp(x, small_v)
        
        # Should be close to original point
        assert jnp.allclose(result, x, atol=1e-6)
        assert manifold.validate_point(result)


class TestLorentzLogarithmicMap:
    """Test logarithmic map operation."""

    def test_log_same_point(self):
        """Test log of same point returns zero vector."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(42)
        
        x = manifold.random_point(key)
        
        result = manifold.log(x, x)
        
        # Should be zero vector
        assert jnp.allclose(result, jnp.zeros_like(x), atol=1e-10)
        assert manifold.validate_tangent(x, result)

    def test_log_validates_result(self):
        """Test log always returns valid tangent vectors."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(42)
        
        for i in range(10):
            subkey1, subkey2, key = jax.random.split(key, 3)
            x = manifold.random_point(subkey1)
            y = manifold.random_point(subkey2)
            
            result = manifold.log(x, y)
            
            # Result should be valid tangent vector
            assert manifold.validate_tangent(x, result)

    def test_exp_log_consistency(self):
        """Test that exp and log are inverse operations."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(123)
        
        for i in range(5):
            subkey1, subkey2, key = jax.random.split(key, 3)
            x = manifold.random_point(subkey1)
            v = 0.1 * manifold.random_tangent(subkey2, x)  # Small tangent vector
            
            # exp followed by log should recover original vector
            y = manifold.exp(x, v)
            recovered_v = manifold.log(x, y)
            
            assert jnp.allclose(recovered_v, v, atol=1e-6)


class TestLorentzInnerProduct:
    """Test Riemannian inner product."""

    def test_inner_self_positive(self):
        """Test inner product of vector with itself is positive."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(42)
        
        x = manifold.random_point(key)
        v = manifold.random_tangent(key, x)
        
        # Skip zero vectors
        v_norm = jnp.linalg.norm(v)
        if v_norm > 1e-10:
            result = manifold.inner(x, v, v)
            assert result > 0, "Inner product with self should be positive"

    def test_inner_symmetry(self):
        """Test symmetry of inner product."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(42)
        
        for i in range(5):
            subkey1, subkey2, key = jax.random.split(key, 3)
            x = manifold.random_point(subkey1)
            u = manifold.random_tangent(subkey2, x)
            v = manifold.random_tangent(key, x)
            
            result1 = manifold.inner(x, u, v)
            result2 = manifold.inner(x, v, u)
            
            assert jnp.allclose(result1, result2, atol=1e-10)

    def test_inner_linearity(self):
        """Test linearity of inner product."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(42)
        
        x = manifold.random_point(key)
        u = manifold.random_tangent(key, x)
        v = manifold.random_tangent(key, x)
        
        a, b = 2.5, -1.3
        
        # Test <a*u + b*v, w> = a*<u,w> + b*<v,w>
        w = manifold.random_tangent(key, x)
        
        left = manifold.inner(x, a * u + b * v, w)
        right = a * manifold.inner(x, u, w) + b * manifold.inner(x, v, w)
        
        assert jnp.allclose(left, right, atol=1e-10)


class TestLorentzDistance:
    """Test distance function."""

    def test_dist_self_zero(self):
        """Test distance from point to itself is zero."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(42)
        
        x = manifold.random_point(key)
        
        result = manifold.dist(x, x)
        
        assert jnp.allclose(result, 0.0, atol=1e-10)

    def test_dist_positive(self):
        """Test distance is always non-negative."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(42)
        
        for i in range(10):
            subkey1, subkey2, key = jax.random.split(key, 3)
            x = manifold.random_point(subkey1)
            y = manifold.random_point(subkey2)
            
            result = manifold.dist(x, y)
            
            assert result >= 0, "Distance should be non-negative"

    def test_dist_symmetry(self):
        """Test distance symmetry."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(42)
        
        for i in range(5):
            subkey1, subkey2, key = jax.random.split(key, 3)
            x = manifold.random_point(subkey1)
            y = manifold.random_point(subkey2)
            
            dist_xy = manifold.dist(x, y)
            dist_yx = manifold.dist(y, x)
            
            assert jnp.allclose(dist_xy, dist_yx, atol=1e-10)

    def test_dist_exp_consistency(self):
        """Test distance consistency with exponential map."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(123)
        
        x = manifold.random_point(key)
        v = 0.5 * manifold.random_tangent(key, x)  # Small tangent vector
        
        # Distance should match tangent vector norm
        y = manifold.exp(x, v)
        
        dist_xy = manifold.dist(x, y)
        v_norm = jnp.sqrt(manifold.inner(x, v, v))
        
        assert jnp.allclose(dist_xy, v_norm, atol=1e-6)


class TestLorentzProjection:
    """Test projection operation."""

    def test_proj_on_manifold(self):
        """Test projection of point already on manifold."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(42)
        
        x = manifold.random_point(key)
        
        result = manifold.proj_to_manifold(x)
        
        # Should be unchanged and valid
        assert jnp.allclose(result, x, atol=1e-10)
        assert manifold.validate_point(result)

    def test_proj_validates_result(self):
        """Test projection always returns valid points."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(42)
        
        for i in range(10):
            subkey, key = jax.random.split(key)
            
            # Random point in ambient space
            ambient_point = jax.random.normal(subkey, (3,))
            
            result = manifold.proj_to_manifold(ambient_point)
            
            # Result should be valid point on hyperboloid
            assert manifold.validate_point(result)

    def test_proj_forward_sheet(self):
        """Test projection ensures forward sheet condition."""
        manifold = Lorentz(dimension=2)
        
        # Point that would be on backward sheet
        backward_point = jnp.array([-2.0, 1.0, jnp.sqrt(3.0)])
        
        result = manifold.proj_to_manifold(backward_point)
        
        # Should be on forward sheet
        assert result[0] > 0
        assert manifold.validate_point(result)


class TestLorentzRetraction:
    """Test retraction operation."""

    def test_retr_equals_exp(self):
        """Test retraction equals exponential map for this manifold."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(42)
        
        for i in range(5):
            subkey1, subkey2, key = jax.random.split(key, 3)
            x = manifold.random_point(subkey1)
            v = manifold.random_tangent(subkey2, x)
            
            retr_result = manifold.retr(x, v)
            exp_result = manifold.exp(x, v)
            
            assert jnp.allclose(retr_result, exp_result, atol=1e-10)

    def test_retr_validates_result(self):
        """Test retraction always returns valid points."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(42)
        
        for i in range(10):
            subkey1, subkey2, key = jax.random.split(key, 3)
            x = manifold.random_point(subkey1)
            v = manifold.random_tangent(subkey2, x)
            
            result = manifold.retr(x, v)
            
            assert manifold.validate_point(result)


class TestLorentzGeometricConsistency:
    """Test consistency between different geometric operations."""

    def test_all_operations_consistency(self):
        """Test consistency between all geometric operations."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(456)
        
        # Generate test points and vectors
        x = manifold.random_point(key)
        v = 0.3 * manifold.random_tangent(key, x)
        
        # Test consistency chain
        y = manifold.exp(x, v)
        recovered_v = manifold.log(x, y)
        
        # Exponential and logarithmic should be inverses
        assert jnp.allclose(recovered_v, v, atol=1e-6)
        
        # Distance should match vector norm
        dist_xy = manifold.dist(x, y)
        v_norm = jnp.sqrt(manifold.inner(x, v, v))
        assert jnp.allclose(dist_xy, v_norm, atol=1e-6)
        
        # Retraction should equal exponential
        y_retr = manifold.retr(x, v)
        assert jnp.allclose(y_retr, y, atol=1e-10)

    def test_batch_operations(self):
        """Test geometric operations work with batch dimensions."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(789)
        
        # Batch of points and vectors
        batch_size = 5
        x_batch = manifold.random_point(key, batch_size)
        v_batch = manifold.random_tangent(key, x_batch)
        
        # All operations should work with batches
        y_batch = manifold.exp(x_batch, v_batch)
        recovered_v_batch = manifold.log(x_batch, y_batch)
        dist_batch = manifold.dist(x_batch, y_batch)
        inner_batch = manifold.inner(x_batch, v_batch, v_batch)
        
        # Check shapes
        assert y_batch.shape == x_batch.shape
        assert recovered_v_batch.shape == v_batch.shape
        assert dist_batch.shape == (batch_size,)
        assert inner_batch.shape == (batch_size,)
        
        # Check all points are valid
        for i in range(batch_size):
            assert manifold.validate_point(y_batch[i])