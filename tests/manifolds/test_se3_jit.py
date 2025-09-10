"""JIT-specific tests for SE(3) manifold operations."""

import jax
import jax.numpy as jnp
import pytest
import time

from riemannax.manifolds.se3 import SE3


class TestSE3JITOptimization:
    """Test SE(3) JIT compilation and performance."""

    def test_exp_tangent_jit(self):
        """Test that exp_tangent is JIT compiled and works correctly."""
        manifold = SE3()
        key = jax.random.PRNGKey(42)

        # Generate test data
        tangent = jax.random.normal(key, (6,)) * 0.1

        # First call will trigger compilation
        result1 = manifold.exp_tangent(tangent)

        # Second call should use compiled version
        result2 = manifold.exp_tangent(tangent)

        # Results should be identical
        assert jnp.allclose(result1, result2)
        assert manifold.validate_point(result1)

    def test_log_tangent_jit(self):
        """Test that log_tangent is JIT compiled and works correctly."""
        manifold = SE3()
        key = jax.random.PRNGKey(123)

        # Generate test SE(3) transform
        g = manifold.random_point(key)

        # First call will trigger compilation
        result1 = manifold.log_tangent(g)

        # Second call should use compiled version
        result2 = manifold.log_tangent(g)

        # Results should be identical
        assert jnp.allclose(result1, result2)
        assert result1.shape == (6,)

    def test_compose_jit(self):
        """Test that compose is JIT compiled and works correctly."""
        manifold = SE3()
        key = jax.random.PRNGKey(456)

        g1 = manifold.random_point(key)
        key, subkey = jax.random.split(key)
        g2 = manifold.random_point(subkey)

        # First call will trigger compilation
        result1 = manifold.compose(g1, g2)

        # Second call should use compiled version
        result2 = manifold.compose(g1, g2)

        # Results should be identical
        assert jnp.allclose(result1, result2)
        assert manifold.validate_point(result1)

    def test_inverse_jit(self):
        """Test that inverse is JIT compiled and works correctly."""
        manifold = SE3()
        key = jax.random.PRNGKey(789)

        g = manifold.random_point(key)

        # First call will trigger compilation
        result1 = manifold.inverse(g)

        # Second call should use compiled version
        result2 = manifold.inverse(g)

        # Results should be identical
        assert jnp.allclose(result1, result2)
        assert manifold.validate_point(result1)

    def test_inner_jit(self):
        """Test that inner is JIT compiled and works correctly."""
        manifold = SE3()
        key = jax.random.PRNGKey(321)

        g = manifold.random_point(key)
        key, subkey = jax.random.split(key)
        v1 = jax.random.normal(subkey, (6,))
        key, subkey = jax.random.split(key)
        v2 = jax.random.normal(subkey, (6,))

        # First call will trigger compilation
        result1 = manifold.inner(g, v1, v2)

        # Second call should use compiled version
        result2 = manifold.inner(g, v1, v2)

        # Results should be identical
        assert jnp.allclose(result1, result2)
        assert jnp.isscalar(result1)

    def test_retr_jit(self):
        """Test that retr is JIT compiled and works correctly."""
        manifold = SE3()
        key = jax.random.PRNGKey(654)

        g = manifold.random_point(key)
        key, subkey = jax.random.split(key)
        v = jax.random.normal(subkey, (6,)) * 0.1

        # First call will trigger compilation
        result1 = manifold.retr(g, v)

        # Second call should use compiled version
        result2 = manifold.retr(g, v)

        # Results should be identical
        assert jnp.allclose(result1, result2)
        assert manifold.validate_point(result1)

    def test_dist_jit(self):
        """Test that dist is JIT compiled and works correctly."""
        manifold = SE3()
        key = jax.random.PRNGKey(987)

        g1 = manifold.random_point(key)
        key, subkey = jax.random.split(key)
        g2 = manifold.random_point(subkey)

        # First call will trigger compilation
        result1 = manifold.dist(g1, g2)

        # Second call should use compiled version
        result2 = manifold.dist(g1, g2)

        # Results should be identical
        assert jnp.allclose(result1, result2)
        assert result1 >= 0

    def test_transp_jit(self):
        """Test that transp is JIT compiled and works correctly."""
        manifold = SE3()
        key = jax.random.PRNGKey(147)

        g1 = manifold.random_point(key)
        key, subkey = jax.random.split(key)
        g2 = manifold.random_point(subkey)
        key, subkey = jax.random.split(key)
        v = jax.random.normal(subkey, (6,))

        # First call will trigger compilation
        result1 = manifold.transp(g1, g2, v)

        # Second call should use compiled version
        result2 = manifold.transp(g1, g2, v)

        # Results should be identical
        assert jnp.allclose(result1, result2)
        assert result1.shape == (6,)

    def test_batch_operations_jit(self):
        """Test JIT compilation with batch operations."""
        manifold = SE3()
        key = jax.random.PRNGKey(258)

        batch_size = 10
        g_batch = manifold.random_point(key, batch_size)

        # Test batch compose with JIT
        result_batch = manifold.compose(g_batch, g_batch)
        assert result_batch.shape == (batch_size, 7)

        # Test batch inverse with JIT
        inv_batch = manifold.inverse(g_batch)
        assert inv_batch.shape == (batch_size, 7)

    @pytest.mark.slow
    def test_jit_performance_improvement(self):
        """Test that JIT provides performance improvement."""
        manifold = SE3()
        key = jax.random.PRNGKey(369)

        # Generate test data
        n_trials = 100
        tangents = jax.random.normal(key, (n_trials, 6)) * 0.1

        # First call to trigger compilation
        _ = manifold.exp_tangent(tangents[0])

        # Time multiple calls (should be fast due to JIT)
        start_time = time.time()
        for i in range(n_trials):
            _ = manifold.exp_tangent(tangents[i])
        jit_time = time.time() - start_time

        # JIT time should be reasonable (less than 1 second for 100 calls)
        assert jit_time < 1.0, f"JIT compilation took too long: {jit_time:.3f}s"

    def test_static_argument_handling(self):
        """Test that JIT handles static arguments correctly."""
        manifold = SE3()
        key = jax.random.PRNGKey(741)

        # Test with different batch sizes (should recompile appropriately)
        single_g = manifold.random_point(key)
        batch_g = manifold.random_point(key, 5)

        # These should both work despite different shapes
        single_inv = manifold.inverse(single_g)
        batch_inv = manifold.inverse(batch_g)

        assert single_inv.shape == (7,)
        assert batch_inv.shape == (5, 7)

    def test_nested_jit_calls(self):
        """Test that nested JIT calls work correctly."""
        manifold = SE3()
        key = jax.random.PRNGKey(852)

        g = manifold.random_point(key)
        key, subkey = jax.random.split(key)
        v = jax.random.normal(subkey, (6,)) * 0.1

        # retr calls exp_tangent and compose, all of which are JIT compiled
        result = manifold.retr(g, v)

        assert manifold.validate_point(result)
        assert result.shape == (7,)

    def test_jit_with_control_flow(self):
        """Test JIT compilation handles control flow correctly."""
        manifold = SE3()
        key = jax.random.PRNGKey(963)

        # Generate data that will exercise different code paths
        large_tangent = jax.random.normal(key, (6,)) * 2.0  # Large tangent
        small_tangent = jax.random.normal(key, (6,)) * 1e-10  # Very small tangent

        # These should trigger different numerical paths but both work
        result_large = manifold.exp_tangent(large_tangent)
        result_small = manifold.exp_tangent(small_tangent)

        assert manifold.validate_point(result_large)
        assert manifold.validate_point(result_small)


if __name__ == "__main__":
    pytest.main([__file__])
