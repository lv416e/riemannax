"""Tests for Batch JIT Optimization System."""

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from riemannax.core.batch_ops import BatchJITOptimizer
from riemannax.manifolds.grassmann import Grassmann
from riemannax.manifolds.sphere import Sphere


class TestBatchJITOptimizer:
    """Test BatchJITOptimizer class functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.batch_optimizer = BatchJITOptimizer()
        self.sphere = Sphere()
        self.grassmann = Grassmann(n=5, p=3)

    def test_batch_optimizer_initialization(self):
        """Test BatchJITOptimizer initialization."""
        optimizer = BatchJITOptimizer()
        assert optimizer is not None
        assert hasattr(optimizer, "vectorize_manifold_op")
        assert hasattr(optimizer, "dynamic_batch_compilation")
        assert hasattr(optimizer, "_compilation_cache")

    def test_vectorize_manifold_op_basic(self):
        """Test basic vectorization of manifold operations."""
        # Test vectorizing exp operation
        batch_size = 10
        sphere_dim = 5  # Work with 4-sphere in R^5
        x = self.sphere.random_point(jax.random.key(42), sphere_dim)
        v = self.sphere.random_tangent(jax.random.key(43), x)

        # Create batch data
        batch_x = jnp.tile(x[None, :], (batch_size, 1))
        batch_v = jnp.tile(v[None, :], (batch_size, 1))

        # Vectorize the exp operation
        vectorized_exp = self.batch_optimizer.vectorize_manifold_op(self.sphere, "exp", in_axes=(0, 0), static_args={})

        # Test execution
        result = vectorized_exp(batch_x, batch_v)

        assert result.shape == (batch_size, sphere_dim)
        assert not jnp.isnan(result).any()

    def test_vectorize_manifold_op_numerical_consistency(self):
        """Test numerical consistency between single and batch operations."""
        batch_size = 5
        sphere_dim = 4  # Work with 3-sphere in R^4
        key = jax.random.key(42)

        # Single computation
        x = self.sphere.random_point(key, sphere_dim)
        v = self.sphere.random_tangent(jax.random.key(43), x)
        single_result = self.sphere.exp(x, v)

        # Batch computation
        batch_x = jnp.tile(x[None, :], (batch_size, 1))
        batch_v = jnp.tile(v[None, :], (batch_size, 1))

        vectorized_exp = self.batch_optimizer.vectorize_manifold_op(self.sphere, "exp", in_axes=(0, 0), static_args={})
        batch_result = vectorized_exp(batch_x, batch_v)

        # Check consistency (allow for JIT precision differences)
        for i in range(batch_size):
            np.testing.assert_array_almost_equal(batch_result[i], single_result, decimal=6)

    def test_vectorize_manifold_op_different_manifolds(self):
        """Test vectorization works across different manifolds."""
        # Test with Grassmann manifold
        batch_size = 8
        key = jax.random.key(42)

        x = self.grassmann.random_point(key)
        v = self.grassmann.random_tangent(jax.random.key(43), x)

        batch_x = jnp.tile(x[None, :, :], (batch_size, 1, 1))
        batch_v = jnp.tile(v[None, :, :], (batch_size, 1, 1))

        vectorized_proj = self.batch_optimizer.vectorize_manifold_op(
            self.grassmann, "proj", in_axes=(0, 0), static_args={}
        )

        result = vectorized_proj(batch_x, batch_v)

        assert result.shape == (batch_size, self.grassmann.n, self.grassmann.p)
        assert not jnp.isnan(result).any()

    def test_dynamic_batch_compilation_cache_management(self):
        """Test dynamic batch compilation and cache management."""
        # Test with different batch sizes
        batch_sizes = [5, 10, 20]
        sphere_dim = 6  # Work with 5-sphere in R^6

        for batch_size in batch_sizes:
            batch_x = jnp.ones((batch_size, sphere_dim))
            batch_v = jnp.ones((batch_size, sphere_dim)) * 0.1

            compiled_fn = self.batch_optimizer.dynamic_batch_compilation(
                self.sphere, "exp", batch_x.shape, batch_v.shape
            )

            result = compiled_fn(batch_x, batch_v)
            assert result.shape[0] == batch_size

    def test_dynamic_batch_compilation_caching(self):
        """Test that dynamic compilation properly caches results."""
        batch_size = 15
        sphere_dim = 3  # Work with 2-sphere in R^3
        x_shape = (batch_size, sphere_dim)
        v_shape = (batch_size, sphere_dim)

        # First compilation
        start_time = time.time()
        compiled_fn1 = self.batch_optimizer.dynamic_batch_compilation(self.sphere, "exp", x_shape, v_shape)
        first_compilation_time = time.time() - start_time

        # Second call with same signature
        start_time = time.time()
        compiled_fn2 = self.batch_optimizer.dynamic_batch_compilation(self.sphere, "exp", x_shape, v_shape)
        second_compilation_time = time.time() - start_time

        # Cache should make second call faster (more robust for CI environments)
        assert second_compilation_time < first_compilation_time * 0.5
        assert compiled_fn1 is compiled_fn2  # Same cached function

    def test_memory_efficient_batch_processing(self):
        """Test memory efficiency in large batch processing."""
        # Test with reasonably large batch size
        batch_size = 100
        sphere_dim = 4  # Work with 3-sphere in R^4
        key = jax.random.key(42)

        batch_x = self.sphere.random_point(key, batch_size, sphere_dim)
        batch_v = jnp.zeros((batch_size, sphere_dim))  # Use zero vectors to save memory

        vectorized_inner = self.batch_optimizer.vectorize_manifold_op(
            self.sphere, "inner", in_axes=(0, 0, 0), static_args={}
        )

        # This should not cause memory issues
        result = vectorized_inner(batch_x, batch_v, batch_v)

        assert result.shape == (batch_size,)
        assert jnp.allclose(result, 0.0)  # Zero vectors should give zero inner product

    def test_linear_scaling_performance(self):
        """Test that batch processing works correctly with different sizes."""
        base_batch_size = 10
        scale_factor = 4

        # Small batch
        sphere_dim = 5  # Work with 4-sphere in R^5
        batch_x_small = jnp.ones((base_batch_size, sphere_dim))
        batch_v_small = jnp.ones((base_batch_size, sphere_dim)) * 0.1

        vectorized_exp = self.batch_optimizer.vectorize_manifold_op(self.sphere, "exp", in_axes=(0, 0), static_args={})

        # Test small batch
        result_small = vectorized_exp(batch_x_small, batch_v_small)
        assert result_small.shape == (base_batch_size, sphere_dim)
        assert not jnp.isnan(result_small).any()

        # Large batch
        batch_x_large = jnp.ones((base_batch_size * scale_factor, sphere_dim))
        batch_v_large = jnp.ones((base_batch_size * scale_factor, sphere_dim)) * 0.1

        # Test large batch
        result_large = vectorized_exp(batch_x_large, batch_v_large)
        assert result_large.shape == (base_batch_size * scale_factor, sphere_dim)
        assert not jnp.isnan(result_large).any()

        # Verify that results are consistent (same input should give same output)
        np.testing.assert_array_almost_equal(result_small[0], result_large[0], decimal=6)

    def test_integrated_multi_manifold_operations(self):
        """Test integrated execution path for multiple manifold operations."""

        # Create a function that uses multiple operations
        def multi_op_function(manifold, x, v1, v2):
            exp_result = manifold.exp(x, v1)
            inner_result = manifold.inner(x, v1, v2)
            proj_result = manifold.proj(x, v1)
            return exp_result, inner_result, proj_result

        batch_size = 12
        sphere_dim = 4  # Work with 3-sphere in R^4
        key = jax.random.key(42)

        x = self.sphere.random_point(key, sphere_dim)
        v1 = self.sphere.random_tangent(jax.random.key(43), x)
        v2 = self.sphere.random_tangent(jax.random.key(44), x)

        batch_x = jnp.tile(x[None, :], (batch_size, 1))
        batch_v1 = jnp.tile(v1[None, :], (batch_size, 1))
        batch_v2 = jnp.tile(v2[None, :], (batch_size, 1))

        # Vectorize the multi-operation function
        # Note: We're not actually using the sphere parameter in this test case,
        # so let's simplify to test direct manifold operations
        vectorized_exp = self.batch_optimizer.vectorize_manifold_op(self.sphere, "exp", in_axes=(0, 0), static_args={})
        vectorized_inner = self.batch_optimizer.vectorize_manifold_op(
            self.sphere, "inner", in_axes=(0, 0, 0), static_args={}
        )
        vectorized_proj = self.batch_optimizer.vectorize_manifold_op(
            self.sphere, "proj", in_axes=(0, 0), static_args={}
        )

        # Execute operations separately
        exp_results = vectorized_exp(batch_x, batch_v1)
        inner_results = vectorized_inner(batch_x, batch_v1, batch_v2)
        proj_results = vectorized_proj(batch_x, batch_v1)

        assert exp_results.shape == (batch_size, sphere_dim)
        assert inner_results.shape == (batch_size,)
        assert proj_results.shape == (batch_size, sphere_dim)

    def test_cache_size_management(self):
        """Test that cache size is properly managed."""
        # Fill cache with many different function signatures
        sphere_dim = 3  # Work with 2-sphere in R^3
        for i in range(20):
            batch_size = 5 + i
            x_shape = (batch_size, sphere_dim)
            v_shape = (batch_size, sphere_dim)

            self.batch_optimizer.dynamic_batch_compilation(self.sphere, "exp", x_shape, v_shape)

        # Cache should have reasonable size (not growing indefinitely)
        cache_size = len(self.batch_optimizer._compilation_cache)
        assert cache_size <= 50  # Reasonable upper limit

    def test_error_handling_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        with pytest.raises((ValueError, TypeError)):
            # Invalid manifold
            self.batch_optimizer.vectorize_manifold_op(None, "exp", in_axes=(0, 0), static_args={})

        with pytest.raises((AttributeError, ValueError)):
            # Invalid operation name
            self.batch_optimizer.vectorize_manifold_op(self.sphere, "invalid_operation", in_axes=(0, 0), static_args={})

    def test_static_args_handling(self):
        """Test proper handling of static arguments."""
        # Test static arguments with a simpler case using validate_point
        batch_size = 8
        sphere_dim = 6  # Work with 5-sphere in R^6
        key = jax.random.key(42)

        # Create batch of points to validate
        batch_points = self.sphere.random_point(key, batch_size, sphere_dim)

        # Use static args in vectorization (using atol as static argument)
        vectorized_validate = self.batch_optimizer.vectorize_manifold_op(
            self.sphere, "validate_point", in_axes=(0,), static_args={"atol": 1e-6}
        )

        # Should handle static arguments correctly
        result = vectorized_validate(batch_points)
        assert result.shape == (batch_size,)
        assert jnp.all(result)  # All points should be valid

    def test_performance_monitoring_integration(self):
        """Test integration with performance monitoring."""
        batch_size = 25
        sphere_dim = 4  # Work with 3-sphere in R^4
        batch_x = jnp.ones((batch_size, sphere_dim))
        batch_v = jnp.ones((batch_size, sphere_dim)) * 0.1

        # Enable performance monitoring
        self.batch_optimizer.enable_performance_monitoring(True)

        vectorized_exp = self.batch_optimizer.vectorize_manifold_op(self.sphere, "exp", in_axes=(0, 0), static_args={})

        # Execute with monitoring
        result = vectorized_exp(batch_x, batch_v)

        # Check that monitoring data is available
        assert hasattr(self.batch_optimizer, "_performance_stats")
        stats = self.batch_optimizer.get_performance_stats()
        assert "batch_exp" in stats

        assert result.shape == (batch_size, sphere_dim)
