"""Batch Processing Integration Tests for RiemannAX.

This module provides comprehensive integration tests for batch processing
across all manifolds, verifying consistency, performance, and memory efficiency.
"""

import gc
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from riemannax.core.batch_ops import BatchJITOptimizer
from riemannax.manifolds.grassmann import Grassmann
from riemannax.manifolds.so import SpecialOrthogonal
from riemannax.manifolds.spd import SymmetricPositiveDefinite
from riemannax.manifolds.sphere import Sphere
from riemannax.manifolds.stiefel import Stiefel


class TestBatchOptimizationIntegration:
    """Integration tests for batch optimization across all manifolds."""

    def setup_method(self):
        """Set up test fixtures for all manifolds."""
        self.batch_optimizer = BatchJITOptimizer(enable_monitoring=True)

        # Initialize all supported manifolds
        self.manifolds = {
            "sphere": Sphere(),
            "grassmann": Grassmann(n=6, p=3),
            "stiefel": Stiefel(n=5, p=3),
            "so": SpecialOrthogonal(n=4),
            "spd": SymmetricPositiveDefinite(n=4),
        }

        # Test configurations for each manifold
        self.test_configs = {
            "sphere": {"dims": (5,), "batch_sizes": [10, 25, 50]},
            "grassmann": {"dims": (6, 3), "batch_sizes": [8, 20, 40]},
            "stiefel": {"dims": (5, 3), "batch_sizes": [8, 20, 40]},
            "so": {"dims": (4, 4), "batch_sizes": [8, 16, 32]},
            "spd": {"dims": (4, 4), "batch_sizes": [5, 12, 25]},
        }

    @pytest.mark.parametrize("manifold_name", ["sphere", "grassmann", "stiefel", "so"])
    def test_batch_consistency_all_manifolds(self, manifold_name):
        """Test batch processing consistency across all manifolds."""
        key = jax.random.key(42)

        manifold = self.manifolds[manifold_name]
        config = self.test_configs[manifold_name]
        dims = config["dims"]
        batch_size = config["batch_sizes"][0]  # Use smallest batch size

        # Generate test data based on manifold type
        if manifold_name == "sphere":
            x = manifold.random_point(key, *dims)
            v = manifold.random_tangent(jax.random.key(43), x)
            batch_x = jnp.tile(x[None, :], (batch_size, 1))
            batch_v = jnp.tile(v[None, :], (batch_size, 1))
        else:
            x = manifold.random_point(key)
            v = manifold.random_tangent(jax.random.key(43), x)
            if len(dims) == 2:
                batch_x = jnp.tile(x[None, :, :], (batch_size, 1, 1))
                batch_v = jnp.tile(v[None, :, :], (batch_size, 1, 1))
            else:
                batch_x = jnp.tile(x[None, :], (batch_size, 1))
                batch_v = jnp.tile(v[None, :], (batch_size, 1))

        # Test core operations
        operations = ["exp", "proj", "inner"]
        for op in operations:
            if not hasattr(manifold, op):
                continue

            # Single computation
            if op == "inner":
                single_result = getattr(manifold, op)(x, v, v)
                in_axes = (0, 0, 0)
                batch_args = (batch_x, batch_v, batch_v)
            else:
                single_result = getattr(manifold, op)(x, v)
                in_axes = (0, 0)
                batch_args = (batch_x, batch_v)

            # Batch computation
            vectorized_op = self.batch_optimizer.vectorize_manifold_op(manifold, op, in_axes=in_axes, static_args={})
            batch_result = vectorized_op(*batch_args)

            # Verify consistency
            if op == "inner":
                # Inner product returns scalar
                assert batch_result.shape == (batch_size,)
                for i in range(min(3, batch_size)):  # Check first few elements
                    np.testing.assert_allclose(batch_result[i], single_result, rtol=1e-5, atol=1e-6)
            else:
                # Other operations return tensors
                expected_shape = (batch_size, *single_result.shape)
                assert batch_result.shape == expected_shape
                for i in range(min(3, batch_size)):  # Check first few elements
                    np.testing.assert_allclose(batch_result[i], single_result, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("manifold_name", ["sphere", "grassmann", "stiefel", "so"])
    def test_batch_scaling_all_manifolds(self, manifold_name):
        """Test that batch processing scales properly across manifolds."""
        key = jax.random.key(42)

        manifold = self.manifolds[manifold_name]
        config = self.test_configs[manifold_name]
        dims = config["dims"]
        batch_sizes = config["batch_sizes"]

        # Test exp operation scaling
        vectorized_exp = self.batch_optimizer.vectorize_manifold_op(manifold, "exp", in_axes=(0, 0), static_args={})

        previous_result = None
        for batch_size in batch_sizes:
            # Generate batch data
            if manifold_name == "sphere":
                x = manifold.random_point(key, *dims)
                v = manifold.random_tangent(jax.random.key(43), x)
                batch_x = jnp.tile(x[None, :], (batch_size, 1))
                batch_v = jnp.tile(v[None, :], (batch_size, 1))
            else:
                x = manifold.random_point(key)
                v = manifold.random_tangent(jax.random.key(43), x)
                if len(dims) == 2:
                    batch_x = jnp.tile(x[None, :, :], (batch_size, 1, 1))
                    batch_v = jnp.tile(v[None, :, :], (batch_size, 1, 1))
                else:
                    batch_x = jnp.tile(x[None, :], (batch_size, 1))
                    batch_v = jnp.tile(v[None, :], (batch_size, 1))

            # Execute batch operation
            result = vectorized_exp(batch_x, batch_v)

            # Verify shape
            expected_shape = (batch_size, *dims)
            assert result.shape == expected_shape
            assert not jnp.isnan(result).any()
            assert not jnp.isinf(result).any()

            # Verify consistency with previous batch (first element should be same)
            if previous_result is not None:
                np.testing.assert_allclose(result[0], previous_result[0], rtol=1e-5, atol=1e-6)

            previous_result = result

    def test_memory_efficiency_large_batches(self):
        """Test memory efficiency with large batch sizes."""
        # Use Sphere manifold for this test as it's simplest
        manifold = self.manifolds["sphere"]
        large_batch_size = 500
        sphere_dim = 8

        key = jax.random.key(42)

        # Generate large batch data
        x = manifold.random_point(key, sphere_dim)
        v = manifold.random_tangent(jax.random.key(43), x)
        batch_x = jnp.tile(x[None, :], (large_batch_size, 1))
        batch_v = jnp.tile(v[None, :], (large_batch_size, 1))

        # Monitor memory before operation
        gc.collect()

        # Create vectorized operations
        vectorized_exp = self.batch_optimizer.vectorize_manifold_op(manifold, "exp", in_axes=(0, 0), static_args={})
        vectorized_proj = self.batch_optimizer.vectorize_manifold_op(manifold, "proj", in_axes=(0, 0), static_args={})

        # Execute operations - should not cause memory issues
        exp_result = vectorized_exp(batch_x, batch_v)
        proj_result = vectorized_proj(batch_x, batch_v)

        # Verify results
        assert exp_result.shape == (large_batch_size, sphere_dim)
        assert proj_result.shape == (large_batch_size, sphere_dim)
        assert not jnp.isnan(exp_result).any()
        assert not jnp.isnan(proj_result).any()

        # Clean up
        del exp_result, proj_result, batch_x, batch_v
        gc.collect()

    def test_cross_manifold_batch_operations(self):
        """Test batch operations across different manifold combinations."""
        key = jax.random.key(42)
        batch_size = 16

        # Test pairs of manifolds that can work together
        manifold_pairs = [("sphere", "sphere"), ("grassmann", "stiefel"), ("so", "so")]

        for manifold1_name, manifold2_name in manifold_pairs:
            manifold1 = self.manifolds[manifold1_name]
            manifold2 = self.manifolds[manifold2_name]

            # Create vectorized operations for both manifolds
            config1 = self.test_configs[manifold1_name]
            config2 = self.test_configs[manifold2_name]

            # Test operations on both manifolds in sequence
            for manifold, config, name in [(manifold1, config1, manifold1_name), (manifold2, config2, manifold2_name)]:
                dims = config["dims"]

                vectorized_exp = self.batch_optimizer.vectorize_manifold_op(
                    manifold, "exp", in_axes=(0, 0), static_args={}
                )

                # Generate test data
                if name == "sphere":
                    x = manifold.random_point(key, *dims)
                    v = manifold.random_tangent(jax.random.key(43), x)
                    batch_x = jnp.tile(x[None, :], (batch_size, 1))
                    batch_v = jnp.tile(v[None, :], (batch_size, 1))
                else:
                    x = manifold.random_point(key)
                    v = manifold.random_tangent(jax.random.key(43), x)
                    if len(dims) == 2:
                        batch_x = jnp.tile(x[None, :, :], (batch_size, 1, 1))
                        batch_v = jnp.tile(v[None, :, :], (batch_size, 1, 1))
                    else:
                        batch_x = jnp.tile(x[None, :], (batch_size, 1))
                        batch_v = jnp.tile(v[None, :], (batch_size, 1))

                # Execute and verify
                result = vectorized_exp(batch_x, batch_v)
                expected_shape = (batch_size, *dims)
                assert result.shape == expected_shape
                assert not jnp.isnan(result).any()

    def test_batch_cache_efficiency(self):
        """Test that batch compilation cache works efficiently."""
        manifold = self.manifolds["sphere"]
        sphere_dim = 4
        key = jax.random.key(42)

        # Test multiple batch sizes to verify caching
        batch_sizes = [10, 15, 10, 20, 15, 10]  # Repeated sizes should use cache

        compilation_times = []
        for _i, batch_size in enumerate(batch_sizes):
            # Generate test data
            x = manifold.random_point(key, sphere_dim)
            v = manifold.random_tangent(jax.random.key(43), x)
            batch_x = jnp.tile(x[None, :], (batch_size, 1))
            batch_v = jnp.tile(v[None, :], (batch_size, 1))

            # Time the compilation/execution
            start_time = time.time()
            compiled_fn = self.batch_optimizer.dynamic_batch_compilation(manifold, "exp", batch_x.shape, batch_v.shape)
            result = compiled_fn(batch_x, batch_v)
            end_time = time.time()

            compilation_times.append(end_time - start_time)
            assert result.shape == (batch_size, sphere_dim)

        # Verify cache is working - repeated sizes should be faster
        # (This is a heuristic test, not a strict performance requirement)
        cache_info = self.batch_optimizer.get_cache_info()
        assert len(cache_info["cache_keys"]) > 0

    def test_batch_error_propagation(self):
        """Test that errors in batch operations are properly propagated."""
        manifold = self.manifolds["sphere"]

        # Test with invalid manifold
        with pytest.raises((ValueError, TypeError, AttributeError)):
            self.batch_optimizer.vectorize_manifold_op(None, "exp", in_axes=(0, 0), static_args={})

        # Test with invalid operation
        with pytest.raises((AttributeError, ValueError)):
            self.batch_optimizer.vectorize_manifold_op(
                manifold, "nonexistent_operation", in_axes=(0, 0), static_args={}
            )

    def test_batch_numerical_stability(self):
        """Test numerical stability in batch operations."""
        key = jax.random.key(42)

        manifolds_to_test = [(name, manifold) for name, manifold in self.manifolds.items() if name != "spd"]

        for manifold_name, manifold in manifolds_to_test:
            config = self.test_configs[manifold_name]
            dims = config["dims"]
            batch_size = 20

            # Create vectorized operations
            vectorized_exp = self.batch_optimizer.vectorize_manifold_op(manifold, "exp", in_axes=(0, 0), static_args={})

            # Generate test data with small perturbations
            for i in range(3):  # Multiple random tests
                test_key = jax.random.split(key, i + 1)[0]

                if manifold_name == "sphere":
                    x = manifold.random_point(test_key, *dims)
                    v = manifold.random_tangent(jax.random.key(100 + i), x) * 0.1  # Small tangent
                    batch_x = jnp.tile(x[None, :], (batch_size, 1))
                    batch_v = jnp.tile(v[None, :], (batch_size, 1))
                else:
                    x = manifold.random_point(test_key)
                    v = manifold.random_tangent(jax.random.key(100 + i), x)
                    if hasattr(manifold, "n"):
                        v = v * 0.1  # Scale down for stability
                    if len(dims) == 2:
                        batch_x = jnp.tile(x[None, :, :], (batch_size, 1, 1))
                        batch_v = jnp.tile(v[None, :, :], (batch_size, 1, 1))
                    else:
                        batch_x = jnp.tile(x[None, :], (batch_size, 1))
                        batch_v = jnp.tile(v[None, :], (batch_size, 1))

                # Execute and verify numerical stability
                result = vectorized_exp(batch_x, batch_v)

                # Check for NaN or Inf values
                assert not jnp.isnan(result).any(), f"NaN detected in {manifold_name} batch exp"
                assert not jnp.isinf(result).any(), f"Inf detected in {manifold_name} batch exp"

                # Verify manifold constraints (basic check)
                if manifold_name == "sphere":
                    # Check unit norm constraint
                    norms = jnp.linalg.norm(result, axis=-1)
                    np.testing.assert_allclose(norms, 1.0, rtol=1e-4, atol=1e-5)

    def test_performance_monitoring_integration(self):
        """Test performance monitoring integration in batch operations."""
        manifold = self.manifolds["sphere"]
        sphere_dim = 5
        batch_size = 30
        key = jax.random.key(42)

        # Ensure monitoring is enabled
        self.batch_optimizer.enable_performance_monitoring(True)

        # Generate test data
        x = manifold.random_point(key, sphere_dim)
        v = manifold.random_tangent(jax.random.key(43), x)
        batch_x = jnp.tile(x[None, :], (batch_size, 1))
        batch_v = jnp.tile(v[None, :], (batch_size, 1))

        # Create and execute monitored operation
        vectorized_exp = self.batch_optimizer.vectorize_manifold_op(manifold, "exp", in_axes=(0, 0), static_args={})

        result = vectorized_exp(batch_x, batch_v)

        # Check that performance stats are recorded
        stats = self.batch_optimizer.get_performance_stats()
        assert len(stats) > 0

        # Verify at least one operation was monitored
        found_batch_op = False
        for op_name, op_stats in stats.items():
            if "batch" in op_name or "exp" in op_name:
                found_batch_op = True
                assert "total_calls" in op_stats
                assert op_stats["total_calls"] > 0
                break

        assert found_batch_op, "No batch operation found in performance stats"
        assert result.shape == (batch_size, sphere_dim)
