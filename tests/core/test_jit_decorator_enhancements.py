"""Tests for enhanced JIT decorator functionality.

This module tests the advanced JIT optimization features including:
- Automatic static_argnums detection
- Conditional branching helpers using jax.lax.cond
- Enhanced caching mechanisms
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from unittest.mock import patch, Mock

from riemannax.core.jit_decorator import (
    jit_optimized,
    JITOptimizer,
    clear_jit_cache,
    get_cache_info,
    # These will be implemented
    auto_static_detection,
    conditional_jit_helper,
)


class TestAutoStaticDetection:
    """Test automatic static arguments detection functionality."""

    def test_auto_static_detection_with_scalar_args(self):
        """Test that scalar arguments are automatically detected as static."""

        @auto_static_detection
        @jit_optimized()
        def test_function(x: jax.Array, dim: int, tol: float) -> jax.Array:
            """Test function with mixed array and scalar arguments."""
            # Use JAX-compatible operations when dim is static
            return jax.lax.dynamic_slice(x, (0,), (dim,)) * tol

        # This should automatically detect dim (pos 1) and tol (pos 2) as static
        x = jnp.ones(10)
        result = test_function(x, 5, 0.1)

        assert result.shape == (5,)
        assert jnp.allclose(result, 0.1)

        # Verify that the function was compiled with correct static args
        assert hasattr(test_function, '_static_args')
        assert test_function._static_args == (1, 2)

    def test_auto_static_detection_with_no_scalars(self):
        """Test that functions with only array arguments have no static args."""

        @auto_static_detection
        @jit_optimized()
        def test_function(x: jax.Array, y: jax.Array) -> jax.Array:
            return x + y

        x = jnp.ones(3)
        y = jnp.ones(3)
        result = test_function(x, y)

        assert jnp.allclose(result, 2.0)
        assert test_function._static_args == ()

    def test_auto_static_detection_mixed_types(self):
        """Test detection with various scalar types."""

        @auto_static_detection
        @jit_optimized()
        def test_function(
            x: jax.Array,
            n: int,
            alpha: float,
            y: jax.Array,
            flag: bool
        ) -> jax.Array:
            # Since n and alpha are static, we can use them directly in conditional branches
            # by capturing them in closures rather than passing as arguments
            def true_branch():
                return jax.lax.dynamic_slice(x, (0,), (n,)) * alpha + jax.lax.dynamic_slice(y, (0,), (n,))

            def false_branch():
                return jax.lax.dynamic_slice(x, (0,), (n,)) * alpha

            # Use jax.lax.cond directly since static args are captured in closures
            return jax.lax.cond(flag, lambda: true_branch(), lambda: false_branch())

        x = jnp.ones(10)
        y = jnp.ones(10)
        result = test_function(x, 5, 2.0, y, True)

        assert result.shape == (5,)
        assert jnp.allclose(result, 3.0)
        assert test_function._static_args == (1, 2, 4)  # n, alpha, flag


class TestConditionalJITHelper:
    """Test conditional branching helper functions."""

    def test_conditional_jit_helper_basic(self):
        """Test basic conditional execution with jax.lax.cond."""

        def true_func(x):
            return x * 2

        def false_func(x):
            return x + 1

        x = jnp.array(5.0)

        # Test True condition
        result = conditional_jit_helper(True, true_func, false_func, x)
        assert result == 10.0

        # Test False condition
        result = conditional_jit_helper(False, true_func, false_func, x)
        assert result == 6.0

    def test_conditional_jit_helper_with_arrays(self):
        """Test conditional execution with array operations."""

        def exp_map_exact(x, v):
            # Simulate exact exponential map computation
            return x + v

        def exp_map_approx(x, v):
            # Simulate approximate computation
            return x + 0.9 * v

        x = jnp.ones(3)
        v = jnp.array([0.1, 0.2, 0.3])

        # Use exact method for small vectors
        norm_v = jnp.linalg.norm(v)
        use_exact = norm_v < 0.5

        result = conditional_jit_helper(use_exact, exp_map_exact, exp_map_approx, x, v)
        expected = x + v  # Should use exact method
        assert jnp.allclose(result, expected)

    def test_conditional_jit_helper_compilation(self):
        """Test that conditional helper is JIT compilable."""

        @jax.jit
        def compiled_conditional(pred, x):
            def true_func(x):
                return jnp.sin(x)

            def false_func(x):
                return jnp.cos(x)

            return conditional_jit_helper(pred, true_func, false_func, x)

        x = jnp.array(0.5)

        result_true = compiled_conditional(True, x)
        result_false = compiled_conditional(False, x)

        assert jnp.allclose(result_true, jnp.sin(x))
        assert jnp.allclose(result_false, jnp.cos(x))


class TestEnhancedJITOptimizer:
    """Test enhanced JIT optimizer functionality."""

    def test_enhanced_cache_with_static_args_detection(self):
        """Test that cache correctly handles automatically detected static args."""

        optimizer = JITOptimizer(cache_size=10)

        def test_func(x: jax.Array, dim: int) -> jax.Array:
            return x[:dim]

        # Manually specify static args for this test
        compiled_func1 = optimizer.compile(test_func, static_args=(1,))
        compiled_func2 = optimizer.compile(test_func, static_args=(1,))

        # Should return the same cached function
        assert compiled_func1 is compiled_func2

        # Cache should contain one entry
        assert len(optimizer._cache) == 1

    def test_cache_eviction_with_enhanced_keys(self):
        """Test LRU cache eviction with enhanced cache keys."""

        optimizer = JITOptimizer(cache_size=2)

        def func1(x): return x + 1
        def func2(x): return x - 1
        def func3(x): return x * 2

        # Fill cache
        optimizer.compile(func1)
        optimizer.compile(func2)
        assert len(optimizer._cache) == 2

        # Adding third function should evict first
        optimizer.compile(func3)
        assert len(optimizer._cache) == 2

        # func1 should be evicted (LRU)
        cache_keys = list(optimizer._cache.keys())
        assert any('func2' in key[0] for key in cache_keys)
        assert any('func3' in key[0] for key in cache_keys)

    def test_conditional_branching_integration(self):
        """Test integration of conditional branching with JIT optimization."""

        @jit_optimized()
        def adaptive_computation(x: jax.Array, use_exact: bool) -> jax.Array:
            """Function that chooses computation method based on condition."""

            def exact_method(x):
                return jnp.exp(x) - 1

            def approx_method(x):
                return x + 0.5 * x**2  # Taylor approximation

            return conditional_jit_helper(use_exact, exact_method, approx_method, x)

        x = jnp.array([0.1, 0.01, 0.001])

        # Test exact method
        result_exact = adaptive_computation(x, True)
        expected_exact = jnp.exp(x) - 1
        assert jnp.allclose(result_exact, expected_exact)

        # Test approximate method
        result_approx = adaptive_computation(x, False)
        expected_approx = x + 0.5 * x**2
        assert jnp.allclose(result_approx, expected_approx)


class TestJITCacheVerification:
    """Test JIT compilation cache verification."""

    def test_cache_info_functionality(self):
        """Test that cache info provides accurate statistics."""

        clear_jit_cache()  # Start with clean cache

        @jit_optimized()
        def test_func1(x): return x + 1

        @jit_optimized(static_args=(1,))
        def test_func2(x, n): return x[:n]

        # Initially empty
        info = get_cache_info()
        assert info["cache_size"] == 0

        # After first compilation
        test_func1(jnp.ones(3))
        info = get_cache_info()
        assert info["cache_size"] == 1

        # After second compilation
        test_func2(jnp.ones(5), 3)
        info = get_cache_info()
        assert info["cache_size"] == 2

        # Check cached functions
        cached_funcs = info["cached_functions"]
        assert len(cached_funcs) == 2

    def test_cache_clear_verification(self):
        """Test that cache clearing works correctly."""

        @jit_optimized()
        def test_func(x): return x * 2

        # Populate cache
        test_func(jnp.ones(3))
        info = get_cache_info()
        assert info["cache_size"] > 0

        # Clear cache
        clear_jit_cache()
        info = get_cache_info()
        assert info["cache_size"] == 0

    def test_cache_performance_with_repeated_calls(self):
        """Test cache performance benefits with repeated calls."""

        @jit_optimized()
        def expensive_computation(x: jax.Array) -> jax.Array:
            # Simulate expensive computation
            for _ in range(100):
                x = jnp.sin(x) + jnp.cos(x)
            return x

        x = jnp.ones(1000)

        # First call (compilation + execution)
        import time
        start = time.time()
        result1 = expensive_computation(x)
        first_call_time = time.time() - start

        # Second call (cached execution only)
        start = time.time()
        result2 = expensive_computation(x)
        second_call_time = time.time() - start

        # Results should be identical
        assert jnp.allclose(result1, result2)

        # Second call should be significantly faster
        # (This is a heuristic test - exact timing depends on system)
        if first_call_time > 0.01:  # Only test if first call was measurable
            assert second_call_time < first_call_time


if __name__ == "__main__":
    pytest.main([__file__])
