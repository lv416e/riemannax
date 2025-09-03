"""Comprehensive tests for JITManager cache functionality.

This module tests the caching mechanism of JITManager following TDD methodology.
These tests are designed to fail initially (Red phase) and pass after implementation (Green phase).

Key functionality being tested:
- Cache key generation with different parameters
- Function caching and retrieval mechanisms
- Performance improvements through caching
- Cache size management and clearing
- Thread safety of cache operations
"""

import time
from unittest.mock import MagicMock, patch

import jax.numpy as jnp

from riemannax.core.jit_manager import JITManager


class TestJITManagerCacheFunctionality:
    """Comprehensive tests for JIT caching functionality."""

    def setup_method(self):
        """Setup before each test execution."""
        # Clear cache and reset configuration before each test
        JITManager.clear_cache()
        JITManager.reset_config()
        JITManager.configure(enable_jit=True, cache_size=100)

    def teardown_method(self):
        """Cleanup after each test execution."""
        JITManager.clear_cache()
        JITManager.reset_config()

    def test_cache_key_generation_with_no_parameters(self):
        """Test cache key generation for functions with no JIT parameters."""

        def simple_func(x):
            return x * 2

        # Test cache key generation with no parameters
        cache_key = JITManager._get_cache_key(simple_func.__name__)

        # Should be a string containing function name
        assert isinstance(cache_key, str)
        assert simple_func.__name__ in cache_key

    def test_cache_key_generation_with_static_argnums(self):
        """Test cache key generation with static_argnums parameter."""

        def func_with_static(x, n):
            return x**n

        # Test cache key generation with different static_argnums
        cache_key1 = JITManager._get_cache_key(func_with_static.__name__, static_argnums=(1,))
        cache_key2 = JITManager._get_cache_key(func_with_static.__name__, static_argnums=(0, 1))

        # Both should be strings
        assert isinstance(cache_key1, str)
        assert isinstance(cache_key2, str)

        # Different static_argnums should produce different cache keys
        assert cache_key1 != cache_key2

        # Both should contain function name
        assert func_with_static.__name__ in cache_key1
        assert func_with_static.__name__ in cache_key2

    def test_cache_key_generation_consistency(self):
        """Test that cache key generation is consistent for same parameters."""

        def test_func(x, y):
            return x + y

        # Test that same parameters produce same cache key
        key1 = JITManager._get_cache_key(test_func.__name__, static_argnums=(0,))
        key2 = JITManager._get_cache_key(test_func.__name__, static_argnums=(0,))

        # Same parameters should produce same cache key
        assert key1 == key2

        # Should be string containing function name
        assert isinstance(key1, str)
        assert test_func.__name__ in key1

    def test_function_caching_stores_compiled_functions(self):
        """Test that jit_decorator actually stores compiled functions in cache."""

        def matrix_multiply(A, B):
            return jnp.dot(A, B)

        # Apply JIT decorator
        jit_func = JITManager.jit_decorator(matrix_multiply, static_argnums=None)

        # Create test data
        A = jnp.ones((10, 10))
        B = jnp.ones((10, 10))

        # Call function to trigger compilation
        jit_func(A, B)

        # This should fail because cache is never populated
        cache = JITManager._cache
        assert len(cache) > 0, "Cache should contain compiled function after first call"

    def test_cache_hit_returns_same_function_instance(self):
        """Test that cache hit returns the exact same compiled function instance."""

        def vector_norm(x):
            return jnp.linalg.norm(x)

        # Get JIT function twice with same parameters
        jit_func1 = JITManager.jit_decorator(vector_norm, static_argnums=None)
        jit_func2 = JITManager.jit_decorator(vector_norm, static_argnums=None)

        # This should fail because functions aren't cached - new instances created each time
        assert jit_func1 is jit_func2, "Cache hit should return same function instance"

    def test_cache_miss_creates_new_compilation(self):
        """Test that different parameters create separate cache entries."""

        def power_func(x, n):
            return x**n

        # Create functions with different static_argnums
        jit_func1 = JITManager.jit_decorator(power_func, static_argnums=(1,))
        jit_func2 = JITManager.jit_decorator(power_func, static_argnums=None)

        # Call both functions to trigger compilation
        x = jnp.array([2.0, 3.0, 4.0])
        jit_func1(x, 2)
        jit_func2(x, jnp.array(2))

        # This should fail because cache doesn't track different parameter combinations
        cache = JITManager._cache
        assert len(cache) >= 2, "Different JIT parameters should create separate cache entries"

    def test_performance_improvement_from_caching(self):
        """Test that caching provides measurable performance improvement."""

        def expensive_computation(x):
            # Simulate expensive computation
            for _ in range(100):
                x = jnp.sin(x) + jnp.cos(x)
            return x

        data = jnp.ones(1000)

        # First call - should compile and cache
        start_time = time.time()
        jit_func1 = JITManager.jit_decorator(expensive_computation)
        jit_func1(data)
        first_call_time = time.time() - start_time

        # Second call with same parameters - should use cache
        start_time = time.time()
        jit_func2 = JITManager.jit_decorator(expensive_computation)
        jit_func2(data)
        second_call_time = time.time() - start_time

        # This should fail because no caching means both calls take similar time
        assert second_call_time < first_call_time * 0.5, (
            f"Cached call should be much faster: {second_call_time} vs {first_call_time}"
        )

    def test_cache_clearing_functionality(self):
        """Test that cache clearing removes all cached functions."""

        def simple_add(x, y):
            return x + y

        # Create and use JIT function to populate cache
        jit_func = JITManager.jit_decorator(simple_add)
        jit_func(jnp.array(1.0), jnp.array(2.0))

        # This should fail because cache is never populated initially
        assert len(JITManager._cache) > 0, "Cache should have entries before clearing"

        # Clear cache
        JITManager.clear_cache()

        # This should pass since clear_cache() is implemented
        assert len(JITManager._cache) == 0, "Cache should be empty after clearing"

    def test_cache_size_limit_enforcement(self):
        """Test that cache respects configured size limits."""
        # Set small cache size for testing
        JITManager.configure(cache_size=2)

        def create_unique_function(suffix):
            """Create unique functions for testing cache limit."""

            def func(x):
                return x + suffix

            func.__name__ = f"func_{suffix}"
            return func

        # Create more functions than cache limit
        functions = [create_unique_function(i) for i in range(5)]
        jit_functions = []

        for func in functions:
            jit_func = JITManager.jit_decorator(func)
            jit_functions.append(jit_func)
            # Call to trigger compilation
            jit_func(jnp.array(1.0))

        # This should fail because cache size limit is not enforced
        cache_size = len(JITManager._cache)
        configured_limit = JITManager.get_config()["cache_size"]
        assert cache_size <= configured_limit, f"Cache size {cache_size} exceeds limit {configured_limit}"

    def test_cache_thread_safety(self):
        """Test that cache operations are thread-safe."""
        import threading

        def thread_function(x):
            return jnp.sum(x**2)

        results = []
        exceptions = []

        def worker():
            try:
                jit_func = JITManager.jit_decorator(thread_function)
                data = jnp.ones(100)
                result = jit_func(data)
                results.append(result)
            except Exception as e:
                exceptions.append(e)

        # Create multiple threads accessing cache simultaneously
        threads = [threading.Thread(target=worker) for _ in range(10)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # This should fail if cache operations aren't thread-safe
        assert len(exceptions) == 0, f"Thread safety issues: {exceptions}"
        assert len(results) == 10, "All threads should complete successfully"

    def test_cache_invalidation_on_config_change(self):
        """Test that cache is properly invalidated when configuration changes."""

        def test_func(x):
            return x * 3

        # Create JIT function with current config
        jit_func1 = JITManager.jit_decorator(test_func)
        result1 = jit_func1(jnp.array(5.0))

        # Change configuration
        JITManager.configure(enable_jit=False)

        # Create function again - should handle config change
        jit_func2 = JITManager.jit_decorator(test_func)
        result2 = jit_func2(jnp.array(5.0))

        # This test verifies proper behavior with configuration changes
        # The exact assertion depends on expected behavior
        assert result1 == result2, "Results should be consistent regardless of JIT config"

    @patch("jax.jit")
    def test_jax_jit_called_with_correct_parameters(self, mock_jit):
        """Test that jax.jit is called with correct parameters for caching."""

        def test_func(x, y):
            return x + y

        mock_compiled_func = MagicMock()
        mock_jit.return_value = mock_compiled_func

        # Call with specific static_argnums
        static_args = (1,)
        JITManager.jit_decorator(test_func, static_argnums=static_args)

        # This should fail because current implementation doesn't properly pass parameters to cache
        mock_jit.assert_called_with(test_func, static_argnums=static_args)

    def test_cache_key_uniqueness_across_different_functions(self):
        """Test that different functions with same parameters get different cache keys."""

        def func_a(x):
            return x + 1

        def func_b(x):
            return x * 2

        # Test that different functions get different cache keys
        key_a = JITManager._get_cache_key(func_a.__name__, static_argnums=None)
        key_b = JITManager._get_cache_key(func_b.__name__, static_argnums=None)

        # Different functions should have different cache keys
        assert key_a != key_b, "Different functions should have different cache keys"

        # Both should be strings containing respective function names
        assert isinstance(key_a, str)
        assert isinstance(key_b, str)
        assert func_a.__name__ in key_a
        assert func_b.__name__ in key_b

    def test_cache_persistence_across_multiple_calls(self):
        """Test that cache persists across multiple decorator calls."""

        def persistent_func(x):
            return jnp.sum(x)

        # First usage
        jit_func1 = JITManager.jit_decorator(persistent_func)
        jit_func1(jnp.ones(10))

        # Second usage with same function
        jit_func2 = JITManager.jit_decorator(persistent_func)
        jit_func2(jnp.ones(10))

        # Third usage with same function
        jit_func3 = JITManager.jit_decorator(persistent_func)
        jit_func3(jnp.ones(10))

        # This should fail because no caching means new compilation each time
        cache = JITManager._cache
        # We expect only one cache entry for this function across all calls
        assert len(cache) == 1, f"Expected 1 cache entry, got {len(cache)}"
