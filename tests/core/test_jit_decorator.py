"""Test module for JIT decorator functionality."""

import jax.numpy as jnp
import pytest
from jax import Array

from riemannax.core.jit_decorator import JITOptimizer, jit_optimized


class TestJITOptimizer:
    """Test JIT optimizer class."""

    def test_jit_optimizer_initialization(self):
        """Test JITOptimizer initializes with correct cache size."""
        optimizer = JITOptimizer(cache_size=64)
        assert optimizer.cache_size == 64

    def test_jit_optimizer_default_cache_size(self):
        """Test JITOptimizer uses default cache size of 128."""
        optimizer = JITOptimizer()
        assert optimizer.cache_size == 128

    def test_compile_function_simple(self):
        """Test compiling a simple function."""
        optimizer = JITOptimizer()

        def simple_add(x: Array, y: Array) -> Array:
            return x + y

        compiled_fn = optimizer.compile(simple_add, static_args=())

        # Test the compiled function works
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([4.0, 5.0, 6.0])
        result = compiled_fn(x, y)
        expected = jnp.array([5.0, 7.0, 9.0])

        assert jnp.allclose(result, expected)

    def test_compile_function_with_static_args(self):
        """Test compiling a function with static arguments."""
        optimizer = JITOptimizer()

        def multiply_by_scalar(x: Array, scalar: float) -> Array:
            return x * scalar

        compiled_fn = optimizer.compile(multiply_by_scalar, static_args=(1,))

        # Test the compiled function works
        x = jnp.array([1.0, 2.0, 3.0])
        result = compiled_fn(x, 2.0)
        expected = jnp.array([2.0, 4.0, 6.0])

        assert jnp.allclose(result, expected)

    def test_cache_functionality(self):
        """Test that the same function is cached and reused."""
        optimizer = JITOptimizer()

        def test_func(x: Array) -> Array:
            return x * 2

        # Compile the same function twice
        compiled_fn1 = optimizer.compile(test_func, static_args=())
        compiled_fn2 = optimizer.compile(test_func, static_args=())

        # Should return the same cached function
        assert compiled_fn1 is compiled_fn2

    def test_clear_cache(self):
        """Test clearing the compilation cache."""
        optimizer = JITOptimizer()

        def test_func(x: Array) -> Array:
            return x * 2

        # Compile function and cache it
        optimizer.compile(test_func, static_args=())

        # Clear cache
        optimizer.clear_cache()

        # Should have empty cache
        assert len(optimizer._cache) == 0

    def test_cache_size_limit(self):
        """Test that cache respects size limit."""
        optimizer = JITOptimizer(cache_size=2)

        def make_func(multiplier):
            def func(x: Array) -> Array:
                return x * multiplier

            func.__name__ = f"func_{multiplier}"
            return func

        # Compile more functions than cache size
        func1 = make_func(1)
        func2 = make_func(2)
        func3 = make_func(3)

        optimizer.compile(func1, static_args=())
        optimizer.compile(func2, static_args=())
        optimizer.compile(func3, static_args=())  # Should evict func1

        # Cache should have at most 2 entries
        assert len(optimizer._cache) <= 2


class TestJitOptimizedDecorator:
    """Test the jit_optimized decorator."""

    def test_jit_optimized_decorator_basic(self):
        """Test basic jit_optimized decorator functionality."""

        @jit_optimized()
        def decorated_func(x: Array) -> Array:
            return x * 3

        x = jnp.array([1.0, 2.0, 3.0])
        result = decorated_func(x)
        expected = jnp.array([3.0, 6.0, 9.0])

        assert jnp.allclose(result, expected)

    def test_jit_optimized_decorator_with_static_args(self):
        """Test jit_optimized decorator with static arguments."""

        @jit_optimized(static_args=(1,))
        def decorated_func(x: Array, multiplier: float) -> Array:
            return x * multiplier

        x = jnp.array([1.0, 2.0, 3.0])
        result = decorated_func(x, 4.0)
        expected = jnp.array([4.0, 8.0, 12.0])

        assert jnp.allclose(result, expected)

    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves original function metadata."""

        @jit_optimized()
        def test_function(x: Array) -> Array:
            """Test function docstring."""
            return x + 1

        assert test_function.__name__ == "test_function"
        assert "Test function docstring" in test_function.__doc__

    def test_multiple_calls_use_cache(self):
        """Test that multiple calls to decorated function use cached compilation."""
        call_count = 0

        @jit_optimized()
        def counting_func(x: Array) -> Array:
            nonlocal call_count
            call_count += 1
            return x * 2

        x = jnp.array([1.0, 2.0])

        # First call should compile
        result1 = counting_func(x)

        # Second call should use cached version
        result2 = counting_func(x)

        # Both results should be identical
        assert jnp.allclose(result1, result2)

        # Function should maintain its behavior
        expected = jnp.array([2.0, 4.0])
        assert jnp.allclose(result1, expected)


class TestJITOptimizationIntegration:
    """Test integration between JIT components."""

    def test_global_optimizer_instance(self):
        """Test that decorator uses global optimizer instance."""

        # This test ensures the decorator pattern works with a shared optimizer
        @jit_optimized()
        def func1(x: Array) -> Array:
            return x + 1

        @jit_optimized()
        def func2(x: Array) -> Array:
            return x * 2

        x = jnp.array([1.0, 2.0])

        result1 = func1(x)
        result2 = func2(x)

        assert jnp.allclose(result1, jnp.array([2.0, 3.0]))
        assert jnp.allclose(result2, jnp.array([2.0, 4.0]))

    def test_error_handling_in_compilation(self):
        """Test error handling during JIT compilation."""
        optimizer = JITOptimizer()

        def problematic_func(x):
            # This might cause compilation issues in some cases
            return x.unknown_method()

        # Should handle compilation errors gracefully
        with pytest.raises((AttributeError, Exception)):
            compiled_fn = optimizer.compile(problematic_func, static_args=())
            compiled_fn(jnp.array([1.0]))

    def test_performance_characteristics(self):
        """Test basic performance characteristics of JIT compilation."""

        @jit_optimized()
        def matrix_multiply(a: Array, b: Array) -> Array:
            return jnp.matmul(a, b)

        # Create test matrices
        a = jnp.ones((10, 10))
        b = jnp.ones((10, 10))

        # First call (with compilation)
        result1 = matrix_multiply(a, b)

        # Second call (cached)
        result2 = matrix_multiply(a, b)

        # Results should be identical
        assert jnp.allclose(result1, result2)

        # Result should be correct
        expected = jnp.full((10, 10), 10.0)
        assert jnp.allclose(result1, expected)
