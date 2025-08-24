"""JIT optimization decorator for separating JIT concerns from manifold logic.

This module implements a decorator pattern that cleanly separates JIT compilation
concerns from mathematical manifold operations, following the Single Responsibility
Principle and improving code maintainability.
"""

import functools
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import jax


class JITOptimizer:
    """JIT optimizer with LRU caching for compiled functions.

    This class manages JIT compilation with configurable caching to avoid
    recompilation overhead while maintaining memory efficiency.
    """

    def __init__(self, cache_size: int = 128):
        """Initialize JIT optimizer with specified cache size.

        Args:
            cache_size: Maximum number of compiled functions to cache (default: 128)
        """
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple[str, tuple[int, ...]], Callable[..., Any]] = OrderedDict()

    def compile(self, func: Callable[..., Any], static_args: tuple[int, ...] = ()) -> Callable[..., Any]:
        """Compile function with JIT and cache the result.

        Args:
            func: Function to compile
            static_args: Tuple of argument positions to treat as static

        Returns:
            JIT-compiled function
        """
        # Create cache key from function qualified name and static args to avoid conflicts
        # between methods with the same name across different classes
        qualified_name = f"{func.__qualname__}" if hasattr(func, "__qualname__") else func.__name__
        cache_key = (qualified_name, static_args)

        # Check if already cached
        if cache_key in self._cache:
            # Move to end (LRU)
            self._cache.move_to_end(cache_key)
            cached_func = self._cache[cache_key]
            return cached_func

        # Compile function with JIT
        compiled_func: Callable[..., Any] = jax.jit(func, static_argnums=static_args) if static_args else jax.jit(func)

        # Add to cache
        self._cache[cache_key] = compiled_func

        # Enforce cache size limit (LRU eviction)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)  # Remove least recently used

        return compiled_func

    def clear_cache(self) -> None:
        """Clear the JIT compilation cache."""
        self._cache.clear()


# Global optimizer instance for decorator usage
_global_optimizer = JITOptimizer()


def jit_optimized(static_args: tuple[int, ...] = ()) -> Callable[..., Any]:
    """Decorator for JIT optimization with caching support.

    This decorator applies JIT compilation to functions while maintaining
    a cache to avoid recompilation overhead. It cleanly separates JIT
    optimization concerns from the core mathematical logic.

    Args:
        static_args: Tuple of argument positions to treat as static during compilation

    Returns:
        Decorator function that applies JIT optimization

    Examples:
        >>> @jit_optimized()
        ... def exp_map(x: Array, v: Array) -> Array:
        ...     return x + v

        >>> @jit_optimized(static_args=(2,))
        ... def proj(x: Array, v: Array, dim: int) -> Array:
        ...     return v  # simplified example
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get compiled function from global optimizer
            compiled_func = _global_optimizer.compile(func, static_args)
            return compiled_func(*args, **kwargs)

        # Store original function for potential inspection
        wrapper._original_func = func  # type: ignore
        wrapper._static_args = static_args  # type: ignore

        return wrapper

    return decorator


def clear_jit_cache() -> None:
    """Clear the global JIT compilation cache.

    This is useful for testing or when memory usage needs to be reduced.
    """
    _global_optimizer.clear_cache()


def get_cache_info() -> dict[str, Any]:
    """Get information about the current JIT cache state.

    Returns:
        Dictionary with cache statistics including size and capacity
    """
    return {
        "cache_size": len(_global_optimizer._cache),
        "cache_capacity": _global_optimizer.cache_size,
        "cached_functions": list(_global_optimizer._cache.keys()),
    }
