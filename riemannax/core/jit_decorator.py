"""JIT optimization decorator for separating JIT concerns from manifold logic.

This module implements a decorator pattern that cleanly separates JIT compilation
concerns from mathematical manifold operations, following the Single Responsibility
Principle and improving code maintainability.

Enhanced features:
- Automatic static arguments detection
- Conditional branching helpers using jax.lax.cond
- Advanced caching with intelligent cache key management
"""

import functools
import inspect
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Union

import jax
import jax.numpy as jnp


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
        # If function is wrapped, get the original function for compilation
        # but use the wrapper's name for caching
        target_func = func
        if hasattr(func, "_original_func"):
            target_func = func._original_func

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

        # Compile the target function (original, unwrapped) with JIT
        compiled_func: Callable[..., Any] = (
            jax.jit(target_func, static_argnums=static_args) if static_args else jax.jit(target_func)
        )

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

    Enhanced functionality:
    - Automatically uses static_args from @auto_static_detection if available
    - Falls back to provided static_args parameter
    - Supports manual override of auto-detected static args

    Args:
        static_args: Tuple of argument positions to treat as static during compilation.
                    If empty and function has auto-detected static args, uses those instead.

    Returns:
        Decorator function that applies JIT optimization

    Examples:
        >>> @jit_optimized()
        ... def exp_map(x: Array, v: Array) -> Array:
        ...     return x + v

        >>> @jit_optimized(static_args=(2,))
        ... def proj(x: Array, v: Array, dim: int) -> Array:
        ...     return v  # simplified example

        >>> @auto_static_detection
        ... @jit_optimized()
        ... def adaptive_func(x: Array, n: int) -> Array:
        ...     # n (position 1) automatically detected as static
        ...     return x[:n]
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Debug info for development
        # print(f"jit_optimized decorating: {func}")
        # print(f"  has _static_args: {hasattr(func, '_static_args')}")
        # print(f"  has _original_func: {hasattr(func, '_original_func')}")
        # if hasattr(func, '_static_args'):
        #     print(f"  _static_args value: {func._static_args}")
        # if hasattr(func, '_original_func'):
        #     print(f"  _original_func: {func._original_func}")
        #     if hasattr(func._original_func, '_static_args'):
        #         print(f"  _original_func._static_args: {func._original_func._static_args}")

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Determine which static_args to use
            effective_static_args = static_args

            # If no static_args provided, check for auto-detected ones on the function or its original
            # We check this at call time to handle decorator order issues
            if not static_args:
                # Check the wrapper itself (this is the decorated function)
                if hasattr(wrapper, "_static_args"):
                    effective_static_args = wrapper._static_args
                # Check the original function passed to this decorator
                elif hasattr(func, "_static_args"):
                    effective_static_args = func._static_args
                # Check nested wrapped functions
                elif hasattr(func, "_original_func") and hasattr(func._original_func, "_static_args"):
                    effective_static_args = func._original_func._static_args

            # Get the actual function to compile (unwrap if needed)
            target_func = func
            # Keep unwrapping until we get to the original function
            while hasattr(target_func, "_original_func"):
                target_func = target_func._original_func

            # Get compiled function from global optimizer
            compiled_func = _global_optimizer.compile(target_func, effective_static_args)
            return compiled_func(*args, **kwargs)

        # Store original function and static args for potential inspection
        wrapper._original_func = func  # type: ignore

        # Determine effective static args for storage - check nested wrappers
        effective_static_args = static_args
        if not static_args:
            if hasattr(func, "_static_args"):
                effective_static_args = func._static_args
            elif hasattr(func, "_original_func") and hasattr(func._original_func, "_static_args"):
                effective_static_args = func._original_func._static_args

        wrapper._static_args = effective_static_args  # type: ignore

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


def _detect_static_args(func: Callable[..., Any]) -> tuple[int, ...]:
    """Automatically detect which arguments should be static based on type hints.

    Static arguments are typically scalars (int, float, bool) rather than arrays.

    Args:
        func: Function to analyze for static arguments

    Returns:
        Tuple of argument positions that should be treated as static
    """
    try:
        sig = inspect.signature(func)
        static_positions = []

        for i, (_param_name, param) in enumerate(sig.parameters.items()):
            if param.annotation in (int, float, bool, type(None)):
                static_positions.append(i)
            elif hasattr(param.annotation, "__origin__") and param.annotation.__origin__ is Union:
                # Handle Union types like Optional[int] = Union[int, None]
                args = param.annotation.__args__
                # Check if it's a union with basic scalar types
                scalar_types = {int, float, bool, type(None)}
                if any(arg in scalar_types for arg in args) and len(args) <= 2 and type(None) in args:  # Optional type
                    non_none_types = [arg for arg in args if arg is not type(None)]
                    if len(non_none_types) == 1 and non_none_types[0] in {int, float, bool}:
                        static_positions.append(i)

        return tuple(static_positions)
    except (AttributeError, TypeError):
        # If we can't analyze the signature, return empty tuple
        return ()


def auto_static_detection(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that automatically detects static arguments for JIT compilation.

    This decorator analyzes function type hints to determine which arguments
    should be treated as static during JIT compilation. Static arguments are
    typically scalars (int, float, bool) that don't change shape or type.

    Args:
        func: Function to decorate with automatic static detection

    Returns:
        Decorated function with _static_args attribute set

    Examples:
        >>> @auto_static_detection
        ... @jit_optimized()
        ... def manifold_exp(x: Array, v: Array, max_iter: int) -> Array:
        ...     # max_iter (position 2) will be automatically detected as static
        ...     return x + v  # simplified
    """
    # Find the original function for static args detection
    original_func = func
    while hasattr(original_func, "_original_func"):
        original_func = original_func._original_func

    static_args = _detect_static_args(original_func)

    # Debug prints for development
    # print(f"auto_static_detection: func={func}")
    # print(f"auto_static_detection: original_func={original_func}")
    # print(f"auto_static_detection: detected static_args={static_args}")

    # If the function is already a jit_optimized wrapper, update its static args
    if hasattr(func, "_static_args") and hasattr(func, "_original_func"):
        # print("auto_static_detection: updating existing wrapper's static_args")
        func._static_args = static_args
        return func

    # Otherwise, create a new wrapper
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    # Store detected static args for inspection and use by jit_optimized
    wrapper._static_args = static_args  # type: ignore
    wrapper._original_func = func  # type: ignore

    # Debug verification
    # print(f"auto_static_detection: wrapper._static_args={getattr(wrapper, '_static_args', 'NOT SET')}")

    return wrapper


def conditional_jit_helper(
    condition: bool | jax.Array, true_func: Callable[..., Any], false_func: Callable[..., Any], *args: Any
) -> Any:
    """Helper function for conditional branching that is JIT-compilable.

    This function uses jax.lax.cond for efficient conditional execution
    that can be compiled with JAX's JIT compiler. It's particularly useful
    for switching between different computational methods based on runtime
    conditions (e.g., exact vs approximate algorithms based on input magnitude).

    Args:
        condition: Boolean condition or JAX array that evaluates to boolean
        true_func: Function to execute if condition is True
        false_func: Function to execute if condition is False
        *args: Arguments to pass to the selected function

    Returns:
        Result of executing either true_func or false_func with given args

    Examples:
        >>> def exact_method(x): return jnp.exp(x)
        >>> def approx_method(x): return 1 + x
        >>> x = jnp.array(0.01)
        >>> use_exact = jnp.linalg.norm(x) < 0.1
        >>> result = conditional_jit_helper(use_exact, exact_method, approx_method, x)
    """
    # Ensure condition is a JAX-compatible boolean
    if isinstance(condition, bool):
        condition = jnp.array(condition)
    elif not isinstance(condition, jax.Array):
        condition = jnp.array(bool(condition))

    # Use jax.lax.cond for JIT-compilable conditional execution
    return jax.lax.cond(condition, lambda args: true_func(*args), lambda args: false_func(*args), args)
