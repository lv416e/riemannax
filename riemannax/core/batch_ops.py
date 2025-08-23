"""Batch JIT Optimization System for RiemannAX.

This module provides efficient batch processing capabilities for manifold operations
using JAX's vmap and JIT compilation features.
"""

import functools
import hashlib
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import jax

from .jit_manager import JITManager
from .performance import PerformanceMonitor


class BatchJITOptimizer:
    """Batch JIT optimization system for manifold operations.

    This class provides efficient batch processing by combining JAX's vmap
    with JIT compilation for optimal performance on manifold operations.

    Features:
    - Dynamic batch size compilation and caching
    - Memory-efficient vectorized operations
    - Integration with performance monitoring
    - Automatic fallback mechanisms
    """

    def __init__(self, cache_size: int = 100, enable_monitoring: bool = False):
        """Initialize BatchJITOptimizer.

        Args:
            cache_size: Maximum number of cached compiled functions
            enable_monitoring: Whether to enable performance monitoring
        """
        self._compilation_cache: dict[str, Callable] = {}
        self._cache_size = cache_size
        self._performance_monitoring = enable_monitoring
        self._performance_stats: dict[str, dict[str, Any]] = defaultdict(dict)
        self._jit_manager = JITManager()
        self._performance_monitor = PerformanceMonitor() if enable_monitoring else None

    def vectorize_manifold_op(
        self,
        manifold: Any,
        operation: str | Callable,
        in_axes: int | tuple[int, ...] | None = None,
        static_args: dict[str, Any] | None = None,
    ) -> Callable:
        """Vectorize a manifold operation using JAX vmap and JIT.

        Args:
            manifold: The manifold instance
            operation: Operation name (string) or callable function
            in_axes: Axes to vectorize over (passed to vmap)
            static_args: Static arguments for the operation

        Returns:
            Vectorized and JIT-compiled function

        Raises:
            ValueError: If manifold or operation is invalid
            AttributeError: If operation doesn't exist on manifold
        """
        if manifold is None:
            raise ValueError("Manifold cannot be None")

        static_args = static_args or {}

        # Get the operation function
        if isinstance(operation, str):
            if not hasattr(manifold, operation):
                raise AttributeError(f"Manifold {type(manifold).__name__} has no operation '{operation}'")
            op_func = getattr(manifold, operation)
            op_name = operation
        elif callable(operation):
            op_func = operation
            op_name = getattr(operation, "__name__", "custom_function")
        else:
            raise ValueError("Operation must be string or callable")

        # Create cache key
        cache_key = self._create_cache_key(manifold, op_name, in_axes, static_args)

        # Check cache first
        if cache_key in self._compilation_cache:
            return self._compilation_cache[cache_key]

        # Create vectorized function
        def vectorized_op(*args, **kwargs):
            # Merge static args with runtime kwargs
            merged_kwargs = {**static_args, **kwargs}

            # Apply the operation with merged arguments
            if merged_kwargs:
                return op_func(*args, **merged_kwargs)
            else:
                return op_func(*args)

        # Apply vmap for vectorization
        vmapped_op = jax.vmap(vectorized_op, in_axes=in_axes) if in_axes is not None else vectorized_op

        # Apply JIT compilation
        try:
            # Determine static argument positions for JIT
            static_arg_names = list(static_args.keys()) if static_args else None

            if static_arg_names:
                jitted_op = jax.jit(vmapped_op, static_argnames=static_arg_names)
            else:
                jitted_op = jax.jit(vmapped_op)
        except Exception as e:
            # Fallback to non-JIT version
            print(f"Warning: JIT compilation failed for {op_name}: {e}")
            jitted_op = vmapped_op  # type: ignore[assignment]

        # Wrap with performance monitoring if enabled
        if self._performance_monitoring:
            monitored_op = self._wrap_with_monitoring(jitted_op, f"batch_{op_name}")
        else:
            monitored_op = jitted_op

        # Cache the result
        self._cache_compiled_function(cache_key, monitored_op)

        return monitored_op

    def dynamic_batch_compilation(self, manifold: Any, operation: str, *input_shapes: tuple[int, ...]) -> Callable:
        """Compile function dynamically based on batch shapes.

        Args:
            manifold: The manifold instance
            operation: Operation name
            input_shapes: Shapes of input arrays

        Returns:
            Compiled function optimized for given shapes
        """
        # Create shape-based cache key
        shape_key = self._create_shape_cache_key(manifold, operation, input_shapes)

        # Check if already compiled for these shapes
        if shape_key in self._compilation_cache:
            return self._compilation_cache[shape_key]

        # Determine batch axes from shapes
        input_shapes[0][0] if input_shapes else 1
        in_axes = tuple(0 for _ in input_shapes)  # Assume batch is first dimension

        # Get manifold operation
        if not hasattr(manifold, operation):
            raise AttributeError(f"Manifold {type(manifold).__name__} has no operation '{operation}'")

        op_func = getattr(manifold, operation)

        # Create optimized function for this batch size
        vmapped_op = jax.vmap(op_func, in_axes=in_axes)

        # Apply JIT compilation with shape constraints
        try:
            # Use static arguments from manifold if available
            static_args = {}
            if hasattr(manifold, "_get_static_args"):
                static_args = manifold._get_static_args()

            if static_args:
                jitted_op = jax.jit(vmapped_op, static_argnames=list(static_args.keys()))
            else:
                jitted_op = jax.jit(vmapped_op)

        except Exception as e:
            print(f"Warning: Dynamic JIT compilation failed for {operation}: {e}")
            jitted_op = vmapped_op

        # Cache and return
        self._cache_compiled_function(shape_key, jitted_op)
        return jitted_op  # type: ignore[no-any-return]

    def enable_performance_monitoring(self, enable: bool = True) -> None:
        """Enable or disable performance monitoring.

        Args:
            enable: Whether to enable monitoring
        """
        self._performance_monitoring = enable
        if enable and self._performance_monitor is None:
            self._performance_monitor = PerformanceMonitor()

    def get_performance_stats(self) -> dict[str, dict[str, Any]]:
        """Get performance statistics.

        Returns:
            Dictionary of performance statistics
        """
        return dict(self._performance_stats)

    def clear_cache(self) -> None:
        """Clear compilation cache."""
        self._compilation_cache.clear()
        if self._performance_monitor:
            self._performance_monitor.clear_stats()

    def get_cache_info(self) -> dict[str, Any]:
        """Get cache information.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_size": len(self._compilation_cache),
            "max_cache_size": self._cache_size,
            "cache_keys": list(self._compilation_cache.keys()),
        }

    def _create_cache_key(
        self, manifold: Any, operation: str, in_axes: int | tuple | None, static_args: dict[str, Any]
    ) -> str:
        """Create cache key for function signature."""
        manifold_type = type(manifold).__name__
        manifold_params = self._get_manifold_params(manifold)

        key_components = [
            manifold_type,
            str(manifold_params),
            operation,
            str(in_axes),
            str(sorted(static_args.items()) if static_args else ""),
        ]

        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _create_shape_cache_key(self, manifold: Any, operation: str, input_shapes: tuple[tuple[int, ...], ...]) -> str:
        """Create cache key based on input shapes."""
        manifold_type = type(manifold).__name__
        manifold_params = self._get_manifold_params(manifold)

        key_components = [manifold_type, str(manifold_params), operation, str(input_shapes)]

        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_manifold_params(self, manifold: Any) -> dict[str, Any]:
        """Extract relevant parameters from manifold for caching."""
        params = {}

        # Common manifold parameters
        if hasattr(manifold, "n"):
            params["n"] = manifold.n
        if hasattr(manifold, "p"):
            params["p"] = manifold.p
        if hasattr(manifold, "k"):
            params["k"] = manifold.k

        return params

    def _cache_compiled_function(self, key: str, func: Callable) -> None:
        """Cache compiled function with size management."""
        # Remove oldest entries if cache is full
        if len(self._compilation_cache) >= self._cache_size:
            # Remove first entry (simple FIFO strategy)
            oldest_key = next(iter(self._compilation_cache))
            del self._compilation_cache[oldest_key]

        self._compilation_cache[key] = func

    def _wrap_with_monitoring(self, func: Callable, operation_name: str) -> Callable:
        """Wrap function with performance monitoring."""

        @functools.wraps(func)
        def monitored_func(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Record performance stats
                if operation_name not in self._performance_stats:
                    self._performance_stats[operation_name] = {
                        "total_calls": 0,
                        "total_time": 0.0,
                        "average_time": 0.0,
                        "last_execution_time": 0.0,
                    }

                stats = self._performance_stats[operation_name]
                stats["total_calls"] += 1
                stats["total_time"] += execution_time
                stats["average_time"] = stats["total_time"] / stats["total_calls"]
                stats["last_execution_time"] = execution_time

                return result

            except Exception as e:
                execution_time = time.time() - start_time

                # Record error stats
                if operation_name not in self._performance_stats:
                    self._performance_stats[operation_name] = {"total_calls": 0, "error_count": 0}

                stats = self._performance_stats[operation_name]
                stats["total_calls"] += 1
                stats["error_count"] = stats.get("error_count", 0) + 1
                stats["last_error"] = str(e)
                stats["last_error_time"] = execution_time

                raise

        return monitored_func


# Global batch optimizer instance for convenience
_global_batch_optimizer = None


def get_batch_optimizer() -> BatchJITOptimizer:
    """Get global batch optimizer instance.

    Returns:
        Global BatchJITOptimizer instance
    """
    global _global_batch_optimizer
    if _global_batch_optimizer is None:
        _global_batch_optimizer = BatchJITOptimizer()
    return _global_batch_optimizer


def vectorize(manifold: Any, operation: str, **vmap_kwargs) -> Callable:
    """Convenience function for vectorizing manifold operations.

    Args:
        manifold: Manifold instance
        operation: Operation name
        **vmap_kwargs: Additional arguments for vmap (in_axes, etc.)

    Returns:
        Vectorized function
    """
    optimizer = get_batch_optimizer()
    return optimizer.vectorize_manifold_op(manifold, operation, **vmap_kwargs)


def batch_compile(manifold: Any, operation: str, *shapes: tuple[int, ...]) -> Callable:
    """Convenience function for dynamic batch compilation.

    Args:
        manifold: Manifold instance
        operation: Operation name
        *shapes: Input shapes

    Returns:
        Compiled batch function
    """
    optimizer = get_batch_optimizer()
    return optimizer.dynamic_batch_compilation(manifold, operation, *shapes)
