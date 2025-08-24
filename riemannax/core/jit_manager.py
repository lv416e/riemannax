"""JIT Management System for RiemannAX optimization."""

from collections.abc import Callable
from functools import wraps
from typing import Any, ClassVar

import jax


class JITManager:
    """Central management system for JIT optimization."""

    # Manage configuration and cache through class variables
    _config: ClassVar[dict[str, Any]] = {
        "enable_jit": True,
        "cache_size": 10000,
        "fallback_on_error": True,
        "debug_mode": False,
    }

    _cache: ClassVar[dict[str, Any]] = {}

    @staticmethod
    def _get_cache_key(func_name: str, **kwargs: Any) -> str:
        """Generate unique cache key for function and JIT parameters.

        Args:
            func_name: Name of the function being cached
            **kwargs: JIT parameters (static_argnums, device, etc.)

        Returns:
            Unique cache key string
        """
        # Sort kwargs to ensure consistent key generation
        sorted_kwargs = sorted(kwargs.items())
        kwargs_hash = hash(frozenset(sorted_kwargs))
        return f"{func_name}_{kwargs_hash}"

    @classmethod
    def _enforce_cache_limit(cls) -> None:
        """Enforce cache size limit by removing oldest entries."""
        cache_limit = cls._config["cache_size"]
        if len(cls._cache) >= cache_limit:
            # Remove oldest entries (simple FIFO strategy)
            keys_to_remove = list(cls._cache.keys())[: -cache_limit + 1]
            for key in keys_to_remove:
                cls._cache.pop(key, None)

    @classmethod
    def configure(cls, **kwargs: Any) -> None:
        """Update JIT configuration.

        Args:
            **kwargs: Configuration parameters
                - enable_jit: Enable/disable JIT optimization
                - cache_size: Cache size limit
                - fallback_on_error: Fallback on error
                - debug_mode: Debug mode
        """
        cls._config.update(kwargs)

    @classmethod
    def jit_decorator(
        cls, func: Callable[..., Any], static_argnums: tuple[int, ...] | None = None, device: str | None = None
    ) -> Callable[..., Any]:
        """Unified JIT decorator with caching support.

        Args:
            func: Function to be JIT-optimized
            static_argnums: Indices of static arguments
            device: Execution device specification (cpu, gpu, tpu)

        Returns:
            JIT-optimized function (cached if previously compiled)
        """
        if not cls._config["enable_jit"]:
            return func

        # Prepare JIT configuration
        jit_kwargs: dict[str, Any] = {}
        if static_argnums is not None:
            jit_kwargs["static_argnums"] = static_argnums

        # Prepare cache key (include device for cache differentiation)
        cache_kwargs = dict(jit_kwargs)
        if device is not None:
            cache_kwargs["device"] = device

        # Generate cache key
        cache_key = cls._get_cache_key(func.__name__, **cache_kwargs)

        # Check if function is already cached
        if cache_key in cls._cache:
            cached_func: Callable[..., Any] = cls._cache[cache_key]
            return cached_func

        # Enforce cache size limit before adding new entry
        cls._enforce_cache_limit()

        # Create JIT-optimized function
        jit_func = jax.jit(func, **jit_kwargs)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return jit_func(*args, **kwargs)
            except Exception as e:
                if cls._config["fallback_on_error"]:
                    # Fallback execution
                    return func(*args, **kwargs)
                else:
                    raise e

        # Store in cache before returning
        cls._cache[cache_key] = wrapper
        return wrapper

    @classmethod
    def clear_cache(cls) -> None:
        """Clear JIT cache."""
        cls._cache.clear()

    @classmethod
    def get_config(cls) -> dict[str, Any]:
        """Get current configuration.

        Returns:
            Current configuration dictionary
        """
        return cls._config.copy()

    @classmethod
    def reset_config(cls) -> None:
        """Reset configuration to default."""
        cls._config = {"enable_jit": True, "cache_size": 10000, "fallback_on_error": True, "debug_mode": False}
