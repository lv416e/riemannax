"""RiemannAX: High-performance Riemannian optimization library with JAX JIT compilation.

RiemannAX provides implementations of Riemannian manifolds with JAX-based JIT optimization
for high-performance geometric computing and optimization.

Features:
- Comprehensive manifold implementations (Sphere, Grassmann, Stiefel, SO(n), SPD)
- JAX JIT compilation for 5x+ performance improvements
- Unified optimization interface
- Batch processing support
- GPU acceleration ready

Example:
    >>> import riemannax as rx
    >>> # Enable JIT globally (enabled by default)
    >>> rx.enable_jit()
    >>> # Create a sphere manifold
    >>> manifold = rx.manifolds.Sphere()
    >>> # Generate random point with JIT optimization
    >>> x = manifold.random_point(rx.random.key(42), 10)
"""

__version__ = "0.1.0-jit"
__author__ = "RiemannAX Contributors"
__email__ = "dev@riemannax.org"

# Core JIT system imports
import jax.numpy as jnp

# JAX utilities
import jax.random as random

from .core.batch_ops import BatchJITOptimizer
from .core.device_manager import DeviceManager
from .core.jit_manager import JITManager
from .core.performance import PerformanceMonitor

# Original imports (maintained for backward compatibility)
from .manifolds import Grassmann, SpecialOrthogonal, Sphere, Stiefel, SymmetricPositiveDefinite
from .optimizers import riemannian_adam, riemannian_gradient_descent, riemannian_momentum
from .problems import RiemannianProblem
from .solvers import OptimizeResult, minimize

# Global JIT configuration
_global_jit_enabled = True
_global_performance_monitoring = False
_global_device = "auto"


def enable_jit(cache_size: int = 10000, fallback_on_error: bool = True, debug_mode: bool = False) -> None:
    """Enable JIT compilation globally across all manifolds.

    Args:
        cache_size: Maximum number of compiled functions to cache
        fallback_on_error: Whether to fall back to non-JIT on compilation errors
        debug_mode: Enable debug mode for JIT compilation

    Example:
        >>> import riemannax as rx
        >>> rx.enable_jit(cache_size=5000, debug_mode=True)
    """
    global _global_jit_enabled
    _global_jit_enabled = True

    JITManager.configure(
        enable_jit=True, cache_size=cache_size, fallback_on_error=fallback_on_error, debug_mode=debug_mode
    )

    print(f"RiemannAX JIT optimization enabled (cache_size={cache_size})")


def disable_jit() -> None:
    """Disable JIT compilation globally across all manifolds.

    Example:
        >>> import riemannax as rx
        >>> rx.disable_jit()  # For debugging or testing
    """
    global _global_jit_enabled
    _global_jit_enabled = False

    JITManager.configure(enable_jit=False)
    print("RiemannAX JIT optimization disabled")


def get_jit_config() -> dict:
    """Get current JIT configuration.

    Returns:
        Dictionary containing JIT configuration settings
    """
    return JITManager.get_config()


def clear_jit_cache() -> None:
    """Clear all JIT compilation caches.

    This can be useful for debugging or to free up memory.

    Example:
        >>> import riemannax as rx
        >>> rx.clear_jit_cache()
    """
    JITManager.clear_cache()
    print("RiemannAX JIT cache cleared")


def enable_performance_monitoring() -> None:
    """Enable performance monitoring globally.

    Example:
        >>> import riemannax as rx
        >>> rx.enable_performance_monitoring()
        >>> # Your computation here
        >>> print(rx.get_performance_stats())
    """
    global _global_performance_monitoring
    _global_performance_monitoring = True
    PerformanceMonitor.enable()
    print("RiemannAX performance monitoring enabled")


def disable_performance_monitoring() -> None:
    """Disable performance monitoring globally."""
    global _global_performance_monitoring
    _global_performance_monitoring = False
    PerformanceMonitor.disable()
    print("RiemannAX performance monitoring disabled")


def get_performance_stats() -> dict:
    """Get current performance monitoring statistics.

    Returns:
        Dictionary containing timing and optimization statistics

    Example:
        >>> import riemannax as rx
        >>> rx.enable_performance_monitoring()
        >>> # Your computation here
        >>> stats = rx.get_performance_stats()
        >>> print(f"JIT speedup: {stats.get('avg_speedup', 1.0):.2f}x")
    """
    return PerformanceMonitor.get_stats()


def clear_performance_stats() -> None:
    """Clear all performance monitoring statistics."""
    PerformanceMonitor.clear_stats()


def set_device(device: str) -> None:
    """Set the default device for computations.

    Args:
        device: Device to use ('cpu', 'gpu', 'auto')

    Example:
        >>> import riemannax as rx
        >>> rx.set_device('gpu')  # Use GPU acceleration
    """
    global _global_device
    _global_device = device
    DeviceManager.set_default_device(device)
    print(f"RiemannAX default device set to: {device}")


def get_device_info() -> dict:
    """Get information about available devices.

    Returns:
        Dictionary containing device information
    """
    return DeviceManager.get_device_info()


def benchmark_manifold(manifold_name: str = "sphere") -> str:
    """Quick benchmark of manifold operations.

    Args:
        manifold_name: Name of manifold to benchmark

    Returns:
        Formatted benchmark report

    Example:
        >>> import riemannax as rx
        >>> report = rx.benchmark_manifold('sphere')
        >>> print(report)
    """
    try:
        from benchmarks.performance_benchmark import run_quick_benchmark

        manifold_map = {
            "sphere": "sphere_3d",
            "grassmann": "grassmann_5_3",
            "stiefel": "stiefel_5_3",
            "so": "so_3",
            "spd": "spd_3",
        }

        manifold_id = manifold_map.get(manifold_name, "sphere_3d")
        return run_quick_benchmark(manifolds=[manifold_id])
    except ImportError:
        return "Benchmark module not available. Install with performance extras."


def test_jit_compatibility() -> float:
    """Test JIT compatibility across all manifolds.

    Returns:
        Pass rate (0.0 to 1.0)

    Example:
        >>> import riemannax as rx
        >>> pass_rate = rx.test_jit_compatibility()
        >>> print(f"JIT compatibility: {pass_rate:.1%}")
    """
    try:
        from tests.test_jit_compatibility import run_compatibility_verification

        return float(run_compatibility_verification())
    except ImportError:
        print("JIT compatibility test module not available.")
        return 0.0


# Convenience namespace for manifolds (with JIT optimization)
class manifolds:  # noqa: N801
    """Namespace for manifold classes with JIT optimization."""

    Sphere = Sphere
    Grassmann = Grassmann
    Stiefel = Stiefel
    SpecialOrthogonal = SpecialOrthogonal
    SO = SpecialOrthogonal  # Alias
    SymmetricPositiveDefinite = SymmetricPositiveDefinite
    SPD = SymmetricPositiveDefinite  # Alias


# Enhanced __all__ list including JIT functionality
__all__ = [
    # Core JIT components
    "BatchJITOptimizer",
    # Original exports (backward compatibility)
    "Grassmann",
    "OptimizeResult",
    "RiemannianProblem",
    "SpecialOrthogonal",
    "Sphere",
    "Stiefel",
    "SymmetricPositiveDefinite",
    "__author__",
    "__email__",
    # Version info
    "__version__",
    "benchmark_manifold",
    "clear_jit_cache",
    "clear_performance_stats",
    "disable_jit",
    "disable_performance_monitoring",
    # JIT functionality
    "enable_jit",
    "enable_performance_monitoring",
    "get_device_info",
    "get_jit_config",
    "get_performance_stats",
    "jnp",
    # Namespaces
    "manifolds",
    "minimize",
    "random",
    "riemannian_adam",
    "riemannian_gradient_descent",
    "riemannian_momentum",
    "set_device",
    "test_jit_compatibility",
]

# Initialize JIT system on import
enable_jit()
