"""Configuration constants for RiemannAX library.

This module defines numerical constants and performance thresholds used throughout
the library to ensure consistent behavior and eliminate magic numbers.
"""


class NumericalConstants:
    """Numerical constants for stability and tolerance in mathematical operations.

    These constants are used throughout the library for numerical comparisons,
    convergence criteria, and stability checks in manifold operations.
    """

    EPSILON: float = 1e-10
    """Numerical stability threshold for small value detection."""

    RTOL: float = 1e-8
    """Relative tolerance for numerical comparisons."""

    ATOL: float = 1e-10
    """Absolute tolerance for numerical comparisons."""

    HIGH_PRECISION_EPSILON: float = 1e-12
    """High precision epsilon for critical numerical operations."""

    WEISZFELD_EPSILON: float = 1e-8
    """Epsilon for Weiszfeld algorithm to prevent division by zero."""

    MEDIUM_PRECISION_EPSILON: float = 1e-6
    """Medium precision epsilon for manifold projections."""

    SYMMETRY_TOLERANCE: float = 1e-5
    """Tolerance for checking matrix symmetry."""

    VALIDATION_TOLERANCE: float = 1e-6
    """Tolerance for validating points on manifolds."""


class PerformanceThresholds:
    """Performance thresholds for JIT compilation and optimization validation.

    These thresholds define minimum acceptable speedup ratios when JIT compilation
    is enabled, used to validate that optimization provides meaningful benefits.

    Note: Thresholds are set conservatively to account for compilation overhead
    and varying hardware performance characteristics.
    """

    MIN_CPU_SPEEDUP: float = 1.1
    """Minimum speedup ratio required for CPU JIT compilation."""

    MIN_GPU_SPEEDUP: float = 2.0
    """Minimum speedup ratio required for GPU JIT compilation."""
