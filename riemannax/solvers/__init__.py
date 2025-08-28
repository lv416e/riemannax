"""Solver implementations for Riemannian optimization problems.

This module provides solvers for minimizing functions on Riemannian manifolds,
with various optimization algorithms and termination conditions.

Components:
- Native RiemannAX solvers (minimize, OptimizeResult)
- Optimistix integration for advanced optimization algorithms
"""

from .minimize import OptimizeResult, minimize

# Optimistix integration (optional dependency)
try:
    from .optimistix_adapter import (
        ManifoldMinimizer,
        RiemannianProblemAdapter,
        euclidean_to_riemannian_gradient,
        least_squares_on_manifold,
        minimize_on_manifold,
    )

    __all__ = [
        "ManifoldMinimizer",
        "OptimizeResult",
        "RiemannianProblemAdapter",
        "euclidean_to_riemannian_gradient",
        "least_squares_on_manifold",
        "minimize",
        "minimize_on_manifold",
    ]

except ImportError:
    # Optimistix not available, only export core solvers
    __all__ = ["OptimizeResult", "minimize"]
