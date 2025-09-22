"""High-level APIs for RiemannAX.

This module provides scikit-learn compatible interfaces, automatic manifold detection,
and practical problem templates for Riemannian optimization.
"""

from .results import ConvergenceStatus, OptimizationResult

__all__ = [
    "ConvergenceStatus",
    "OptimizationResult",
]
