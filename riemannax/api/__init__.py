"""High-level API for RiemannAX.

This module provides scikit-learn compatible interfaces for Riemannian optimization,
automatic manifold detection, and practical problem templates.
"""

from .detection import ManifoldDetector, minimize
from .estimators import RiemannianAdam, RiemannianSGD

__all__ = [
    "ManifoldDetector",
    "RiemannianAdam",
    "RiemannianSGD",
    "minimize",
]
