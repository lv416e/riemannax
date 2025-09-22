"""High-level scikit-learn compatible APIs for RiemannAX.

This module provides user-friendly, scikit-learn compatible interfaces for
Riemannian optimization, making it easy to integrate manifold optimization
into standard machine learning workflows.
"""

from .base import RiemannianEstimator
from .exceptions import (
    ConstraintViolationError,
    InvalidManifoldError,
    ParameterValidationError,
    RiemannianOptimizationError,
)

__all__ = [
    "ConstraintViolationError",
    "InvalidManifoldError",
    "ParameterValidationError",
    "RiemannianEstimator",
    "RiemannianOptimizationError",
]
