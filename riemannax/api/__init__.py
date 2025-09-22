"""High-level APIs for RiemannAX.

This module provides scikit-learn compatible interfaces, automatic manifold detection,
and practical problem templates for Riemannian optimization.
"""

from .detection import ManifoldDetectionResult, ManifoldDetector, ManifoldType
from .errors import (
    ConstraintViolationError,
    ManifoldDetectionError,
    ParameterValidationError,
    RiemannAXAPIError,
)
from .estimators import RiemannianAdam, RiemannianEstimator, RiemannianSGD
from .results import ConvergenceStatus, OptimizationResult
from .validation import ValidationResult, validate_sphere_constraint

__all__ = [
    "ConstraintViolationError",
    "ConvergenceStatus",
    "ManifoldDetectionError",
    "ManifoldDetectionResult",
    "ManifoldDetector",
    "ManifoldType",
    "OptimizationResult",
    "ParameterValidationError",
    "RiemannAXAPIError",
    "RiemannianAdam",
    "RiemannianEstimator",
    "RiemannianSGD",
    "ValidationResult",
    "validate_sphere_constraint",
]
