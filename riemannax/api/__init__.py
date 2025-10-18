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
from .problems import (
    ManifoldConstrainedParameter,
    ManifoldPCA,
    MatrixCompletion,
    RobustCovarianceEstimation,
)
from .results import ConvergenceStatus, OptimizationResult
from .validation import (
    ValidationResult,
    validate_orthogonal_constraint,
    validate_spd_constraint,
    validate_sphere_constraint,
)

__all__ = [
    "ConstraintViolationError",
    "ConvergenceStatus",
    "ManifoldConstrainedParameter",
    "ManifoldDetectionError",
    "ManifoldDetectionResult",
    "ManifoldDetector",
    "ManifoldPCA",
    "ManifoldType",
    "MatrixCompletion",
    "OptimizationResult",
    "ParameterValidationError",
    "RiemannAXAPIError",
    "RiemannianAdam",
    "RiemannianEstimator",
    "RiemannianSGD",
    "RobustCovarianceEstimation",
    "ValidationResult",
    "validate_orthogonal_constraint",
    "validate_spd_constraint",
    "validate_sphere_constraint",
]
