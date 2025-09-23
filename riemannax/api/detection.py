"""Automatic manifold detection system for RiemannAX high-level APIs."""

import dataclasses
from enum import Enum

import jax.numpy as jnp
from jaxtyping import Array

from .errors import ManifoldDetectionError
from .validation import (
    ValidationResult,
    validate_orthogonal_constraint,
    validate_spd_constraint,
    validate_sphere_constraint,
)


class ManifoldType(Enum):
    """Enumeration of supported manifold types for automatic detection."""

    SPHERE = "sphere"
    STIEFEL = "stiefel"
    SPD = "spd"
    SO = "so"
    UNKNOWN = "unknown"


@dataclasses.dataclass
class ManifoldCandidate:
    """Candidate manifold type with confidence score."""

    manifold_type: ManifoldType
    confidence: float
    reason: str


@dataclasses.dataclass
class ManifoldDetectionResult:
    """Result of manifold detection with confidence and validation information.

    Attributes:
        detected_type: The detected manifold type.
        confidence: Confidence score in the detection (0.0 to 1.0).
        constraints_satisfied: Whether the point satisfies manifold constraints.
        validation_errors: List of constraint validation error messages.
        alternatives: List of alternative manifold type suggestions.
    """

    detected_type: ManifoldType
    confidence: float
    constraints_satisfied: bool
    validation_errors: list[str] = dataclasses.field(default_factory=list)
    alternatives: list[ManifoldType] = dataclasses.field(default_factory=list)


class ManifoldDetector:
    """Static class for automatic manifold detection and validation."""

    @staticmethod
    def detect_manifold(x: Array, atol: float = 1e-6) -> ManifoldDetectionResult:
        """Automatically detect the appropriate manifold type for given data.

        Args:
            x: Array to analyze for manifold type detection.
            atol: Absolute tolerance for constraint validation.

        Returns:
            ManifoldDetectionResult with detected type and validation info.
        """
        # Handle edge cases
        if x.size == 0:
            return ManifoldDetectionResult(
                detected_type=ManifoldType.UNKNOWN,
                confidence=0.0,
                constraints_satisfied=False,
                validation_errors=["Empty array cannot be classified"],
                alternatives=[],
            )

        if x.ndim == 0:  # Scalar
            return ManifoldDetectionResult(
                detected_type=ManifoldType.UNKNOWN,
                confidence=0.0,
                constraints_satisfied=False,
                validation_errors=["Scalar values cannot be classified as manifold points"],
                alternatives=[ManifoldType.SPHERE],  # Could be embedded in higher dimensions
            )

        # Try to detect manifold type based on array structure
        candidates = ManifoldDetector._analyze_structure(x, atol)

        if not candidates:
            return ManifoldDetectionResult(
                detected_type=ManifoldType.UNKNOWN,
                confidence=0.0,
                constraints_satisfied=False,
                validation_errors=["Could not determine manifold type"],
                alternatives=[],
            )

        # Select the highest confidence candidate
        best_candidate = max(candidates, key=lambda c: c.confidence)

        # Validate constraints for the detected type
        validation_result = ManifoldDetector.validate_constraints(x, best_candidate.manifold_type, atol)

        return ManifoldDetectionResult(
            detected_type=best_candidate.manifold_type,
            confidence=best_candidate.confidence,
            constraints_satisfied=validation_result.constraints_satisfied,
            validation_errors=validation_result.validation_errors,
            alternatives=[c.manifold_type for c in candidates[1:]],
        )

    @staticmethod
    def _analyze_structure(x: Array, atol: float) -> list[ManifoldCandidate]:
        """Analyze array structure to suggest manifold types."""
        candidates = []

        # Check for sphere manifold (vector)
        if x.ndim == 1 and x.size > 0:
            norm = float(jnp.linalg.norm(x))
            if norm > atol:  # Non-zero vector
                confidence = 1.0 if abs(norm - 1.0) < atol else max(0.1, 1.0 - abs(norm - 1.0))
                candidates.append(
                    ManifoldCandidate(
                        manifold_type=ManifoldType.SPHERE,
                        confidence=confidence,
                        reason=f"1D array with norm {norm:.6f}",
                    )
                )

        # Check for matrix manifolds
        elif x.ndim == 2:
            m, n = x.shape

            # Check for SPD manifold (symmetric positive definite)
            if m == n:  # Square matrix
                spd_confidence = ManifoldDetector._assess_spd_likelihood(x, atol)
                if spd_confidence > 0.1:
                    candidates.append(
                        ManifoldCandidate(
                            manifold_type=ManifoldType.SPD,
                            confidence=spd_confidence,
                            reason=f"Square matrix {m}x{n} with SPD characteristics",
                        )
                    )

            # Check for Stiefel manifold (orthogonal matrices)
            stiefel_confidence = ManifoldDetector._assess_stiefel_likelihood(x, atol)
            if stiefel_confidence > 0.1:
                candidates.append(
                    ManifoldCandidate(
                        manifold_type=ManifoldType.STIEFEL,
                        confidence=stiefel_confidence,
                        reason=f"Matrix {m}x{n} with orthogonal characteristics",
                    )
                )

            # Check for SO(n) (special orthogonal: orthogonal with det +1)
            if m == n and stiefel_confidence > 0.1:
                try:
                    det = float(jnp.linalg.det(x))
                    det_closeness = max(0.0, 1.0 - abs(det - 1.0) / atol)
                    so_confidence = min(1.0, stiefel_confidence * det_closeness)
                    if so_confidence > 0.1:
                        candidates.append(
                            ManifoldCandidate(
                                manifold_type=ManifoldType.SO,
                                confidence=so_confidence,
                                reason=f"Square matrix with orthogonal characteristics and det={det:.6f}",
                            )
                        )
                except Exception:
                    # Skip SO detection if determinant computation fails
                    pass

        return sorted(candidates, key=lambda c: c.confidence, reverse=True)

    @staticmethod
    def _assess_spd_likelihood(x: Array, atol: float) -> float:
        """Assess likelihood that matrix is SPD."""
        m, n = x.shape
        if m != n:
            return 0.0

        # Check symmetry
        symmetry_error = float(jnp.max(jnp.abs(x - x.T)))
        symmetry_score = max(0.0, 1.0 - symmetry_error / (10 * atol))  # More lenient

        # Check positive definiteness
        try:
            if symmetry_error < 10 * atol:
                # Use more stable eigvalsh for nearly symmetric matrices
                X_sym = 0.5 * (x + x.T)
                eigenvals = jnp.linalg.eigvalsh(X_sym)
                min_eigenval = float(jnp.min(eigenvals))
            else:
                # Fallback to general eigvals for non-symmetric matrices
                eigenvals = jnp.linalg.eigvals(x)
                min_eigenval = float(jnp.min(jnp.real(eigenvals)))
            pd_score = 1.0 if min_eigenval > atol else max(0.0, min_eigenval / atol)
        except Exception:
            pd_score = 0.0

        # Combine scores (both need to be reasonable)
        combined_score = symmetry_score * pd_score
        return min(1.0, max(0.0, combined_score))

    @staticmethod
    def _assess_stiefel_likelihood(x: Array, atol: float) -> float:
        """Assess likelihood that matrix has orthogonal columns."""
        _m, n = x.shape

        # Compute X^T X
        XTX = x.T @ x
        I = jnp.eye(n)

        # Check how close X^T X is to identity
        orthogonality_error = float(jnp.max(jnp.abs(XTX - I)))
        return max(0.0, 1.0 - orthogonality_error / atol)

    @staticmethod
    def validate_constraints(x: Array, manifold_type: ManifoldType, atol: float = 1e-6) -> ManifoldDetectionResult:
        """Validate that array satisfies constraints for given manifold type.

        Args:
            x: Array to validate.
            manifold_type: Manifold type to validate against.
            atol: Absolute tolerance for validation.

        Returns:
            ManifoldDetectionResult with constraint validation information.

        Raises:
            ManifoldDetectionError: If manifold type is not supported for validation.
        """
        if manifold_type == ManifoldType.SPHERE:
            validation_result = validate_sphere_constraint(x, atol)
        elif manifold_type == ManifoldType.STIEFEL:
            validation_result = validate_orthogonal_constraint(x, atol)
        elif manifold_type == ManifoldType.SPD:
            validation_result = validate_spd_constraint(x, atol)
        elif manifold_type == ManifoldType.SO:
            # For SO(n), it's the same as Stiefel but with square matrices and det(X) = 1
            if x.ndim != 2 or x.shape[0] != x.shape[1]:
                validation_result = ValidationResult(
                    is_valid=False,
                    violations=["SO(n) manifold requires square matrices"],
                    suggestions=["Use a square orthogonal matrix"],
                )
            else:
                validation_result = validate_orthogonal_constraint(x, atol)
                if validation_result.is_valid:
                    det = jnp.linalg.det(x)
                    if not jnp.allclose(det, 1.0, atol=atol):
                        validation_result = ValidationResult(
                            is_valid=False,
                            violations=[
                                *validation_result.violations,
                                f"Matrix determinant must be +1 for SO(n), got {float(det):.6f}",
                            ],
                            suggestions=[*validation_result.suggestions, "Ensure the matrix has a determinant of +1."],
                        )
        else:
            raise ManifoldDetectionError(f"Unsupported manifold type for validation: {manifold_type}")

        return ManifoldDetectionResult(
            detected_type=manifold_type,
            confidence=1.0 if validation_result.is_valid else 0.0,
            constraints_satisfied=validation_result.is_valid,
            validation_errors=validation_result.violations,
            alternatives=[],
        )

    @staticmethod
    def suggest_manifold(x: Array, atol: float = 1e-6) -> list[ManifoldCandidate]:
        """Suggest possible manifold types for given array.

        Args:
            x: Array to analyze.
            atol: Absolute tolerance for analysis.

        Returns:
            List of ManifoldCandidate objects sorted by confidence.
        """
        return ManifoldDetector._analyze_structure(x, atol)
