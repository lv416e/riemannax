"""Tests for automatic manifold detection system."""

import pytest
import jax.numpy as jnp

from riemannax.api.detection import (
    ManifoldDetector,
    ManifoldDetectionResult,
    ManifoldType,
)
from riemannax.api.errors import ManifoldDetectionError


class TestManifoldType:
    """Test ManifoldType enum functionality."""

    def test_manifold_type_enum_values(self):
        """Test ManifoldType enum has expected values."""
        assert ManifoldType.SPHERE.value == "sphere"
        assert ManifoldType.STIEFEL.value == "stiefel"
        assert ManifoldType.SPD.value == "spd"
        assert ManifoldType.SO.value == "so"
        assert ManifoldType.UNKNOWN.value == "unknown"


class TestManifoldDetectionResult:
    """Test ManifoldDetectionResult data structure."""

    def test_detection_result_creation(self):
        """Test ManifoldDetectionResult creation and attributes."""
        alternatives = [ManifoldType.SPHERE, ManifoldType.STIEFEL]
        validation_errors = ["Test error"]

        result = ManifoldDetectionResult(
            detected_type=ManifoldType.SPHERE,
            confidence=0.95,
            constraints_satisfied=True,
            validation_errors=validation_errors,
            alternatives=alternatives
        )

        assert result.detected_type == ManifoldType.SPHERE
        assert result.confidence == 0.95
        assert result.constraints_satisfied == True
        assert result.validation_errors == validation_errors
        assert result.alternatives == alternatives

    def test_detection_result_default_values(self):
        """Test ManifoldDetectionResult with default values."""
        result = ManifoldDetectionResult(
            detected_type=ManifoldType.SPHERE,
            confidence=1.0,
            constraints_satisfied=True
        )

        assert result.validation_errors == []
        assert result.alternatives == []


class TestManifoldDetector:
    """Test ManifoldDetector class functionality."""

    def test_detect_sphere_manifold_valid(self):
        """Test detection of valid sphere manifold points."""
        # Unit vector in 2D
        x = jnp.array([1.0, 0.0])
        result = ManifoldDetector.detect_manifold(x)

        assert result.detected_type == ManifoldType.SPHERE
        assert result.confidence > 0.9
        assert result.constraints_satisfied == True
        assert len(result.validation_errors) == 0

        # Unit vector in 3D
        x = jnp.array([0.6, 0.8, 0.0])
        result = ManifoldDetector.detect_manifold(x)

        assert result.detected_type == ManifoldType.SPHERE
        assert result.confidence > 0.9
        assert result.constraints_satisfied == True

    def test_detect_sphere_manifold_invalid(self):
        """Test detection fails for invalid sphere points."""
        # Non-unit vector
        x = jnp.array([2.0, 0.0])
        result = ManifoldDetector.detect_manifold(x)

        # Should either detect as sphere with constraint violations
        # or not detect as sphere at all
        if result.detected_type == ManifoldType.SPHERE:
            assert result.constraints_satisfied == False
            assert len(result.validation_errors) > 0
        else:
            assert result.detected_type == ManifoldType.UNKNOWN

    def test_detect_stiefel_manifold_valid(self):
        """Test detection of valid Stiefel manifold points."""
        # Orthogonal matrix (square)
        theta = jnp.pi / 4
        X = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                       [jnp.sin(theta), jnp.cos(theta)]])
        result = ManifoldDetector.detect_manifold(X)

        assert result.detected_type == ManifoldType.STIEFEL
        assert result.confidence > 0.9
        assert result.constraints_satisfied == True
        assert len(result.validation_errors) == 0

        # Rectangular matrix with orthonormal columns
        X = jnp.array([[1.0, 0.0],
                       [0.0, 1.0],
                       [0.0, 0.0]])
        result = ManifoldDetector.detect_manifold(X)

        assert result.detected_type == ManifoldType.STIEFEL
        assert result.confidence > 0.9
        assert result.constraints_satisfied == True

    def test_detect_stiefel_manifold_invalid(self):
        """Test detection fails for invalid Stiefel points."""
        # Non-orthogonal matrix (this is actually SPD, so update expectation)
        X = jnp.array([[2.0, 0.0],
                       [0.0, 1.0]])
        result = ManifoldDetector.detect_manifold(X)

        # This matrix could be detected as SPD (which is correct) or other types
        # The key is that if detected as Stiefel, it should show constraint violations
        if result.detected_type == ManifoldType.STIEFEL:
            assert result.constraints_satisfied == False
            assert len(result.validation_errors) > 0
        elif result.detected_type == ManifoldType.SPD:
            # This is actually correct - the matrix is SPD
            assert result.constraints_satisfied == True
        else:
            assert result.detected_type == ManifoldType.UNKNOWN

    def test_detect_spd_manifold_valid(self):
        """Test detection of valid SPD manifold points."""
        # Simple SPD matrix
        X = jnp.array([[2.0, 1.0],
                       [1.0, 2.0]])
        result = ManifoldDetector.detect_manifold(X)

        assert result.detected_type == ManifoldType.SPD
        assert result.confidence > 0.9
        assert result.constraints_satisfied == True
        assert len(result.validation_errors) == 0

        # Identity matrix
        X = jnp.eye(3)
        result = ManifoldDetector.detect_manifold(X)

        assert result.detected_type == ManifoldType.SPD
        assert result.confidence > 0.9
        assert result.constraints_satisfied == True

    def test_detect_spd_manifold_invalid(self):
        """Test detection fails for invalid SPD points."""
        # Non-symmetric matrix
        X = jnp.array([[1.0, 2.0],
                       [0.0, 1.0]])
        result = ManifoldDetector.detect_manifold(X)

        if result.detected_type == ManifoldType.SPD:
            assert result.constraints_satisfied == False
            assert len(result.validation_errors) > 0
        else:
            assert result.detected_type == ManifoldType.UNKNOWN

    def test_detect_manifold_unknown(self):
        """Test detection returns UNKNOWN for ambiguous cases."""
        # Scalar value
        x = jnp.array(5.0)
        result = ManifoldDetector.detect_manifold(x)

        assert result.detected_type == ManifoldType.UNKNOWN
        assert result.confidence < 0.5
        assert len(result.alternatives) > 0

    def test_detect_manifold_with_tolerance(self):
        """Test detection with custom tolerance."""
        # Nearly unit vector
        x = jnp.array([1.001, 0.0])

        # Should fail with strict tolerance
        result = ManifoldDetector.detect_manifold(x, atol=1e-6)
        if result.detected_type == ManifoldType.SPHERE:
            assert result.constraints_satisfied == False

        # Should pass with loose tolerance
        result = ManifoldDetector.detect_manifold(x, atol=1e-2)
        assert result.detected_type == ManifoldType.SPHERE
        assert result.constraints_satisfied == True

    def test_validate_constraints_sphere(self):
        """Test constraint validation for sphere manifold."""
        x = jnp.array([1.0, 0.0])
        result = ManifoldDetector.validate_constraints(x, ManifoldType.SPHERE)

        assert result.constraints_satisfied == True
        assert len(result.validation_errors) == 0

        # Invalid sphere point
        x = jnp.array([2.0, 0.0])
        result = ManifoldDetector.validate_constraints(x, ManifoldType.SPHERE)

        assert result.constraints_satisfied == False
        assert len(result.validation_errors) > 0

    def test_validate_constraints_unsupported_manifold(self):
        """Test validation with unsupported manifold type."""
        x = jnp.array([1.0, 0.0])

        with pytest.raises(ManifoldDetectionError) as exc_info:
            ManifoldDetector.validate_constraints(x, ManifoldType.UNKNOWN)

        assert "Unsupported manifold type" in str(exc_info.value)

    def test_suggest_manifold_alternatives(self):
        """Test manifold suggestion functionality."""
        # Vector that could be sphere or part of Stiefel
        x = jnp.array([0.6, 0.8])
        suggestions = ManifoldDetector.suggest_manifold(x)

        assert len(suggestions) > 0
        assert any(s.manifold_type == ManifoldType.SPHERE for s in suggestions)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Zero vector
        x = jnp.array([0.0, 0.0])
        result = ManifoldDetector.detect_manifold(x)
        assert result.detected_type == ManifoldType.UNKNOWN

        # Empty array
        x = jnp.array([])
        result = ManifoldDetector.detect_manifold(x)
        assert result.detected_type == ManifoldType.UNKNOWN

        # Single element array
        x = jnp.array([1.0])
        result = ManifoldDetector.detect_manifold(x)
        # Could be sphere in 1D, but implementation dependent
        assert result.detected_type in [ManifoldType.SPHERE, ManifoldType.UNKNOWN]
