"""Tests for hyperbolic-specific data models."""

import jax.numpy as jnp
import pytest

from riemannax.manifolds.data_models import (
    DataModelError,
    HyperbolicPoint,
    ManifoldParameters,
    SE3Transform,
)


class TestHyperbolicPoint:
    """Test the HyperbolicPoint dataclass."""

    def test_poincare_ball_point_creation(self):
        """Test HyperbolicPoint creation for Poincare ball model."""
        coords = jnp.array([0.1, 0.2, 0.3])
        point = HyperbolicPoint(coords, model="poincare", validate=True)

        assert point.model == "poincare"
        assert jnp.allclose(point.coordinates, coords)
        assert point.curvature == -1.0
        assert point.is_valid

    def test_lorentz_point_creation(self):
        """Test HyperbolicPoint creation for Lorentz model."""
        # Construct coordinates satisfying x₀² - Σxᵢ² = 1 (for curvature = -1)
        spatial = jnp.array([0.5, 0.8, 0.2])
        x0 = jnp.sqrt(1.0 + jnp.sum(spatial**2))  # x₀ = √(1 + Σxᵢ²)
        coords = jnp.concatenate([jnp.array([x0]), spatial])
        point = HyperbolicPoint(coords, model="lorentz", validate=True)

        assert point.model == "lorentz"
        assert jnp.allclose(point.coordinates, coords)
        assert point.curvature == -1.0
        assert point.is_valid

    def test_custom_curvature(self):
        """Test HyperbolicPoint with custom curvature parameter."""
        coords = jnp.array([0.1, 0.2])
        point = HyperbolicPoint(coords, model="poincare", curvature=-2.0, validate=True)

        assert point.curvature == -2.0

    def test_poincare_ball_constraint_validation(self):
        """Test Poincare ball constraint (norm < 1) validation."""
        # Valid point inside unit ball
        valid_coords = jnp.array([0.5, 0.3])
        point = HyperbolicPoint(valid_coords, model="poincare", validate=True)
        assert point.is_valid

        # Invalid point outside unit ball
        invalid_coords = jnp.array([1.5, 0.3])
        with pytest.raises(DataModelError):
            HyperbolicPoint(invalid_coords, model="poincare", validate=True)

    def test_lorentz_constraint_validation(self):
        """Test Lorentz model constraint validation."""
        # Valid point satisfying Lorentz constraint
        spatial = jnp.array([0.8, 0.2, 0.5])
        x0 = jnp.sqrt(1.0 + jnp.sum(spatial**2))
        valid_coords = jnp.concatenate([jnp.array([x0]), spatial])
        point = HyperbolicPoint(valid_coords, model="lorentz", validate=True)
        assert point.is_valid

        # Invalid point not satisfying constraint
        invalid_coords = jnp.array([0.5, 0.8, 0.6, 0.4])  # x[0] too small
        with pytest.raises(DataModelError):
            HyperbolicPoint(invalid_coords, model="lorentz", validate=True)

    def test_norm_calculation(self):
        """Test norm calculation for hyperbolic points."""
        coords = jnp.array([0.3, 0.4])
        point = HyperbolicPoint(coords, model="poincare", validate=True)

        expected_norm = jnp.linalg.norm(coords)
        assert jnp.allclose(point.norm(), expected_norm)

    def test_distance_to_origin(self):
        """Test distance calculation to origin in hyperbolic space."""
        coords = jnp.array([0.3, 0.4])
        point = HyperbolicPoint(coords, model="poincare", validate=True)

        # Poincare ball distance to origin formula
        norm_sq = jnp.sum(coords**2)
        expected_distance = jnp.arctanh(jnp.sqrt(norm_sq))
        assert jnp.allclose(point.distance_to_origin(), expected_distance)

    def test_no_validation_mode(self):
        """Test creation without validation."""
        invalid_coords = jnp.array([1.5, 0.3])  # Outside unit ball
        point = HyperbolicPoint(invalid_coords, model="poincare", validate=False)

        assert not point.is_valid
        assert jnp.allclose(point.coordinates, invalid_coords)

    def test_unknown_model_error(self):
        """Test error for unknown hyperbolic model."""
        coords = jnp.array([0.1, 0.2])
        with pytest.raises(DataModelError):
            HyperbolicPoint(coords, model="unknown_model", validate=True)


class TestSE3Transform:
    """Test the SE3Transform dataclass."""

    def test_se3_identity_creation(self):
        """Test SE3Transform creation with identity transformation."""
        transform = SE3Transform.identity()

        assert jnp.allclose(transform.rotation, jnp.eye(3))
        assert jnp.allclose(transform.translation, jnp.zeros(3))
        assert transform.is_valid

    def test_se3_creation_with_rotation_translation(self):
        """Test SE3Transform creation with rotation and translation."""
        rotation = jnp.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # 90° around x-axis
        translation = jnp.array([1.0, 2.0, 3.0])

        transform = SE3Transform(rotation, translation, validate=True)

        assert jnp.allclose(transform.rotation, rotation)
        assert jnp.allclose(transform.translation, translation)
        assert transform.is_valid

    def test_rotation_matrix_validation(self):
        """Test rotation matrix orthogonality validation."""
        # Valid orthogonal matrix
        valid_rotation = jnp.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        translation = jnp.zeros(3)

        transform = SE3Transform(valid_rotation, translation, validate=True)
        assert transform.is_valid

        # Invalid non-orthogonal matrix
        invalid_rotation = jnp.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        with pytest.raises(DataModelError):
            SE3Transform(invalid_rotation, translation, validate=True)

    def test_determinant_validation(self):
        """Test rotation matrix determinant validation (should be +1)."""
        # Valid rotation matrix (det = +1)
        valid_rotation = jnp.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        translation = jnp.zeros(3)

        transform = SE3Transform(valid_rotation, translation, validate=True)
        assert transform.is_valid

        # Invalid reflection matrix (det = -1)
        invalid_rotation = jnp.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        with pytest.raises(DataModelError):
            SE3Transform(invalid_rotation, translation, validate=True)

    def test_homogeneous_matrix_property(self):
        """Test homogeneous matrix property calculation."""
        rotation = jnp.eye(3)
        translation = jnp.array([1.0, 2.0, 3.0])

        transform = SE3Transform(rotation, translation)
        homogeneous = transform.homogeneous_matrix()

        expected = jnp.array([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])

        assert jnp.allclose(homogeneous, expected)

    def test_inverse_transform(self):
        """Test inverse transformation calculation."""
        rotation = jnp.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        translation = jnp.array([1.0, 2.0, 3.0])

        transform = SE3Transform(rotation, translation)
        inverse = transform.inverse()

        # Check that transform * inverse = identity
        composed = transform.compose(inverse)
        assert jnp.allclose(composed.rotation, jnp.eye(3), atol=1e-6)
        assert jnp.allclose(composed.translation, jnp.zeros(3), atol=1e-6)

    def test_transform_composition(self):
        """Test composition of two SE(3) transforms."""
        # First transform: rotation around z-axis
        r1 = jnp.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        t1 = jnp.array([1.0, 0.0, 0.0])
        transform1 = SE3Transform(r1, t1)

        # Second transform: translation
        r2 = jnp.eye(3)
        t2 = jnp.array([0.0, 1.0, 0.0])
        transform2 = SE3Transform(r2, t2)

        composed = transform1.compose(transform2)

        # Verify composition is correct
        expected_rotation = jnp.dot(r1, r2)
        expected_translation = jnp.dot(r1, t2) + t1

        assert jnp.allclose(composed.rotation, expected_rotation)
        assert jnp.allclose(composed.translation, expected_translation)

    def test_no_validation_mode(self):
        """Test creation without validation."""
        invalid_rotation = jnp.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]])  # Not orthogonal
        translation = jnp.zeros(3)

        transform = SE3Transform(invalid_rotation, translation, validate=False)
        assert not transform.is_valid


class TestManifoldParameters:
    """Test the ManifoldParameters dataclass."""

    def test_default_parameters(self):
        """Test ManifoldParameters with default values."""
        params = ManifoldParameters()

        assert params.tolerance == 1e-6
        assert params.max_iterations == 100
        assert params.step_size == 0.01
        assert params.use_retraction is True
        assert params.manifold_type == "riemannian"

    def test_custom_parameters(self):
        """Test ManifoldParameters with custom values."""
        params = ManifoldParameters(
            tolerance=1e-8, max_iterations=200, step_size=0.005, use_retraction=False, manifold_type="hyperbolic"
        )

        assert params.tolerance == 1e-8
        assert params.max_iterations == 200
        assert params.step_size == 0.005
        assert params.use_retraction is False
        assert params.manifold_type == "hyperbolic"

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Valid parameters
        params = ManifoldParameters(tolerance=1e-8, max_iterations=50)
        assert params.is_valid()

        # Invalid tolerance (negative)
        with pytest.raises(DataModelError):
            ManifoldParameters(tolerance=-1e-6)

        # Invalid max_iterations (zero)
        with pytest.raises(DataModelError):
            ManifoldParameters(max_iterations=0)

        # Invalid step_size (negative)
        with pytest.raises(DataModelError):
            ManifoldParameters(step_size=-0.01)

    def test_convergence_check(self):
        """Test convergence criteria checking."""
        params = ManifoldParameters(tolerance=1e-4, max_iterations=100)

        # Test convergence based on tolerance
        assert params.check_convergence(error=1e-5, iteration=10) is True
        assert params.check_convergence(error=1e-3, iteration=10) is False

        # Test convergence based on max iterations
        assert params.check_convergence(error=1e-3, iteration=100) is True
        assert params.check_convergence(error=1e-3, iteration=99) is False

    def test_parameter_summary(self):
        """Test parameter summary generation."""
        params = ManifoldParameters(tolerance=1e-8, max_iterations=200, manifold_type="hyperbolic")

        summary = params.summary()

        assert "tolerance" in summary
        assert "max_iterations" in summary
        assert "manifold_type" in summary
        assert summary["tolerance"] == 1e-8
        assert summary["max_iterations"] == 200
        assert summary["manifold_type"] == "hyperbolic"

    def test_copy_with_modifications(self):
        """Test copying parameters with modifications."""
        original = ManifoldParameters(tolerance=1e-6, max_iterations=100)

        # Copy with modifications
        modified = original.copy_with(tolerance=1e-8, step_size=0.005)

        # Original unchanged
        assert original.tolerance == 1e-6
        assert original.step_size == 0.01

        # Modified copy has new values
        assert modified.tolerance == 1e-8
        assert modified.step_size == 0.005
        assert modified.max_iterations == 100  # Unchanged


class TestDataModelError:
    """Test DataModelError exception."""

    def test_error_creation(self):
        """Test that DataModelError can be created and raised."""
        with pytest.raises(DataModelError) as exc_info:
            raise DataModelError("Test data model error")

        assert "Test data model error" in str(exc_info.value)
