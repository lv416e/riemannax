"""Tests for manifold factory functions."""

import jax
import jax.numpy as jnp
import pytest

from riemannax.manifolds import (
    create_lorentz,
    create_poincare_ball,
    create_poincare_ball_for_embeddings,
    create_se3,
    create_se3_for_robotics,
)
from riemannax.manifolds.lorentz import Lorentz
from riemannax.manifolds.poincare_ball import PoincareBall
from riemannax.manifolds.se3 import SE3


class TestPoincareBallFactory:
    """Test PoincareBall factory functions."""

    def test_create_poincare_ball_basic(self):
        """Test basic PoincareBall factory creation."""
        manifold = create_poincare_ball(dimension=3, curvature=-1.5)

        assert isinstance(manifold, PoincareBall)
        assert manifold.dimension == 3
        assert manifold.ambient_dimension == 3
        assert manifold.curvature == -1.5
        assert manifold.tolerance == 1e-6  # Default value

    def test_create_poincare_ball_custom_tolerance(self):
        """Test PoincareBall factory with custom tolerance."""
        manifold = create_poincare_ball(
            dimension=2,
            curvature=-0.5,
            tolerance=1e-10
        )

        assert isinstance(manifold, PoincareBall)
        assert manifold.tolerance == 1e-10

    def test_create_poincare_ball_parameter_validation(self):
        """Test PoincareBall factory parameter validation."""

        # Test invalid dimension
        with pytest.raises(ValueError, match="Dimension must be positive"):
            create_poincare_ball(dimension=0)

        with pytest.raises(ValueError, match="Dimension must be positive"):
            create_poincare_ball(dimension=-1)

        # Test invalid curvature (must be negative)
        with pytest.raises(ValueError, match="Curvature must be negative"):
            create_poincare_ball(dimension=2, curvature=1.0)

        with pytest.raises(ValueError, match="Curvature must be negative"):
            create_poincare_ball(dimension=2, curvature=0.0)

        # Test invalid tolerance
        with pytest.raises(ValueError, match="Tolerance must be positive"):
            create_poincare_ball(dimension=2, tolerance=0.0)

        with pytest.raises(ValueError, match="Tolerance must be positive"):
            create_poincare_ball(dimension=2, tolerance=-1e-8)

    def test_create_poincare_ball_functionality(self):
        """Test that created PoincareBall functions correctly."""
        manifold = create_poincare_ball(dimension=2, curvature=-2.0)
        key = jax.random.PRNGKey(42)

        # Test random point generation
        point = manifold.random_point(key)
        assert point.shape == (2,)
        assert manifold.validate_point(point)

        # Test that point is within unit ball
        norm = jnp.linalg.norm(point)
        assert norm < 1.0

    def test_create_poincare_ball_for_embeddings(self):
        """Test PoincareBall embeddings preset."""
        manifold = create_poincare_ball_for_embeddings(dimension=10)

        assert isinstance(manifold, PoincareBall)
        assert manifold.dimension == 10
        assert manifold.curvature == -1.0  # Unit curvature for embeddings
        assert manifold.tolerance == 1e-6


class TestLorentzFactory:
    """Test Lorentz factory functions."""

    def test_create_lorentz_basic(self):
        """Test basic Lorentz factory creation."""
        manifold = create_lorentz(dimension=3)

        assert isinstance(manifold, Lorentz)
        assert manifold.dimension == 3
        assert manifold.ambient_dimension == 4  # n+1 dimensional ambient space
        assert manifold.atol == 1e-8  # Default value

    def test_create_lorentz_custom_tolerance(self):
        """Test Lorentz factory with custom tolerance."""
        manifold = create_lorentz(dimension=5, atol=1e-10)

        assert isinstance(manifold, Lorentz)
        assert manifold.atol == 1e-10

    def test_create_lorentz_parameter_validation(self):
        """Test Lorentz factory parameter validation."""

        # Test invalid dimension
        with pytest.raises(ValueError, match="Dimension must be positive"):
            create_lorentz(dimension=0)

        with pytest.raises(ValueError, match="Dimension must be positive"):
            create_lorentz(dimension=-2)

        # Test invalid tolerance
        with pytest.raises(ValueError, match="Absolute tolerance must be positive"):
            create_lorentz(dimension=3, atol=0.0)

        with pytest.raises(ValueError, match="Absolute tolerance must be positive"):
            create_lorentz(dimension=3, atol=-1e-6)

    def test_create_lorentz_functionality(self):
        """Test that created Lorentz functions correctly."""
        manifold = create_lorentz(dimension=2)
        key = jax.random.PRNGKey(123)

        # Test random point generation
        point = manifold.random_point(key)
        assert point.shape == (3,)  # Ambient dimension is 3 for 2D hyperbolic
        assert manifold.validate_point(point)

        # Test hyperboloid constraint: x_0^2 - x_1^2 - x_2^2 = 1
        minkowski_inner = manifold._minkowski_inner(point, point)
        assert jnp.allclose(minkowski_inner, 1.0, atol=manifold.atol)


class TestSE3Factory:
    """Test SE(3) factory functions."""

    def test_create_se3_basic(self):
        """Test basic SE(3) factory creation."""
        manifold = create_se3()

        assert isinstance(manifold, SE3)
        assert manifold.dimension == 6  # SE(3) has 6 DOF
        assert manifold.ambient_dimension == 7  # (qw, qx, qy, qz, tx, ty, tz)
        assert manifold.atol == 1e-8  # Default value

    def test_create_se3_custom_tolerance(self):
        """Test SE(3) factory with custom tolerance."""
        manifold = create_se3(atol=1e-12)

        assert isinstance(manifold, SE3)
        assert manifold.atol == 1e-12

    def test_create_se3_parameter_validation(self):
        """Test SE(3) factory parameter validation."""

        # Test invalid tolerance
        with pytest.raises(ValueError, match="Absolute tolerance must be positive"):
            create_se3(atol=0.0)

        with pytest.raises(ValueError, match="Absolute tolerance must be positive"):
            create_se3(atol=-1e-8)

    def test_create_se3_functionality(self):
        """Test that created SE(3) functions correctly."""
        manifold = create_se3()
        key = jax.random.PRNGKey(456)

        # Test random point generation
        transform = manifold.random_point(key)
        assert transform.shape == (7,)
        assert manifold.validate_point(transform)

        # Test quaternion part is normalized
        quaternion = transform[:4]
        q_norm = jnp.linalg.norm(quaternion)
        assert jnp.allclose(q_norm, 1.0, atol=manifold.atol)

    def test_create_se3_for_robotics(self):
        """Test SE(3) robotics preset."""
        manifold = create_se3_for_robotics()

        assert isinstance(manifold, SE3)
        assert manifold.atol == 1e-6  # Tighter tolerance for robotics


class TestFactoryIntegration:
    """Test factory function integration and consistency."""

    def test_factory_vs_direct_instantiation_poincare_ball(self):
        """Test factory function produces same result as direct instantiation."""
        # Factory creation
        factory_manifold = create_poincare_ball(dimension=3, curvature=-1.5, tolerance=1e-10)

        # Direct instantiation
        direct_manifold = PoincareBall(dimension=3, curvature=-1.5, tolerance=1e-10)

        # Should have same properties
        assert factory_manifold.dimension == direct_manifold.dimension
        assert factory_manifold.ambient_dimension == direct_manifold.ambient_dimension
        assert factory_manifold.curvature == direct_manifold.curvature
        assert factory_manifold.tolerance == direct_manifold.tolerance

    def test_factory_vs_direct_instantiation_lorentz(self):
        """Test factory function produces same result as direct instantiation."""
        # Factory creation
        factory_manifold = create_lorentz(dimension=4, atol=1e-12)

        # Direct instantiation
        direct_manifold = Lorentz(dimension=4, atol=1e-12)

        # Should have same properties
        assert factory_manifold.dimension == direct_manifold.dimension
        assert factory_manifold.ambient_dimension == direct_manifold.ambient_dimension
        assert factory_manifold.atol == direct_manifold.atol

    def test_factory_vs_direct_instantiation_se3(self):
        """Test factory function produces same result as direct instantiation."""
        # Factory creation
        factory_manifold = create_se3(atol=1e-9)

        # Direct instantiation
        direct_manifold = SE3(atol=1e-9)

        # Should have same properties
        assert factory_manifold.dimension == direct_manifold.dimension
        assert factory_manifold.ambient_dimension == direct_manifold.ambient_dimension
        assert factory_manifold.atol == direct_manifold.atol

    def test_all_factories_produce_valid_manifolds(self):
        """Test that all factory functions produce valid manifolds."""
        key = jax.random.PRNGKey(789)

        # Test each factory function
        manifolds_and_keys = [
            (create_poincare_ball(dimension=3), key),
            (create_lorentz(dimension=2), key),
            (create_se3(), key),
            (create_poincare_ball_for_embeddings(dimension=5), key),
            (create_se3_for_robotics(), key),
        ]

        for manifold, test_key in manifolds_and_keys:
            # Test basic operations work
            point = manifold.random_point(test_key)
            assert manifold.validate_point(point)

            # Test manifold properties
            assert manifold.dimension > 0
            assert manifold.ambient_dimension > 0
            # Check tolerance exists (different names for different manifolds)
            assert hasattr(manifold, 'atol') or hasattr(manifold, 'tolerance')

    def test_factory_error_messages_are_clear(self):
        """Test that factory functions provide clear error messages."""

        # Test PoincareBall errors
        with pytest.raises(ValueError) as exc_info:
            create_poincare_ball(dimension=-1)
        assert "Dimension must be positive" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            create_poincare_ball(dimension=2, curvature=1.0)
        assert "Curvature must be negative" in str(exc_info.value)

        # Test Lorentz errors
        with pytest.raises(ValueError) as exc_info:
            create_lorentz(dimension=0)
        assert "Dimension must be positive" in str(exc_info.value)

        # Test SE(3) errors
        with pytest.raises(ValueError) as exc_info:
            create_se3(atol=-1.0)
        assert "Absolute tolerance must be positive" in str(exc_info.value)

    def test_factory_consistency_with_same_parameters(self):
        """Test that factory functions are consistent across calls."""
        # Multiple calls with same parameters should produce equivalent manifolds
        manifold1 = create_poincare_ball(dimension=2, curvature=-0.5)
        manifold2 = create_poincare_ball(dimension=2, curvature=-0.5)

        # Should have identical properties
        assert manifold1.dimension == manifold2.dimension
        assert manifold1.curvature == manifold2.curvature
        assert manifold1.tolerance == manifold2.tolerance

        # Should produce same results with same random key
        key = jax.random.PRNGKey(999)
        point1 = manifold1.random_point(key)
        point2 = manifold2.random_point(key)

        assert jnp.allclose(point1, point2)


if __name__ == "__main__":
    pytest.main([__file__])
