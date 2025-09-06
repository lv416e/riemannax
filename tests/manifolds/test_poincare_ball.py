"""Tests for Poincaré ball manifold implementation."""

import jax
import jax.numpy as jnp
import pytest

from riemannax.manifolds.base import ManifoldError
from riemannax.manifolds.poincare_ball import PoincareBall


class TestPoincareBallInitialization:
    """Test PoincareBall manifold initialization."""

    def test_default_initialization(self):
        """Test PoincareBall with default parameters."""
        manifold = PoincareBall()

        assert manifold.dimension == 2  # Default 2D Poincaré disk
        assert manifold.ambient_dimension == 2
        assert manifold.curvature == -1.0
        assert manifold.tolerance == 1e-6

    def test_custom_dimension_initialization(self):
        """Test PoincareBall with custom dimension."""
        manifold = PoincareBall(dimension=3)

        assert manifold.dimension == 3
        assert manifold.ambient_dimension == 3

    def test_custom_curvature_initialization(self):
        """Test PoincareBall with custom curvature."""
        manifold = PoincareBall(curvature=-0.5)

        assert manifold.curvature == -0.5

    def test_invalid_dimension_error(self):
        """Test error for invalid dimension."""
        with pytest.raises(ManifoldError):
            PoincareBall(dimension=0)

        with pytest.raises(ManifoldError):
            PoincareBall(dimension=-1)

    def test_invalid_curvature_error(self):
        """Test error for invalid curvature (must be negative)."""
        with pytest.raises(ManifoldError):
            PoincareBall(curvature=1.0)  # Positive curvature

        with pytest.raises(ManifoldError):
            PoincareBall(curvature=0.0)  # Zero curvature

    def test_repr_string(self):
        """Test string representation."""
        manifold = PoincareBall(dimension=3, curvature=-2.0)
        repr_str = repr(manifold)

        assert "PoincareBall" in repr_str
        assert "dim=3" in repr_str
        assert "c=-2.0" in repr_str


class TestPoincareBallValidation:
    """Test PoincareBall validation methods."""

    def test_validate_in_ball_valid_points(self):
        """Test validation of points inside the ball."""
        manifold = PoincareBall(dimension=2)

        # Points inside unit ball
        valid_points = [
            jnp.array([0.0, 0.0]),  # Origin
            jnp.array([0.5, 0.3]),  # Inside ball
            jnp.array([0.8, 0.2]),  # Near boundary
        ]

        for point in valid_points:
            assert manifold._validate_in_ball(point)

    def test_validate_in_ball_invalid_points(self):
        """Test validation of points outside the ball."""
        manifold = PoincareBall(dimension=2)

        # Points outside or on unit ball boundary
        invalid_points = [
            jnp.array([1.0, 0.0]),  # On boundary
            jnp.array([1.5, 0.3]),  # Outside ball
            jnp.array([0.8, 0.8]),  # Outside ball (norm > 1)
        ]

        for point in invalid_points:
            assert not manifold._validate_in_ball(point)

    def test_validate_in_ball_with_tolerance(self):
        """Test validation with custom tolerance."""
        manifold = PoincareBall(dimension=2, tolerance=1e-4)

        # Point very close to boundary
        near_boundary = jnp.array([0.9999, 0.0])
        assert manifold._validate_in_ball(near_boundary)

        # Point slightly outside but within tolerance
        slightly_outside = jnp.array([1.00005, 0.0])
        assert not manifold._validate_in_ball(slightly_outside)

    def test_validate_point_method(self):
        """Test public validate_point method."""
        manifold = PoincareBall(dimension=2)

        valid_point = jnp.array([0.3, 0.4])
        invalid_point = jnp.array([1.2, 0.3])

        assert manifold.validate_point(valid_point)
        assert not manifold.validate_point(invalid_point)


class TestPoincareBallRandomGeneration:
    """Test random point and tangent vector generation."""

    def test_random_point_generation(self):
        """Test random point generation within the ball."""
        manifold = PoincareBall(dimension=2)
        key = jax.random.PRNGKey(42)

        # Generate single point
        point = manifold.random_point(key)
        assert point.shape == (2,)
        assert manifold.validate_point(point)

        # Generate multiple points
        points = manifold.random_point(key, 10)
        assert points.shape == (10, 2)

        # All points should be valid
        for i in range(10):
            assert manifold.validate_point(points[i])

    def test_random_point_with_custom_dimension(self):
        """Test random point generation with custom dimension."""
        manifold = PoincareBall(dimension=3)
        key = jax.random.PRNGKey(42)

        point = manifold.random_point(key)
        assert point.shape == (3,)
        assert manifold.validate_point(point)

    def test_random_point_distribution(self):
        """Test that random points are reasonably distributed."""
        manifold = PoincareBall(dimension=2)
        key = jax.random.PRNGKey(42)

        # Generate many points
        points = manifold.random_point(key, 1000)

        # Check distribution properties
        norms = jnp.linalg.norm(points, axis=1)

        # All points should be inside unit ball
        assert jnp.all(norms < 1.0)

        # Should have reasonable spread (not all clustered at origin)
        assert jnp.std(norms) > 0.1

    def test_random_tangent_generation(self):
        """Test random tangent vector generation."""
        manifold = PoincareBall(dimension=2)
        key = jax.random.PRNGKey(42)

        # Generate base point
        point = jnp.array([0.3, 0.4])

        # Generate single tangent vector
        tangent = manifold.random_tangent(key, point)
        assert tangent.shape == (2,)
        assert manifold.validate_tangent(point, tangent)

        # Generate multiple tangent vectors
        tangents = manifold.random_tangent(key, point, 5)
        assert tangents.shape == (5, 2)

        # All tangents should be valid
        for i in range(5):
            assert manifold.validate_tangent(point, tangents[i])

    def test_random_tangent_at_origin(self):
        """Test random tangent vector generation at origin."""
        manifold = PoincareBall(dimension=2)
        key = jax.random.PRNGKey(42)

        origin = jnp.zeros(2)
        tangent = manifold.random_tangent(key, origin)

        # At origin, tangent space is just Euclidean
        assert tangent.shape == (2,)
        assert manifold.validate_tangent(origin, tangent)

    def test_random_generation_reproducibility(self):
        """Test that random generation is reproducible with same key."""
        manifold = PoincareBall(dimension=2)
        key = jax.random.PRNGKey(123)

        # Generate points with same key
        point1 = manifold.random_point(key)
        point2 = manifold.random_point(key)

        # Should be identical
        assert jnp.allclose(point1, point2)


class TestPoincareBallMobiusOperations:
    """Test Möbius operations for hyperbolic geometry."""

    def test_mobius_add_identity(self):
        """Test Möbius addition with zero (identity element)."""
        manifold = PoincareBall(dimension=2)

        point = jnp.array([0.3, 0.4])
        zero = jnp.zeros(2)

        # x ⊕ 0 = x
        result1 = manifold._mobius_add(point, zero)
        assert jnp.allclose(result1, point, atol=1e-6)

        # 0 ⊕ x = x
        result2 = manifold._mobius_add(zero, point)
        assert jnp.allclose(result2, point, atol=1e-6)

    def test_mobius_add_inverse(self):
        """Test Möbius addition with inverse gives origin."""
        manifold = PoincareBall(dimension=2)

        point = jnp.array([0.3, 0.4])
        inverse_point = -point

        # x ⊕ (-x) = 0 (approximately, due to hyperbolic geometry)
        result = manifold._mobius_add(point, inverse_point)

        # Result should be close to origin
        assert jnp.linalg.norm(result) < 1e-6

    def test_mobius_add_non_commutativity(self):
        """Test that Möbius addition is generally non-commutative."""
        manifold = PoincareBall(dimension=2)

        point1 = jnp.array([0.2, 0.3])
        point2 = jnp.array([0.1, 0.4])

        # x ⊕ y ≠ y ⊕ x in general
        result1 = manifold._mobius_add(point1, point2)
        result2 = manifold._mobius_add(point2, point1)

        # They should be different in general
        assert not jnp.allclose(result1, result2, atol=1e-6)

        # Special case: commutative when one point is origin
        origin = jnp.zeros(2)
        result_origin1 = manifold._mobius_add(origin, point1)
        result_origin2 = manifold._mobius_add(point1, origin)
        assert jnp.allclose(result_origin1, point1, atol=1e-6)
        assert jnp.allclose(result_origin2, point1, atol=1e-6)

    def test_mobius_add_stays_in_ball(self):
        """Test that Möbius addition keeps results in ball."""
        manifold = PoincareBall(dimension=2)

        # Generate multiple point pairs
        key = jax.random.PRNGKey(42)
        points1 = manifold.random_point(key, 10)
        points2 = manifold.random_point(jax.random.split(key)[1], 10)

        for i in range(10):
            result = manifold._mobius_add(points1[i], points2[i])
            assert manifold.validate_point(result), f"Result {i} is outside ball: {result}"

    def test_mobius_add_boundary_behavior(self):
        """Test Möbius addition behavior near boundary."""
        manifold = PoincareBall(dimension=2)

        # Points close to boundary
        near_boundary = jnp.array([0.9, 0.0])
        small_point = jnp.array([0.1, 0.0])

        result = manifold._mobius_add(near_boundary, small_point)

        # Result should still be in ball
        assert manifold.validate_point(result)
        assert jnp.linalg.norm(result) < 1.0


class TestPoincareBallNumericalStability:
    """Test numerical stability integration."""

    def test_numerical_stability_integration(self):
        """Test integration with numerical stability manager."""
        manifold = PoincareBall(dimension=2)

        # Very large vector (should trigger stability check)
        large_point = jnp.array([0.999999, 0.0])

        # Should still validate but be near boundary
        assert manifold.validate_point(large_point)

    def test_vector_length_limits(self):
        """Test vector length limits from numerical stability."""
        manifold = PoincareBall(dimension=2)

        # Create vector near Poincaré ball stability limit
        # Based on numerical stability: stable for vectors <38 length
        large_coords = jnp.array([30.0, 20.0])  # Norm > 35

        # This should work with our numerical stability infrastructure
        # The point is outside ball but operations should handle it gracefully
        assert not manifold.validate_point(large_coords)

    def test_curvature_scaling_stability(self):
        """Test numerical stability with different curvature values."""
        # Different curvature values
        curvatures = [-0.1, -1.0, -2.0, -10.0]

        for c in curvatures:
            manifold = PoincareBall(dimension=2, curvature=c)
            key = jax.random.PRNGKey(42)

            # Should be able to generate valid points
            point = manifold.random_point(key)
            assert manifold.validate_point(point)

            # Möbius operations should remain stable
            point2 = manifold.random_point(jax.random.split(key)[1])
            result = manifold._mobius_add(point, point2)
            assert manifold.validate_point(result)
