"""Tests for Lorentz hyperboloid manifold implementation."""

import jax
import jax.numpy as jnp
import pytest

from riemannax.manifolds.lorentz import Lorentz


class TestLorentzInitialization:
    """Test Lorentz manifold initialization and basic properties."""

    def test_default_initialization(self):
        """Test default initialization with standard parameters."""
        manifold = Lorentz(dimension=2)

        assert manifold.dimension == 2
        assert manifold.curvature == -1.0
        assert manifold.atol == 1e-8
        assert manifold.name == "Lorentz"

    def test_custom_dimension_initialization(self):
        """Test initialization with custom dimension."""
        manifold = Lorentz(dimension=5)

        assert manifold.dimension == 5
        assert manifold.curvature == -1.0

    def test_custom_curvature_initialization(self):
        """Test initialization with custom curvature."""
        manifold = Lorentz(dimension=3, curvature=-2.5)

        assert manifold.dimension == 3
        assert manifold.curvature == -2.5

    def test_invalid_dimension_error(self):
        """Test that invalid dimensions raise ValueError."""
        with pytest.raises(ValueError, match="Dimension must be positive"):
            Lorentz(dimension=0)

    def test_invalid_curvature_error(self):
        """Test that non-negative curvature raises ValueError."""
        with pytest.raises(ValueError, match="Curvature must be negative"):
            Lorentz(dimension=2, curvature=1.0)

        with pytest.raises(ValueError, match="Curvature must be negative"):
            Lorentz(dimension=2, curvature=0.0)

    def test_repr_string(self):
        """Test string representation of Lorentz manifold."""
        manifold = Lorentz(dimension=3, curvature=-1.5)
        repr_str = repr(manifold)

        assert "Lorentz" in repr_str
        assert "dim=3" in repr_str
        assert "c=-1.5" in repr_str


class TestLorentzMinkowskiOperations:
    """Test Minkowski inner product operations."""

    def test_minkowski_inner_basic(self):
        """Test basic Minkowski inner product computation."""
        manifold = Lorentz(dimension=2)

        # Test vectors in R^3 (time + 2 spatial dimensions)
        u = jnp.array([2.0, 1.0, 1.0])  # (t=2, x=1, y=1)
        v = jnp.array([1.5, 0.5, 1.5])  # (t=1.5, x=0.5, y=1.5)

        # Expected: 2*1.5 - 1*0.5 - 1*1.5 = 3 - 0.5 - 1.5 = 1
        result = manifold._minkowski_inner(u, v)
        expected = 2.0 * 1.5 - 1.0 * 0.5 - 1.0 * 1.5

        assert jnp.allclose(result, expected)

    def test_minkowski_inner_self(self):
        """Test Minkowski inner product with itself."""
        manifold = Lorentz(dimension=2)

        # Vector on hyperboloid should have B(x,x) = 1
        x = jnp.array([2.0, jnp.sqrt(3.0), 0.0])  # x₀² - x₁² - x₂² = 4 - 3 - 0 = 1

        result = manifold._minkowski_inner(x, x)
        # Should be 1 for points on hyperboloid
        expected = 1.0

        assert jnp.allclose(result, expected)

    def test_minkowski_inner_orthogonal(self):
        """Test Minkowski inner product of orthogonal vectors."""
        manifold = Lorentz(dimension=2)

        # Point on hyperboloid
        x = jnp.array([jnp.sqrt(2.0), 1.0, 0.0])

        # Tangent vector orthogonal to x
        v = jnp.array([1.0 / jnp.sqrt(2.0), 1.0, 0.0])  # Constructed to be B(x,v) = 0

        result = manifold._minkowski_inner(x, v)

        assert jnp.allclose(result, 0.0, atol=1e-6)

    def test_minkowski_inner_batch(self):
        """Test Minkowski inner product with batched inputs."""
        manifold = Lorentz(dimension=2)

        # Batch of vectors
        u = jnp.array([[2.0, 1.0, 1.0], [1.5, 0.5, 0.5]])
        v = jnp.array([[1.0, 0.5, 0.5], [2.0, 1.0, 1.0]])

        result = manifold._minkowski_inner(u, v)

        # Expected for each pair (using +time - space convention)
        expected = jnp.array(
            [
                2.0 * 1.0 - 1.0 * 0.5 - 1.0 * 0.5,  # 2 - 0.5 - 0.5 = 1
                1.5 * 2.0 - 0.5 * 1.0 - 0.5 * 1.0,  # 3 - 0.5 - 0.5 = 2
            ]
        )

        assert jnp.allclose(result, expected)


class TestLorentzRandomGeneration:
    """Test random point and tangent vector generation."""

    def test_random_point_generation(self):
        """Test basic random point generation."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(42)

        point = manifold.random_point(key)

        # Point should be in R^3 (dimension + 1)
        assert point.shape == (3,)

        # Point should satisfy hyperboloid constraint
        assert manifold.validate_point(point)

    def test_random_point_hyperboloid_constraint(self):
        """Test that random points satisfy hyperboloid constraint."""
        manifold = Lorentz(dimension=3)
        key = jax.random.PRNGKey(123)

        # Generate multiple points
        points = manifold.random_point(key, 10)

        # All points should satisfy constraint
        for i in range(10):
            point = points[i]
            minkowski_norm = manifold._minkowski_inner(point, point)

            # Should be 1 for points on hyperboloid
            assert jnp.allclose(minkowski_norm, 1.0, atol=1e-6)

            # Time component should be positive
            assert point[0] > 0

    def test_random_point_with_custom_dimension(self):
        """Test random point generation with custom dimensions."""
        manifold = Lorentz(dimension=5)
        key = jax.random.PRNGKey(456)

        point = manifold.random_point(key)

        # Point should be in R^6 (dimension + 1)
        assert point.shape == (6,)
        assert manifold.validate_point(point)

    def test_random_point_batch_generation(self):
        """Test batch random point generation."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(789)

        points = manifold.random_point(key, 5, 3)

        # Shape should be (5, 3, 3)
        assert points.shape == (5, 3, 3)

        # All points should be valid
        for i in range(5):
            for j in range(3):
                assert manifold.validate_point(points[i, j])

    def test_random_tangent_generation(self):
        """Test basic random tangent vector generation."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(101)

        x = manifold.random_point(key)
        key, subkey = jax.random.split(key)
        v = manifold.random_tangent(subkey, x)

        # Tangent vector should have same shape as point
        assert v.shape == x.shape

        # Should be valid tangent vector
        assert manifold.validate_tangent(x, v)

    def test_random_tangent_orthogonality(self):
        """Test that random tangent vectors are orthogonal to base point."""
        manifold = Lorentz(dimension=3)
        key = jax.random.PRNGKey(202)

        x = manifold.random_point(key)
        key, subkey = jax.random.split(key)

        # Generate multiple tangent vectors
        for _ in range(5):
            key, subkey = jax.random.split(key)
            v = manifold.random_tangent(subkey, x)

            # Should be orthogonal: B(x, v) = 0
            minkowski_prod = manifold._minkowski_inner(x, v)
            assert jnp.allclose(minkowski_prod, 0.0, atol=1e-6)

    def test_random_generation_reproducibility(self):
        """Test that random generation is reproducible with same key."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(303)

        point1 = manifold.random_point(key)
        point2 = manifold.random_point(key)

        assert jnp.allclose(point1, point2)


class TestLorentzValidation:
    """Test validation methods for points and tangent vectors."""

    def test_validate_point_valid_points(self):
        """Test validation of valid points on hyperboloid."""
        manifold = Lorentz(dimension=2)

        # Valid points on hyperboloid
        valid_points = [
            jnp.array([1.0, 0.0, 0.0]),  # Origin in hyperbolic space
            jnp.array([jnp.sqrt(2.0), 1.0, 0.0]),  # Point with x₁ = 1
            jnp.array([jnp.sqrt(6.0), 1.0, 2.0]),  # Point with x₁=1, x₂=2
        ]

        for point in valid_points:
            assert manifold.validate_point(point)

    def test_validate_point_invalid_points(self):
        """Test validation of invalid points."""
        manifold = Lorentz(dimension=2)

        # Invalid points
        invalid_points = [
            jnp.array([0.5, 1.0, 0.0]),  # Doesn't satisfy constraint
            jnp.array([-2.0, 1.0, 1.0]),  # Negative time component
            jnp.array([1.0, 1.0, 1.0]),  # Wrong constraint value
        ]

        for point in invalid_points:
            assert not manifold.validate_point(point)

    def test_validate_point_with_tolerance(self):
        """Test point validation with different tolerances."""
        manifold = Lorentz(dimension=2)

        # Point slightly off the hyperboloid
        point = jnp.array([1.0001, 0.0, 0.0])  # Slightly off x₀² - x₁² - x₂² = 1

        # Should fail with strict tolerance
        assert not manifold.validate_point(point, atol=1e-6)

        # Should pass with loose tolerance
        assert manifold.validate_point(point, atol=1e-3)

    def test_validate_tangent_valid_vectors(self):
        """Test validation of valid tangent vectors."""
        manifold = Lorentz(dimension=2)

        # Point on hyperboloid
        x = jnp.array([jnp.sqrt(2.0), 1.0, 0.0])

        # Valid tangent vector (orthogonal to x)
        v = jnp.array([1.0 / jnp.sqrt(2.0), 1.0, 1.0])

        assert manifold.validate_tangent(x, v)

    def test_validate_tangent_invalid_vectors(self):
        """Test validation of invalid tangent vectors."""
        manifold = Lorentz(dimension=2)

        x = jnp.array([jnp.sqrt(2.0), 1.0, 0.0])

        # Invalid tangent vectors
        invalid_tangents = [
            jnp.array([1.0, 0.0, 0.0]),  # Not orthogonal
            jnp.array([1.0, 1.0]),  # Wrong shape
        ]

        for v in invalid_tangents:
            assert not manifold.validate_tangent(x, v)

    def test_validate_operations_chain(self):
        """Test validation through a chain of operations."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(404)

        # Generate point and tangent vector
        x = manifold.random_point(key)
        key, subkey = jax.random.split(key)
        v = manifold.random_tangent(subkey, x)

        # All should be valid
        assert manifold.validate_point(x)
        assert manifold.validate_tangent(x, v)


class TestLorentzNumericalStability:
    """Test numerical stability and edge cases."""

    def test_numerical_stability_integration(self):
        """Test numerical stability with various operations."""
        manifold = Lorentz(dimension=3)
        key = jax.random.PRNGKey(505)

        # Generate test data
        x = manifold.random_point(key)
        key, subkey = jax.random.split(key)
        v = manifold.random_tangent(subkey, x)

        # Operations should be numerically stable
        minkowski_x_x = manifold._minkowski_inner(x, x)
        minkowski_x_v = manifold._minkowski_inner(x, v)

        # Point should satisfy hyperboloid constraint
        assert jnp.allclose(minkowski_x_x, 1.0, atol=1e-10)

        # Tangent should be orthogonal
        assert jnp.allclose(minkowski_x_v, 0.0, atol=1e-10)

    def test_large_dimension_stability(self):
        """Test stability with larger dimensions."""
        manifold = Lorentz(dimension=10)
        key = jax.random.PRNGKey(606)

        x = manifold.random_point(key)

        # Should still satisfy constraints
        assert manifold.validate_point(x)

        minkowski_norm = manifold._minkowski_inner(x, x)
        assert jnp.allclose(minkowski_norm, 1.0, atol=1e-8)
