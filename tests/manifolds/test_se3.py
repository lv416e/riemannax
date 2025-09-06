"""Tests for SE(3) Special Euclidean Group manifold."""

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from riemannax.manifolds.se3 import SE3


class TestSE3Initialization:
    """Test SE(3) manifold initialization and basic properties."""

    def test_default_initialization(self):
        """Test default SE(3) manifold initialization."""
        manifold = SE3()
        assert hasattr(manifold, 'atol')
        assert manifold.atol == 1e-8
        assert manifold.dimension == 6  # SE(3) has 6 DOF
        assert manifold.ambient_dimension == 7  # (qw, qx, qy, qz, x, y, z)

    def test_custom_tolerance_initialization(self):
        """Test SE(3) initialization with custom tolerance."""
        custom_atol = 1e-10
        manifold = SE3(atol=custom_atol)
        assert manifold.atol == custom_atol

    def test_repr_string(self):
        """Test string representation of SE(3) manifold."""
        manifold = SE3()
        repr_str = repr(manifold)
        assert "SE3" in repr_str


class TestSE3QuaternionOperations:
    """Test quaternion normalization and basic SE(3) operations."""

    def test_quaternion_normalize(self):
        """Test quaternion normalization preserves unit norm."""
        manifold = SE3()
        
        # Test with random quaternion
        key = jax.random.PRNGKey(42)
        q = jax.random.normal(key, (4,))
        
        normalized_q = manifold._quaternion_normalize(q)
        norm = jnp.linalg.norm(normalized_q)
        assert jnp.allclose(norm, 1.0, atol=manifold.atol)

    def test_quaternion_normalize_zero_handling(self):
        """Test quaternion normalization handles zero quaternion gracefully."""
        manifold = SE3()
        
        # Zero quaternion should normalize to (1, 0, 0, 0)
        zero_q = jnp.array([0.0, 0.0, 0.0, 0.0])
        normalized_q = manifold._quaternion_normalize(zero_q)
        expected = jnp.array([1.0, 0.0, 0.0, 0.0])
        assert jnp.allclose(normalized_q, expected, atol=manifold.atol)


class TestSE3RandomGeneration:
    """Test random point generation for SE(3)."""

    def test_random_point_generation(self):
        """Test random SE(3) transform generation."""
        manifold = SE3()
        key = jax.random.PRNGKey(123)
        
        point = manifold.random_point(key)
        
        # Check shape: (7,) for (qw, qx, qy, qz, x, y, z)
        assert point.shape == (7,)
        
        # Check quaternion part is normalized
        q = point[:4]  # quaternion part
        q_norm = jnp.linalg.norm(q)
        assert jnp.allclose(q_norm, 1.0, atol=manifold.atol)

    def test_random_point_batch_generation(self):
        """Test batch generation of random SE(3) transforms."""
        manifold = SE3()
        key = jax.random.PRNGKey(456)
        batch_size = 5
        
        points = manifold.random_point(key, batch_size)
        
        # Check shape: (5, 7)
        assert points.shape == (batch_size, 7)
        
        # Check all quaternions are normalized
        quaternions = points[:, :4]
        norms = jnp.linalg.norm(quaternions, axis=1)
        assert jnp.allclose(norms, 1.0, atol=manifold.atol)

    def test_random_point_reproducibility(self):
        """Test random point generation is reproducible with same key."""
        manifold = SE3()
        key = jax.random.PRNGKey(789)
        
        point1 = manifold.random_point(key)
        point2 = manifold.random_point(key)
        
        assert jnp.allclose(point1, point2, atol=manifold.atol)


class TestSE3Validation:
    """Test SE(3) point and tangent vector validation."""

    def test_validate_point_valid_transforms(self):
        """Test validation of valid SE(3) transforms."""
        manifold = SE3()
        key = jax.random.PRNGKey(101)
        
        # Generate valid random point
        valid_point = manifold.random_point(key)
        
        # Should validate successfully
        is_valid = manifold.validate_point(valid_point)
        assert is_valid

    def test_validate_point_invalid_quaternion(self):
        """Test validation catches invalid quaternions."""
        manifold = SE3()
        
        # Create point with non-normalized quaternion
        invalid_point = jnp.array([2.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
        
        is_valid = manifold.validate_point(invalid_point)
        assert not is_valid

    def test_validate_point_wrong_shape(self):
        """Test validation catches wrong shape."""
        manifold = SE3()
        
        # Wrong shape point
        wrong_shape = jnp.array([1.0, 0.0, 0.0])
        
        is_valid = manifold.validate_point(wrong_shape)
        assert not is_valid


class TestSE3GroupStructure:
    """Test SE(3) group structure properties."""

    def test_identity_element(self):
        """Test SE(3) identity element properties."""
        manifold = SE3()
        
        # Identity should have quaternion (1,0,0,0) and zero translation
        identity = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Identity should be valid
        assert manifold.validate_point(identity)
        
        # Identity applied to any point should return the same point
        key = jax.random.PRNGKey(202)
        test_point = jax.random.normal(key, (3,))  # 3D point to transform
        
        # This will be implemented in later tasks
        # transformed = manifold.apply(identity, test_point)
        # assert jnp.allclose(transformed, test_point, atol=manifold.atol)


class TestSE3NumericalStability:
    """Test numerical stability of SE(3) operations."""

    def test_quaternion_normalization_stability(self):
        """Test quaternion normalization numerical stability."""
        manifold = SE3()
        
        # Test with very small quaternion
        small_q = jnp.array([1e-10, 1e-10, 1e-10, 1e-10])
        normalized = manifold._quaternion_normalize(small_q)
        norm = jnp.linalg.norm(normalized)
        assert jnp.allclose(norm, 1.0, atol=manifold.atol)
        
        # Test with very large quaternion  
        large_q = jnp.array([1e10, 1e10, 1e10, 1e10])
        normalized = manifold._quaternion_normalize(large_q)
        norm = jnp.linalg.norm(normalized)
        assert jnp.allclose(norm, 1.0, atol=manifold.atol)

    def test_random_point_generation_stability(self):
        """Test random point generation numerical stability."""
        manifold = SE3()
        key = jax.random.PRNGKey(303)
        
        # Generate many random points and check all are valid
        for i in range(10):
            subkey = jax.random.fold_in(key, i)
            point = manifold.random_point(subkey)
            assert manifold.validate_point(point), f"Point {i} failed validation"


if __name__ == "__main__":
    pytest.main([__file__])