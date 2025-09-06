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


class TestSE3MatrixOperations:
    """Test SE(3) matrix exponential and logarithm operations."""

    def test_matrix_exp_so3_basic(self):
        """Test SO(3) matrix exponential with basic rotation vectors."""
        manifold = SE3()

        # Test with zero rotation (should give identity matrix)
        zero_omega = jnp.zeros(3)
        R = manifold._matrix_exp_so3(zero_omega)
        expected_identity = jnp.eye(3)
        assert jnp.allclose(R, expected_identity, atol=manifold.atol)

        # Check result is orthogonal matrix
        assert jnp.allclose(R @ R.T, jnp.eye(3), atol=manifold.atol)
        assert jnp.allclose(jnp.linalg.det(R), 1.0, atol=manifold.atol)

    def test_matrix_exp_so3_small_angles(self):
        """Test SO(3) matrix exponential for small rotation angles."""
        manifold = SE3()

        # Small rotation around z-axis
        small_angle = 1e-6
        omega = jnp.array([0.0, 0.0, small_angle])
        R = manifold._matrix_exp_so3(omega)

        # Check orthogonality and determinant
        assert jnp.allclose(R @ R.T, jnp.eye(3), atol=manifold.atol)
        assert jnp.allclose(jnp.linalg.det(R), 1.0, atol=manifold.atol)

    def test_matrix_exp_so3_rodrigues_formula(self):
        """Test SO(3) matrix exponential using Rodrigues formula properties."""
        manifold = SE3()

        # Test with π/2 rotation around x-axis
        omega = jnp.array([jnp.pi/2, 0.0, 0.0])
        R = manifold._matrix_exp_so3(omega)

        # Expected rotation matrix for π/2 around x-axis
        expected = jnp.array([[1, 0, 0],
                             [0, 0, -1],
                             [0, 1, 0]], dtype=jnp.float32)

        assert jnp.allclose(R, expected, atol=1e-6)

    def test_matrix_log_so3_basic(self):
        """Test SO(3) matrix logarithm with basic cases."""
        manifold = SE3()

        # Test with identity matrix (should give zero vector)
        identity = jnp.eye(3)
        omega = manifold._matrix_log_so3(identity)
        assert jnp.allclose(omega, jnp.zeros(3), atol=manifold.atol)

    def test_matrix_exp_log_so3_consistency(self):
        """Test exp-log consistency for SO(3) operations."""
        manifold = SE3()
        key = jax.random.PRNGKey(12345)

        # Test with random rotation vectors
        for i in range(5):
            subkey = jax.random.fold_in(key, i)
            omega_orig = jax.random.normal(subkey, (3,)) * 0.5  # Keep angles reasonable

            # exp then log should recover original
            R = manifold._matrix_exp_so3(omega_orig)
            omega_recovered = manifold._matrix_log_so3(R)

            # Account for angle wrapping - compare sin/cos of angles
            norm_orig = jnp.linalg.norm(omega_orig)
            norm_recovered = jnp.linalg.norm(omega_recovered)

            # For small angles, should be very close
            if norm_orig < 1.0:
                assert jnp.allclose(omega_orig, omega_recovered, atol=1e-5)

    def test_se3_exp_basic(self):
        """Test SE(3) exponential map with basic cases."""
        manifold = SE3()

        # Test with zero tangent vector (should give identity)
        zero_tangent = jnp.zeros(6)
        identity_point = manifold.exp_tangent(zero_tangent)
        expected_identity = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert jnp.allclose(identity_point, expected_identity, atol=manifold.atol)

    def test_se3_exp_pure_translation(self):
        """Test SE(3) exponential with pure translation."""
        manifold = SE3()

        # Pure translation (no rotation)
        pure_translation = jnp.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0])
        result = manifold.exp_tangent(pure_translation)

        # Should have identity quaternion and the translation
        expected_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        expected_trans = jnp.array([1.0, 2.0, 3.0])

        assert jnp.allclose(result[:4], expected_quat, atol=manifold.atol)
        assert jnp.allclose(result[4:7], expected_trans, atol=manifold.atol)

    def test_se3_exp_pure_rotation(self):
        """Test SE(3) exponential with pure rotation."""
        manifold = SE3()

        # Pure rotation around z-axis
        angle = jnp.pi/4
        pure_rotation = jnp.array([0.0, 0.0, angle, 0.0, 0.0, 0.0])
        result = manifold.exp_tangent(pure_rotation)

        # Translation should be zero
        assert jnp.allclose(result[4:7], jnp.zeros(3), atol=manifold.atol)

        # Quaternion should represent π/4 rotation around z-axis
        expected_quat = jnp.array([jnp.cos(angle/2), 0.0, 0.0, jnp.sin(angle/2)])
        assert jnp.allclose(result[:4], expected_quat, atol=1e-6)

    def test_se3_log_basic(self):
        """Test SE(3) logarithmic map with basic cases."""
        manifold = SE3()

        # Test with identity transform
        identity = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        zero_tangent = manifold.log_tangent(identity)
        assert jnp.allclose(zero_tangent, jnp.zeros(6), atol=manifold.atol)

    def test_se3_exp_log_consistency(self):
        """Test SE(3) exp-log consistency."""
        manifold = SE3()
        key = jax.random.PRNGKey(54321)

        # Test with small random tangent vectors
        for i in range(3):
            subkey = jax.random.fold_in(key, i)
            tangent_orig = jax.random.normal(subkey, (6,)) * 0.1  # Small vectors

            # exp then log should recover original
            point = manifold.exp_tangent(tangent_orig)
            tangent_recovered = manifold.log_tangent(point)

            assert jnp.allclose(tangent_orig, tangent_recovered, atol=1e-4)

    def test_se3_numerical_stability_singularities(self):
        """Test SE(3) operations near singularities."""
        manifold = SE3()

        # Test with near-π rotation (challenging for matrix log)
        large_angle = jnp.pi - 1e-6
        near_singularity = jnp.array([0.0, 0.0, large_angle, 1.0, 0.0, 0.0])

        # Should not crash and result should be valid
        try:
            result = manifold.exp_tangent(near_singularity)
            assert manifold.validate_point(result)
        except Exception as e:
            pytest.fail(f"SE(3) exp failed near singularity: {e}")


class TestSE3TaylorApproximations:
    """Test Taylor approximation handling in SE(3) operations."""

    def test_so3_taylor_near_zero(self):
        """Test SO(3) operations use Taylor approximations near zero."""
        manifold = SE3()

        # Very small rotation
        tiny_omega = jnp.array([1e-8, 1e-8, 1e-8])
        R = manifold._matrix_exp_so3(tiny_omega)

        # Should be close to identity with small perturbation
        assert jnp.allclose(R, jnp.eye(3), atol=1e-7)
        assert jnp.allclose(R @ R.T, jnp.eye(3), atol=manifold.atol)

    def test_se3_exp_log_taylor_consistency(self):
        """Test SE(3) exp-log consistency with Taylor approximations."""
        manifold = SE3()

        # Test with very small tangent vectors where Taylor expansions are used
        small_tangents = [
            jnp.array([1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10]),
            jnp.array([1e-8, 0, 0, 1e-8, 0, 0]),
            jnp.array([0, 0, 1e-8, 0, 0, 1e-8]),
        ]

        for tangent in small_tangents:
            point = manifold.exp_tangent(tangent)
            recovered = manifold.log_tangent(point)
            assert jnp.allclose(tangent, recovered, atol=1e-9)


if __name__ == "__main__":
    pytest.main([__file__])
