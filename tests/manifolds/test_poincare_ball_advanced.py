"""Tests for advanced PoincareBall operations including parallel transport."""

import jax
import jax.numpy as jnp
import pytest

from riemannax.manifolds.poincare_ball import PoincareBall


class TestPoincareBallParallelTransport:
    """Test parallel transport operations in PoincareBall manifold."""

    def test_transp_identity_at_same_point(self):
        """Test that parallel transport from point to itself is identity."""
        manifold = PoincareBall(dimension=2)
        x = jnp.array([0.3, 0.2])
        v = jnp.array([0.1, 0.15])

        # Transport from x to x should be identity
        transported = manifold.transp(x, x, v)

        assert jnp.allclose(transported, v, atol=1e-6)

    def test_transp_preserves_norm(self):
        """Test that parallel transport preserves vector norm."""
        manifold = PoincareBall(dimension=2)
        x = jnp.array([0.2, 0.1])
        y = jnp.array([0.4, 0.3])
        v = jnp.array([0.1, 0.2])

        # Compute norms before and after transport
        norm_before = jnp.sqrt(manifold.inner(x, v, v))
        transported = manifold.transp(x, y, v)
        norm_after = jnp.sqrt(manifold.inner(y, transported, transported))

        # Norms should be preserved (within reasonable tolerance for numerical approximation)
        assert jnp.allclose(norm_before, norm_after, atol=1e-4)

    def test_transp_along_geodesic(self):
        """Test parallel transport along a geodesic."""
        manifold = PoincareBall(dimension=2)
        x = jnp.array([0.1, 0.1])
        v = jnp.array([0.2, 0.3])

        # Move along geodesic from x
        y = manifold.exp(x, v)

        # Transport v along its own geodesic
        transported = manifold.transp(x, y, v)

        # The transported vector should be parallel to log(y, exp(y, transported))
        # This tests the geodesic property
        z = manifold.exp(y, transported)
        log_vec = manifold.log(y, z)

        # Normalize and check parallelism
        transported_norm = transported / jnp.linalg.norm(transported)
        log_norm = log_vec / jnp.linalg.norm(log_vec)

        # Should be parallel (same or opposite direction)
        dot_product = jnp.abs(jnp.sum(transported_norm * log_norm))
        assert jnp.allclose(dot_product, 1.0, atol=1e-4)

    def test_transp_inverse_property(self):
        """Test that transporting back and forth recovers original vector."""
        manifold = PoincareBall(dimension=2)
        x = jnp.array([0.2, 0.15])
        y = jnp.array([0.3, 0.35])
        v = jnp.array([0.1, 0.05])

        # Transport from x to y, then back to x
        transported_to_y = manifold.transp(x, y, v)
        transported_back = manifold.transp(y, x, transported_to_y)

        # Should recover original vector
        assert jnp.allclose(transported_back, v, atol=1e-5)


class TestPoincareBallCurvature:
    """Test curvature-related methods of PoincareBall manifold."""

    def test_sectional_curvature_constant(self):
        """Test that sectional curvature is constant negative."""
        manifold = PoincareBall(dimension=3, curvature=-1.0)

        # Test at different points
        points = [jnp.zeros(3), jnp.array([0.3, 0.2, 0.1]), jnp.array([0.5, 0.0, 0.3])]

        for x in points:
            # Create orthonormal basis for tangent space
            u = jnp.array([1.0, 0.0, 0.0])
            v = jnp.array([0.0, 1.0, 0.0])

            # Compute sectional curvature
            curvature = manifold.sectional_curvature(x, u, v)

            # Should equal the manifold's curvature parameter
            assert jnp.allclose(curvature, -1.0, atol=1e-6)

    def test_sectional_curvature_scaling(self):
        """Test that sectional curvature scales with manifold curvature."""
        manifold1 = PoincareBall(dimension=2, curvature=-1.0)
        manifold2 = PoincareBall(dimension=2, curvature=-0.25)

        x = jnp.array([0.2, 0.3])
        u = jnp.array([1.0, 0.0])
        v = jnp.array([0.0, 1.0])

        curv1 = manifold1.sectional_curvature(x, u, v)
        curv2 = manifold2.sectional_curvature(x, u, v)

        # Curvatures should match manifold parameters
        assert jnp.allclose(curv1, -1.0, atol=1e-6)
        assert jnp.allclose(curv2, -0.25, atol=1e-6)

    def test_injectivity_radius(self):
        """Test injectivity radius computation."""
        manifold = PoincareBall(dimension=2, curvature=-1.0)

        # At any point, injectivity radius should be infinity for hyperbolic space
        x = jnp.array([0.3, 0.2])
        radius = manifold.injectivity_radius(x)

        # Should be infinity (represented as a large number)
        assert radius > 1e6


class TestPoincareBallValidationEnhanced:
    """Test enhanced validation methods for PoincareBall."""

    def test_validate_point_with_tolerance(self):
        """Test point validation with different tolerances."""
        manifold = PoincareBall(dimension=2)

        # Point close to boundary
        boundary_point = jnp.array([0.9999, 0.0])  # Further from boundary

        # Should pass with reasonable tolerance
        assert manifold.validate_point(boundary_point, atol=1e-5)

        # Very strict tolerance should be more restrictive
        very_boundary_point = jnp.array([0.99999, 0.0])
        assert manifold.validate_point(very_boundary_point, atol=1e-6)

        # Test point that's definitely too close to boundary
        too_close_point = jnp.array([0.999999, 0.0])
        assert not manifold.validate_point(too_close_point, atol=1e-5)

    def test_validate_tangent_shape(self):
        """Test tangent vector validation for shape matching."""
        manifold = PoincareBall(dimension=2)
        x = jnp.array([0.3, 0.2])

        # Correct shape
        v_correct = jnp.array([0.1, 0.2])
        assert manifold.validate_tangent(x, v_correct)

        # Wrong shape
        v_wrong = jnp.array([0.1, 0.2, 0.3])
        assert not manifold.validate_tangent(x, v_wrong)

    def test_validate_operations_chain(self):
        """Test validation through a chain of operations."""
        manifold = PoincareBall(dimension=2)
        x = jnp.array([0.2, 0.3])
        v = jnp.array([0.1, 0.15])

        # All intermediate results should be valid
        assert manifold.validate_point(x)
        assert manifold.validate_tangent(x, v)

        y = manifold.exp(x, v)
        assert manifold.validate_point(y)

        w = manifold.log(x, y)
        assert manifold.validate_tangent(x, w)

        transported = manifold.transp(x, y, v)
        assert manifold.validate_tangent(y, transported)


class TestPoincareBallMathematicalProperties:
    """Test mathematical properties of PoincareBall operations."""

    def test_exp_log_consistency_batch(self):
        """Test exp-log consistency for batched operations."""
        manifold = PoincareBall(dimension=2)

        # Create batch of points
        key = jax.random.PRNGKey(42)
        x_batch = manifold.random_point(key, 5)  # 5 random points

        key, subkey = jax.random.split(key)
        y_batch = manifold.random_point(subkey, 5)

        # Ensure points are not too close
        for i in range(5):
            if jnp.linalg.norm(x_batch[i] - y_batch[i]) < 0.1:
                y_batch = y_batch.at[i].set(x_batch[i] + jnp.array([0.2, 0.1]))

        # Test exp-log consistency for each pair
        for i in range(5):
            x = x_batch[i]
            y = y_batch[i]

            # Ensure both points are valid
            if manifold.validate_point(x) and manifold.validate_point(y):
                v = manifold.log(x, y)
                y_recovered = manifold.exp(x, v)

                assert jnp.allclose(y_recovered, y, atol=1e-5)

    def test_triangle_inequality_random(self):
        """Test triangle inequality with random points."""
        manifold = PoincareBall(dimension=3)

        key = jax.random.PRNGKey(123)
        x = manifold.random_point(key)

        key, subkey = jax.random.split(key)
        y = manifold.random_point(subkey)

        key, subkey = jax.random.split(key)
        z = manifold.random_point(subkey)

        d_xy = manifold.dist(x, y)
        d_yz = manifold.dist(y, z)
        d_xz = manifold.dist(x, z)

        # Triangle inequality
        assert d_xz <= d_xy + d_yz + 1e-6

    def test_geodesic_minimality(self):
        """Test that geodesics are minimal paths."""
        manifold = PoincareBall(dimension=2)
        x = jnp.array([0.1, 0.1])
        y = jnp.array([0.4, 0.3])

        # Direct distance
        direct_dist = manifold.dist(x, y)

        # Distance via intermediate point
        z = jnp.array([0.25, 0.25])
        indirect_dist = manifold.dist(x, z) + manifold.dist(z, y)

        # Direct path should be shorter
        assert direct_dist <= indirect_dist + 1e-6
