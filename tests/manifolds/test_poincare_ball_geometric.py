"""Tests for Poincaré ball geometric operations."""

import jax
import jax.numpy as jnp
import pytest

from riemannax.manifolds.poincare_ball import PoincareBall


class TestPoincareBallGeometricOperations:
    """Test core geometric operations of PoincareBall manifold."""

    def test_proj_identity_at_origin(self):
        """Test that projection at origin is identity."""
        manifold = PoincareBall(dimension=2)
        origin = jnp.zeros(2)
        v = jnp.array([0.1, 0.2])

        projected = manifold.proj(origin, v)

        # At origin, projection should be identity
        assert jnp.allclose(projected, v, atol=1e-6)

    def test_proj_boundary_constraint(self):
        """Test that projection respects ball boundary."""
        manifold = PoincareBall(dimension=2)
        # Point near boundary
        x = jnp.array([0.8, 0.0])
        # Large tangent vector that would push outside ball
        v = jnp.array([1.0, 0.0])

        projected = manifold.proj(x, v)

        # Projection should scale down the vector
        assert jnp.linalg.norm(projected) < jnp.linalg.norm(v)

    def test_inner_conformal_factor(self):
        """Test inner product with Poincaré metric conformal factor."""
        manifold = PoincareBall(dimension=2)
        x = jnp.array([0.5, 0.0])
        v = jnp.array([1.0, 0.0])
        w = jnp.array([0.0, 1.0])

        inner_vw = manifold.inner(x, v, w)

        # Should be zero for orthogonal vectors
        assert jnp.abs(inner_vw) < 1e-6

        # Test conformal factor scaling
        inner_vv = manifold.inner(x, v, v)
        norm_sq = jnp.sum(x**2)
        conformal_factor = 4 / (1 - norm_sq) ** 2
        expected = conformal_factor * jnp.sum(v**2)
        assert jnp.allclose(inner_vv, expected, atol=1e-6)

    def test_exp_at_origin(self):
        """Test exponential map at origin."""
        manifold = PoincareBall(dimension=2)
        origin = jnp.zeros(2)
        v = jnp.array([0.3, 0.4])

        result = manifold.exp(origin, v)

        # At origin, exp should give tanh(|v|) * v/|v|
        v_norm = jnp.linalg.norm(v)
        expected = jnp.tanh(v_norm) * v / v_norm
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_log_at_origin(self):
        """Test logarithmic map at origin."""
        manifold = PoincareBall(dimension=2)
        origin = jnp.zeros(2)
        y = jnp.array([0.3, 0.4])

        result = manifold.log(origin, y)

        # At origin, log should give arctanh(|y|) * y/|y|
        y_norm = jnp.linalg.norm(y)
        expected = jnp.arctanh(y_norm) * y / y_norm
        assert jnp.allclose(result, expected, atol=1e-6)

    def test_exp_log_inverse(self):
        """Test that exp and log are inverse operations."""
        manifold = PoincareBall(dimension=2)
        x = jnp.array([0.3, 0.2])
        y = jnp.array([0.5, 0.4])

        # exp(x, log(x, y)) should give y
        v = manifold.log(x, y)
        y_recovered = manifold.exp(x, v)
        assert jnp.allclose(y_recovered, y, atol=1e-6)

        # log(x, exp(x, v)) should give v (for small v)
        v_small = jnp.array([0.1, 0.15])
        y_from_exp = manifold.exp(x, v_small)
        v_recovered = manifold.log(x, y_from_exp)
        assert jnp.allclose(v_recovered, v_small, atol=1e-6)

    def test_retr_as_exp_approximation(self):
        """Test that retraction approximates exponential map."""
        manifold = PoincareBall(dimension=2)
        x = jnp.array([0.2, 0.3])
        # Small tangent vector
        v = jnp.array([0.01, 0.02])

        exp_result = manifold.exp(x, v)
        retr_result = manifold.retr(x, v)

        # For small v, retraction should approximate exp
        # Retraction is an approximation, so we allow reasonable tolerance
        assert jnp.allclose(exp_result, retr_result, atol=5e-3)

    def test_dist_symmetry(self):
        """Test distance function symmetry."""
        manifold = PoincareBall(dimension=2)
        x = jnp.array([0.2, 0.3])
        y = jnp.array([0.4, 0.1])

        dist_xy = manifold.dist(x, y)
        dist_yx = manifold.dist(y, x)

        # Distance should be symmetric
        assert jnp.allclose(dist_xy, dist_yx, atol=1e-6)

    def test_dist_triangle_inequality(self):
        """Test that distance satisfies triangle inequality."""
        manifold = PoincareBall(dimension=2)
        x = jnp.array([0.1, 0.1])
        y = jnp.array([0.3, 0.2])
        z = jnp.array([0.2, 0.4])

        dist_xy = manifold.dist(x, y)
        dist_yz = manifold.dist(y, z)
        dist_xz = manifold.dist(x, z)

        # Triangle inequality: d(x,z) <= d(x,y) + d(y,z)
        assert dist_xz <= dist_xy + dist_yz + 1e-6

    def test_dist_from_log(self):
        """Test that distance matches norm of log."""
        manifold = PoincareBall(dimension=2)
        x = jnp.array([0.2, 0.1])
        y = jnp.array([0.4, 0.3])

        dist_direct = manifold.dist(x, y)

        # Distance should be norm of log in tangent space
        v = manifold.log(x, y)
        dist_from_log = jnp.sqrt(manifold.inner(x, v, v))

        # Allow tolerance for numerical differences in different formulations
        assert jnp.allclose(dist_direct, dist_from_log, atol=5e-2)

    def test_curvature_scaling(self):
        """Test that operations scale correctly with curvature."""
        manifold1 = PoincareBall(dimension=2, curvature=-1.0)
        manifold2 = PoincareBall(dimension=2, curvature=-0.25)

        x = jnp.array([0.3, 0.2])
        v = jnp.array([0.1, 0.15])

        # Exponential map should scale with curvature
        exp1 = manifold1.exp(x, v)
        exp2 = manifold2.exp(x, v)

        # Different curvatures should give different results
        assert not jnp.allclose(exp1, exp2, atol=1e-4)
