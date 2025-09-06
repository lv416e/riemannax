"""Tests for Lorentz manifold parallel transport operation.

Tests cover the parallel transport operation and its mathematical properties.
"""

import jax
import jax.numpy as jnp
import pytest

from riemannax.manifolds import Lorentz


class TestLorentzParallelTransport:
    """Test parallel transport operation."""

    def test_transp_same_point(self):
        """Test parallel transport when source and target are the same."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(42)

        x = manifold.random_point(key)
        v = manifold.random_tangent(key, x)

        result = manifold.transp(x, x, v)

        # Should return the same vector
        assert jnp.allclose(result, v, atol=1e-10)

    def test_transp_validates_result(self):
        """Test parallel transport always returns valid tangent vectors."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(42)

        for i in range(10):
            subkey1, subkey2, subkey3, key = jax.random.split(key, 4)
            x = manifold.random_point(subkey1)
            y = manifold.random_point(subkey2)
            v = manifold.random_tangent(subkey3, x)

            result = manifold.transp(x, y, v)

            # Result should be valid tangent vector at y
            assert manifold.validate_tangent(y, result)

    def test_transp_preserves_norm(self):
        """Test that parallel transport preserves vector norms."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(123)

        for i in range(5):
            subkey1, subkey2, subkey3, key = jax.random.split(key, 4)
            x = manifold.random_point(subkey1)
            y = manifold.random_point(subkey2)
            v = manifold.random_tangent(subkey3, x)

            # Original norm
            original_norm = jnp.sqrt(manifold.inner(x, v, v))

            # Transport
            transported = manifold.transp(x, y, v)

            # Transported norm
            transported_norm = jnp.sqrt(manifold.inner(y, transported, transported))

            # Norms should be preserved (up to numerical precision)
            assert jnp.allclose(original_norm, transported_norm, atol=1e-6)

    def test_transp_linearity(self):
        """Test linearity of parallel transport."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(456)

        x = manifold.random_point(key)
        y = manifold.random_point(key)
        u = manifold.random_tangent(key, x)
        v = manifold.random_tangent(key, x)

        a, b = 2.5, -1.3

        # Transport combination
        combined = manifold.transp(x, y, a * u + b * v)

        # Combination of transports
        separate = a * manifold.transp(x, y, u) + b * manifold.transp(x, y, v)

        assert jnp.allclose(combined, separate, atol=1e-10)

    def test_transp_inverse_property(self):
        """Test that transporting back returns original vector."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(789)

        x = manifold.random_point(key)
        y = manifold.random_point(key)
        v = manifold.random_tangent(key, x)

        # Transport x->y then y->x
        transported = manifold.transp(x, y, v)
        back_transported = manifold.transp(y, x, transported)

        # Should get back original vector (up to numerical precision)
        assert jnp.allclose(back_transported, v, atol=1e-6)

    def test_transp_geodesic_direction(self):
        """Test parallel transport of geodesic direction vector."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(101)

        x = manifold.random_point(key)
        y = manifold.random_point(key)

        # Direction from x to y
        v = manifold.log(x, y)

        # Transport this direction
        transported = manifold.transp(x, y, v)

        # Should be proportional to direction from y to x
        expected_direction = manifold.log(y, x)

        # Check if vectors are proportional (opposite directions)
        # Normalize both vectors
        v_norm = jnp.sqrt(manifold.inner(x, v, v))
        transported_norm = jnp.sqrt(manifold.inner(y, transported, transported))
        expected_norm = jnp.sqrt(manifold.inner(y, expected_direction, expected_direction))

        if v_norm > 1e-10 and transported_norm > 1e-10 and expected_norm > 1e-10:
            v_normalized = v / v_norm
            transported_normalized = transported / transported_norm
            expected_normalized = expected_direction / expected_norm

            # They should be opposite (negative proportional)
            dot_product = manifold.inner(y, transported_normalized, expected_normalized)
            assert jnp.allclose(jnp.abs(dot_product), 1.0, atol=1e-6)

    def test_transp_zero_vector(self):
        """Test parallel transport of zero vector."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(42)

        x = manifold.random_point(key)
        y = manifold.random_point(key)
        zero_v = jnp.zeros_like(x)

        result = manifold.transp(x, y, zero_v)

        # Should remain zero
        assert jnp.allclose(result, zero_v, atol=1e-10)

    def test_transp_batch_operations(self):
        """Test parallel transport works with batch dimensions."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(42)

        batch_size = 5
        x_batch = manifold.random_point(key, batch_size)
        y_batch = manifold.random_point(key, batch_size)
        v_batch = manifold.random_tangent(key, x_batch)

        # Batch transport
        transported_batch = manifold.transp(x_batch, y_batch, v_batch)

        # Check shapes
        assert transported_batch.shape == v_batch.shape

        # Check each transport individually
        for i in range(batch_size):
            individual_transport = manifold.transp(x_batch[i], y_batch[i], v_batch[i])
            assert jnp.allclose(transported_batch[i], individual_transport, atol=1e-10)

            # Each result should be valid
            assert manifold.validate_tangent(y_batch[i], transported_batch[i])


class TestLorentzParallelTransportProperties:
    """Test mathematical properties of parallel transport."""

    def test_transp_preserves_inner_products(self):
        """Test that parallel transport preserves inner products between vectors."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(42)

        x = manifold.random_point(key)
        y = manifold.random_point(key)
        u = manifold.random_tangent(key, x)
        v = manifold.random_tangent(key, x)

        # Original inner product
        original_inner = manifold.inner(x, u, v)

        # Transport both vectors
        u_transported = manifold.transp(x, y, u)
        v_transported = manifold.transp(x, y, v)

        # Inner product after transport
        transported_inner = manifold.inner(y, u_transported, v_transported)

        # Should be preserved
        assert jnp.allclose(original_inner, transported_inner, atol=1e-6)

    def test_transp_composition_property(self):
        """Test composition property of parallel transport."""
        manifold = Lorentz(dimension=2)
        key = jax.random.PRNGKey(123)

        x = manifold.random_point(key)
        y = manifold.random_point(key)
        z = manifold.random_point(key)
        v = manifold.random_tangent(key, x)

        # Direct transport x -> z
        direct_transport = manifold.transp(x, z, v)

        # Composed transport x -> y -> z
        intermediate = manifold.transp(x, y, v)
        composed_transport = manifold.transp(y, z, intermediate)

        # Should be approximately equal (within numerical precision)
        # Note: This property holds exactly for parallel transport along geodesics
        assert jnp.allclose(direct_transport, composed_transport, atol=1e-5)
